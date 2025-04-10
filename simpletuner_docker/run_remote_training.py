#!/usr/bin/env python
"""
RunPod SimpleTuner Remote Training Client

This script allows users to start a SimpleTuner training job on a RunPod serverless
endpoint using local files.

Usage:
    python run_remote_training.py \
        --api-key YOUR_RUNPOD_API_KEY \
        --endpoint-id YOUR_ENDPOINT_ID \
        --dataset ./path/to/dataset.zip \
        --config ./path/to/config.json \
        --dataloader ./path/to/dataloader.json \
        [--output-dir ./path/to/save/results]
"""

import os
import sys
import json
import time
import argparse
import requests
from tqdm import tqdm
import runpod

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="RunPod SimpleTuner Remote Training Client")
    
    parser.add_argument("--api-key", required=True, help="Your RunPod API key")
    parser.add_argument("--endpoint-id", required=True, help="Your RunPod endpoint ID")
    parser.add_argument("--dataset", required=True, help="Path to your dataset file/folder")
    parser.add_argument("--config", required=True, help="Path to your config.json file")
    parser.add_argument("--dataloader", required=False, help="Path to your dataloader.json file")
    parser.add_argument("--output-dir", default="./results", help="Directory to save results")
    parser.add_argument("--poll-interval", type=int, default=10, help="Interval in seconds to check job status")
    
    return parser.parse_args()

def upload_file_to_runpod(file_path, api_key):
    """Upload a file to RunPod's temporary storage"""
    print(f"Uploading {file_path} to RunPod temporary storage...")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found")
        return None
    
    # Create upload session
    session_response = requests.post(
        "https://api.runpod.io/v2/upload/createSession",
        headers={"Authorization": f"Bearer {api_key}"}
    )
    
    if session_response.status_code != 200:
        print(f"Error creating upload session: {session_response.text}")
        return None
    
    session_data = session_response.json()
    session_id = session_data.get("id")
    
    # Prepare file for upload
    file_size = os.path.getsize(file_path)
    file_name = os.path.basename(file_path)
    
    # Upload file
    with open(file_path, 'rb') as file, tqdm(
        desc=f"Uploading {file_name}",
        total=file_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        upload_response = requests.post(
            f"https://api.runpod.io/v2/upload/{session_id}",
            headers={"Authorization": f"Bearer {api_key}"},
            data={
                "filename": file_name,
            },
            files={
                "file": file
            }
        )
        
        # Update progress bar (after upload since RunPod doesn't support streaming upload progress)
        progress_bar.update(file_size)
    
    if upload_response.status_code != 200:
        print(f"Error uploading file: {upload_response.text}")
        return None
    
    # Get file URL
    url_response = requests.get(
        f"https://api.runpod.io/v2/upload/{session_id}/{file_name}",
        headers={"Authorization": f"Bearer {api_key}"}
    )
    
    if url_response.status_code != 200:
        print(f"Error getting file URL: {url_response.text}")
        return None
    
    url_data = url_response.json()
    file_url = url_data.get("url")
    
    print(f"Upload successful: {file_url}")
    return file_url

def load_json_file(file_path):
    """Load and validate a JSON file"""
    try:
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} not found")
            return None
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError:
        print(f"Error: {file_path} is not a valid JSON file")
        return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def download_results(urls, output_dir):
    """Download result files from URLs"""
    os.makedirs(output_dir, exist_ok=True)
    
    for file_name, url in urls.items():
        output_path = os.path.join(output_dir, file_name)
        
        # Create directory for file if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f"Downloading {file_name} to {output_path}...")
        
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            print(f"Error downloading {file_name}: {response.text}")
            continue
        
        # Get file size for progress bar
        total_size = int(response.headers.get('content-length', 0))
        
        # Download with progress bar
        with open(output_path, 'wb') as f, tqdm(
            desc=f"Downloading {file_name}",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                bar.update(size)
        
        print(f"Downloaded {file_name} to {output_path}")

def main():
    """Main function"""
    args = parse_args()
    
    # Configure RunPod API
    runpod.api_key = args.api_key
    
    # Load config and dataloader files
    config = load_json_file(args.config)
    if not config:
        return
    
    dataloader = None
    if args.dataloader:
        dataloader = load_json_file(args.dataloader)
        if not dataloader and args.dataloader:
            return
    
    # Upload dataset
    dataset_url = upload_file_to_runpod(args.dataset, args.api_key)
    if not dataset_url:
        return
    
    # Prepare job input
    job_input = {
        "dataset_url": dataset_url,
        "config": config
    }
    
    if dataloader:
        job_input["dataloader"] = dataloader
    
    # Run the job
    print(f"Starting training job on endpoint {args.endpoint_id}...")
    try:
        response = runpod.run_job(args.endpoint_id, job_input)
        job_id = response.get("id")
        
        if not job_id:
            print("Error: No job ID returned")
            return
        
        print(f"Job started with ID: {job_id}")
        
        # Poll for job status
        print(f"Monitoring job status (polling every {args.poll_interval} seconds)...")
        
        while True:
            status_response = runpod.get_job(job_id)
            status = status_response.get("status")
            
            if status == "COMPLETED":
                print("\nJob completed successfully!")
                
                # Get output files
                output = status_response.get("output", {})
                files = output.get("output", {}).get("files", {})
                
                if files:
                    print(f"Downloading result files to {args.output_dir}...")
                    download_results(files, args.output_dir)
                    print(f"All files downloaded to {args.output_dir}")
                else:
                    print("No output files found")
                
                # Print log summary if available
                log_summary = output.get("output", {}).get("log_summary")
                if log_summary:
                    print("\nTraining log summary:")
                    print(log_summary)
                
                break
                
            elif status in ["FAILED", "CANCELLED"]:
                print(f"\nJob {status.lower()}")
                
                # Print error if available
                error = status_response.get("output", {}).get("error")
                if error:
                    print(f"Error: {error}")
                
                break
                
            else:
                # Still running or queued
                print(f"Job status: {status}", end="\r")
                time.sleep(args.poll_interval)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 