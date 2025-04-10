'''
SimpleTuner RunPod Serverless Handler
'''
import os
import sys
import json
import time
import shutil
import logging
import subprocess
import traceback
from pathlib import Path
import runpod
from runpod.serverless.utils import download_files_from_urls, upload_file_to_signed_url
import requests
import psutil
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('SimpleTuner-Handler')

# Define working directories
WORKSPACE_DIR = "/workspace"
INPUT_DIR = f"{WORKSPACE_DIR}/inputs"
OUTPUT_DIR = f"{WORKSPACE_DIR}/outputs"
DATASET_DIR = f"{WORKSPACE_DIR}/dataset"
CONFIG_DIR = f"{WORKSPACE_DIR}/config"
LOG_DIR = f"{WORKSPACE_DIR}/logs"
MODEL_DIR = f"{WORKSPACE_DIR}/models"

# Ensure directories exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Activation of SimpleTuner's virtual environment
SIMPLETUNER_DIR = "/SimpleTuner"
VENV_PATH = f"{SIMPLETUNER_DIR}/.venv/bin/activate"

def init():
    '''
    Initialize the environment
    This function is called once when the container starts
    '''
    logger.info("Initializing SimpleTuner environment...")
    
    # Check if CUDA is available
    cuda_check = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
    if cuda_check.returncode != 0:
        logger.error("CUDA not available: %s", cuda_check.stderr)
        sys.exit(1)
    
    logger.info("GPU information:\n%s", cuda_check.stdout)
    
    # Ensure SimpleTuner is installed and accessible
    simpletuner_check = subprocess.run(
        f"source {VENV_PATH} && python -c 'import helpers'", 
        shell=True, 
        capture_output=True, 
        text=True
    )
    if simpletuner_check.returncode != 0:
        logger.error("SimpleTuner not properly installed: %s", simpletuner_check.stderr)
        sys.exit(1)
    
    logger.info("SimpleTuner environment initialized successfully")
    return True

def download_dataset(dataset_url, extract=True):
    """
    Download dataset from URL and extract if needed
    """
    logger.info(f"Downloading dataset from {dataset_url}")
    
    # Clear dataset directory first
    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    # Determine file type and download path
    file_extension = os.path.splitext(dataset_url.split('?')[0])[1].lower()
    download_path = os.path.join(INPUT_DIR, f"dataset{file_extension}")
    
    try:
        # Use requests with progress bar for direct URLs
        if dataset_url.startswith(('http://', 'https://')):
            with requests.get(dataset_url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                
                with open(download_path, 'wb') as f, tqdm(
                    desc="Downloading dataset",
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for chunk in r.iter_content(chunk_size=8192):
                        size = f.write(chunk)
                        bar.update(size)
        else:
            # Use runpod utility for other URL types (S3 signed, etc.)
            download_files_from_urls(urls=[dataset_url], destination_directory=INPUT_DIR)
            # Find the downloaded file
            for file in os.listdir(INPUT_DIR):
                if os.path.isfile(os.path.join(INPUT_DIR, file)) and file != "config.json" and file != "dataloader.json":
                    download_path = os.path.join(INPUT_DIR, file)
                    break
        
        logger.info(f"Dataset downloaded to {download_path}")
        
        # Extract if needed
        if extract and file_extension in ['.zip', '.tar', '.gz', '.tar.gz', '.tgz']:
            logger.info(f"Extracting dataset from {download_path}")
            
            if file_extension == '.zip':
                subprocess.run(['unzip', '-q', download_path, '-d', DATASET_DIR], check=True)
            elif file_extension in ['.tar', '.tar.gz', '.tgz', '.gz']:
                subprocess.run(['tar', '-xf', download_path, '-C', DATASET_DIR], check=True)
            
            logger.info(f"Dataset extracted to {DATASET_DIR}")
            return DATASET_DIR
        
        # If it's not an archive, just move the file to dataset dir
        if not extract:
            if not os.path.exists(DATASET_DIR):
                os.makedirs(DATASET_DIR)
            dest_path = os.path.join(DATASET_DIR, os.path.basename(download_path))
            shutil.move(download_path, dest_path)
            return dest_path
        
        return download_path
    
    except Exception as e:
        logger.error(f"Error downloading or extracting dataset: {e}")
        logger.error(traceback.format_exc())
        return None

def save_config_file(config_data, filename):
    """
    Save configuration data to a file
    """
    config_path = os.path.join(CONFIG_DIR, filename)
    try:
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        logger.info(f"Saved {filename} to {config_path}")
        return config_path
    except Exception as e:
        logger.error(f"Error saving {filename}: {e}")
        return None

def prepare_training_args(config, dataloader_config, dataset_path):
    """
    Prepare CLI arguments for SimpleTuner's train.py
    """
    args = []
    
    # Add dataset path
    args.extend(["--train_data_dir", dataset_path])
    
    # Add output directory
    args.extend(["--output_dir", OUTPUT_DIR])
    
    # Process all config parameters
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                args.append(f"--{key}")
        elif isinstance(value, (str, int, float)) and value:
            args.extend([f"--{key}", str(value)])
        elif isinstance(value, list) and value:
            for item in value:
                args.extend([f"--{key}", str(item)])
    
    # Add dataloader configuration parameters
    if dataloader_config:
        dataloader_path = os.path.join(CONFIG_DIR, "dataloader.json")
        args.extend(["--dataloader_config", dataloader_path])
    
    return args

def run_training(args):
    """
    Run SimpleTuner training with the specified arguments
    """
    cmd = [
        "bash", "-c",
        f"cd {SIMPLETUNER_DIR} && "
        f"source {VENV_PATH} && "
        f"python train.py " + " ".join([str(arg) for arg in args])
    ]
    
    logger.info(f"Starting training with command: {' '.join(cmd)}")
    
    # Create log file
    log_file_path = os.path.join(LOG_DIR, f"training_{int(time.time())}.log")
    log_file = open(log_file_path, "w")
    
    # Start training process
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Monitor training process and gather output
    logs = []
    try:
        for line in process.stdout:
            line = line.rstrip()
            logs.append(line)
            log_file.write(line + "\n")
            log_file.flush()
            logger.info(line)
    except Exception as e:
        logger.error(f"Error monitoring training process: {e}")
    finally:
        process.wait()
        log_file.close()
    
    # Check if training was successful
    if process.returncode != 0:
        logger.error(f"Training failed with return code {process.returncode}")
        return False, logs, log_file_path
    
    logger.info("Training completed successfully")
    return True, logs, log_file_path

def collect_output_files():
    """
    Collect output files and prepare them for return
    """
    output_files = {}
    
    # Walk through the output directory
    for root, _, files in os.walk(OUTPUT_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, OUTPUT_DIR)
            output_files[rel_path] = file_path
    
    return output_files

def upload_results(output_files, signed_urls=None):
    """
    Upload results to storage provider
    If signed_urls are provided, use them for upload
    Otherwise, compress results and return base64 or file path
    """
    result_urls = {}
    
    if signed_urls and isinstance(signed_urls, dict):
        # Upload to provided signed URLs
        for file_key, file_path in output_files.items():
            if file_key in signed_urls:
                try:
                    upload_file_to_signed_url(file_path, signed_urls[file_key])
                    result_urls[file_key] = signed_urls[file_key].split('?')[0]  # Remove signature part
                except Exception as e:
                    logger.error(f"Failed to upload {file_key}: {e}")
    else:
        # Create a zip file with all results
        zip_path = os.path.join(WORKSPACE_DIR, "results.zip")
        subprocess.run(
            ["zip", "-r", zip_path, "."], 
            cwd=OUTPUT_DIR, 
            check=True,
            capture_output=True
        )
        result_urls["results.zip"] = zip_path
    
    return result_urls

def handler(event):
    """
    Main handler function for RunPod serverless
    """
    logger.info(f"Received event: {event}")
    
    try:
        input_data = event.get("input", {})
        
        # Get dataset URL
        dataset_url = input_data.get("dataset_url")
        if not dataset_url:
            return {
                "error": "No dataset_url provided in the input"
            }
        
        # Get configurations
        config = input_data.get("config", {})
        dataloader_config = input_data.get("dataloader", {})
        
        # Download dataset
        dataset_path = download_dataset(dataset_url)
        if not dataset_path:
            return {
                "error": "Failed to download or extract dataset"
            }
        
        # Save configurations to files
        config_path = save_config_file(config, "config.json")
        dataloader_path = None
        if dataloader_config:
            dataloader_path = save_config_file(dataloader_config, "dataloader.json")
        
        # Prepare training arguments
        training_args = prepare_training_args(config, dataloader_config, dataset_path)
        
        # Run training
        success, logs, log_file_path = run_training(training_args)
        
        # Collect output files
        output_files = collect_output_files()
        
        # Add log file to output files
        rel_log_path = os.path.basename(log_file_path)
        output_files[rel_log_path] = log_file_path
        
        # Upload results
        signed_urls = input_data.get("signed_urls", {})
        result_urls = upload_results(output_files, signed_urls)
        
        # Prepare response
        response = {
            "status": "success" if success else "error",
            "output": {
                "files": result_urls,
                "log_summary": "\n".join(logs[-20:]) if logs else "No logs available"
            }
        }
        
        return response
    
    except Exception as e:
        logger.error(f"Error processing job: {e}")
        logger.error(traceback.format_exc())
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# Initialize the handler
init()

# Start the RunPod handler
runpod.serverless.start({"handler": handler}) 