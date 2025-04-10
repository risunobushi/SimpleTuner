# SimpleTuner RunPod Serverless Implementation

This directory contains the implementation for deploying SimpleTuner as a RunPod serverless instance, allowing for automated training jobs through the RunPod platform.

## Overview

The implementation consists of two main components:

1. **Docker Image Setup**: A specialized Docker image configured for RunPod serverless deployment via GitHub integration
2. **Serverless API Handler**: An API handler that processes incoming requests, manages training jobs, and returns results

## Implementation Plan

### 1. Docker Image Setup

The Docker image will be built from the following files:

- `Dockerfile`: Defines the container environment with all required dependencies
- `handler.py`: RunPod serverless handler that processes incoming job requests
- `requirements.txt`: Python dependencies specific to the serverless implementation
- `start.sh`: Initialization script for the container

#### Dockerfile Strategy

The Dockerfile will:

1. Use NVIDIA CUDA base image (CUDA 12.4+)
2. Install system dependencies
3. Set up Python environment
4. Install SimpleTuner and its dependencies
5. Set up the RunPod handler

### 2. Serverless Handler Implementation

The serverless handler will:

1. Accept incoming jobs with:
   - Dataset (URL or uploaded files)
   - Configuration file (config.json)
   - Dataloader configuration (dataloader.json)
2. Run the training process
3. Monitor training progress
4. Return the output directory contents upon completion

## File Structure

```
simpletuner_docker/
├── Dockerfile
├── handler.py
├── requirements.txt
├── start.sh
└── README.md
```

## Implementation Details

### handler.py

The handler will implement the following functions:

1. `init()`: Initialize the environment, load models, etc.
2. `handler(event)`: Process incoming job requests
   - Parse input parameters
   - Download/extract dataset
   - Set up configuration files
   - Launch training process
   - Monitor training progress
   - Return training outputs

The handler will use the runpod Python library to handle serverless execution:

```python
import runpod
from runpod.serverless.utils import download_files_from_urls, upload_file_to_signed_url

def handler(event):
    # Process input job - extract dataset, config.json, dataloader.json
    # Run training
    # Return results
```

### Data Flow

1. User submits a job to the RunPod endpoint with:
   - Dataset URL (S3, HTTP, etc.) or direct file upload
   - config.json (training configuration)
   - dataloader.json (dataset configuration)

2. Handler processes the job:
   - Downloads and extracts dataset
   - Prepares configuration
   - Launches training
   - Monitors progress

3. Handler returns:
   - Status (success/failure)
   - Output directory contents (model checkpoints, logs, etc.)
   - Performance metrics

## GitHub Integration

This implementation will leverage RunPod's GitHub integration for deployment:

1. Repository will be connected to RunPod
2. On push to the designated branch, RunPod will:
   - Pull the code
   - Build the Docker image
   - Deploy it to the serverless endpoint

## Usage Example

After deployment, users can submit jobs through:

1. RunPod API
2. RunPod web interface
3. Custom client applications

Example API request:
```json
{
  "input": {
    "dataset_url": "https://example.com/dataset.zip",
    "config": {
      "model_name": "black-forest-labs/FLUX.1-dev",
      "resolution": 1024,
      "train_batch_size": 1,
      "num_train_epochs": 10,
      "checkpointing_steps": 500,
      "learning_rate": 1e-5
    },
    "dataloader": {
      "caption_extension": ".txt",
      "shuffle_tags": true,
      "keep_tokens": 1
    }
  }
}
```

Example response:
```json
{
  "id": "job-123456",
  "status": "COMPLETED",
  "output": {
    "training_logs": "https://storage.example.com/logs.txt",
    "model_checkpoint": "https://storage.example.com/checkpoint.safetensors",
    "metrics": {
      "loss": 0.0023,
      "elapsed_time": "2h 15m"
    }
  }
}
```

## Next Steps

1. Create the Dockerfile with all required dependencies
2. Implement the RunPod handler
3. Test the implementation locally
4. Deploy to RunPod via GitHub integration
5. Create documentation for users 