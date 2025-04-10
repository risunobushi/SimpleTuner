# SimpleTuner RunPod Client

This client allows you to start SimpleTuner training jobs on a RunPod serverless endpoint directly from your local machine.

## Requirements

- Python 3.7+
- RunPod API key
- RunPod endpoint ID for your deployed SimpleTuner serverless instance

## Installation

1. Install required packages:

```bash
pip install -r requirements-client.txt
```

## Usage

The client script uploads your local dataset and configuration files to RunPod, starts the training job, monitors progress, and downloads results when complete.

### Basic Usage

```bash
python run_remote_training.py \
    --api-key YOUR_RUNPOD_API_KEY \
    --endpoint-id YOUR_ENDPOINT_ID \
    --dataset ./path/to/dataset.zip \
    --config ./path/to/config.json \
    --dataloader ./path/to/dataloader.json
```

### Arguments

- `--api-key`: Your RunPod API key (required)
- `--endpoint-id`: Your RunPod endpoint ID (required)
- `--dataset`: Path to your dataset file/folder (required)
- `--config`: Path to your config.json file (required)
- `--dataloader`: Path to your dataloader.json file (optional)
- `--output-dir`: Directory to save results (default: ./results)
- `--poll-interval`: Interval in seconds to check job status (default: 10)

### Example config.json

```json
{
  "model_name": "stabilityai/stable-diffusion-xl-base-1.0",
  "resolution": 1024,
  "train_batch_size": 1,
  "num_train_epochs": 10,
  "checkpointing_steps": 500,
  "learning_rate": 1e-5
}
```

### Example dataloader.json

```json
{
  "caption_extension": ".txt",
  "shuffle_tags": true,
  "keep_tokens": 1
}
```

## Running with a Directory of Images

If your dataset is a directory of images with captions, you'll need to zip it first:

```bash
# From the directory containing your images
zip -r ../my_dataset.zip ./*

# Then run the training
python run_remote_training.py \
    --api-key YOUR_API_KEY \
    --endpoint-id YOUR_ENDPOINT_ID \
    --dataset ../my_dataset.zip \
    --config ./config.json
```

## Monitoring Progress

The script automatically monitors the job and displays:
- Upload progress
- Training job status
- Download progress for results

## Results

After successful completion, all training outputs will be downloaded to the specified output directory (default: `./results`), including:
- Model checkpoints
- Training logs
- Configuration files 