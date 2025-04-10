#!/bin/bash

# Activate the virtual environment
source /SimpleTuner/.venv/bin/activate

# Export environment variables
# HF_TOKEN will be set from RunPod secrets if configured
if [[ -n "${HF_TOKEN}" ]]; then
    huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential
fi

# Log start message
echo "Starting SimpleTuner RunPod serverless handler..."

# Start the handler
cd /
python -m runpod.serverless.start 