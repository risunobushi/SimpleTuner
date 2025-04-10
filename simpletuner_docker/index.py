# RunPod Serverless API Index
# This is a placeholder file required by RunPod GitHub integration
# The actual implementation is in handler.py

import runpod
from handler import handler

# Start the RunPod serverless API
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler}) 