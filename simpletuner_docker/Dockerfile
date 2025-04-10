# SimpleTuner RunPod Serverless Image
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Set working directory
WORKDIR /

# Update apt-get
RUN apt-get update -y

# Prevents different commands from being stuck by waiting
# on user input during build
ENV DEBIAN_FRONTEND noninteractive

# Install libg dependencies
RUN apt install libgl1-mesa-glx -y
RUN apt-get install 'ffmpeg'\
    'libsm6'\
    'libxext6'  -y

# Install misc unix libraries
RUN apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    wget \
    curl \
    zip \
    unzip \
    python3 \
    python3-pip \
    python3.10-venv

# Set up git to support LFS
RUN git lfs install

# Python
RUN python3 -m pip install pip --upgrade

# HF
ENV HF_HOME=/workspace/huggingface

# Clone SimpleTuner
RUN git clone https://github.com/bghira/SimpleTuner --branch release /SimpleTuner

# Install SimpleTuner and dependencies
RUN pip3 install poetry
WORKDIR /SimpleTuner
RUN python3 -m venv .venv && \
    . .venv/bin/activate && \
    poetry install --no-root

# Install RunPod serverless requirements
COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

# Create working directories
RUN mkdir -p /workspace/inputs && \
    mkdir -p /workspace/outputs

# Copy handler and startup script
COPY handler.py /handler.py
COPY start.sh /start.sh

# Make startup script executable
RUN chmod +x /start.sh

# Set default environment variables
ENV PYTHONPATH="${PYTHONPATH}:/SimpleTuner"
ENV PYTHONUNBUFFERED=1

# Default command
CMD [ "/start.sh" ] 