FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Installation
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    unzip \
    pkg-config \
    libx11-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgtk2.0-dev \
    libcanberra-gtk* \
    python3.10 \
    python3-pip \
    python3-dev \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Link python → python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# Update pip
RUN pip install --upgrade pip

# PyTorch + TorchVision + Torchaudio с CUDA 11.8
RUN pip install torch==2.6.0+cu118 torchvision==0.21.0+cu118 torchaudio==2.6.0+cu118 \
    -f https://download.pytorch.org/whl/torch_stable.html

# TensorFlow (GPU)
RUN pip install tensorflow==2.15.0 tensorflow-io-gcs-filesystem

# Getting new libs
RUN pip install \
    scikit-learn \
    mtcnn \
    opencv-python==4.11.0.86 \
    opencv-python-headless==4.11.0.86 \
    dlib==19.24.6 \
    numpy==2.0.2 \
    pandas \
    tqdm \
    rich \
    matplotlib \
    pillow \
    jinja2 \
    namex \
    sympy \
    scipy

# Copy project into container
COPY . /app
WORKDIR /app

CMD ["/bin/bash"]