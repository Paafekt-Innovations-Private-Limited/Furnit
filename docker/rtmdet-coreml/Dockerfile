FROM --platform=linux/amd64 python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/openmmlab

RUN git clone --depth 1 --branch main https://github.com/open-mmlab/mmdetection.git \
    && git clone --depth 1 --branch main https://github.com/open-mmlab/mmdeploy.git

RUN python -m pip install --upgrade pip "setuptools<81" wheel \
    && python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install "numpy<2" \
    && python -m pip install -U openmim \
    && mim install "mmcv==2.1.0" "mmdet==3.3.0" "mmengine==0.10.3" \
    && python -m pip install "mmdeploy==1.3.1" coremltools==7.1

WORKDIR /work
