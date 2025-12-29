FROM python:3.10-slim AS python-base

WORKDIR /workspace

RUN pip install --no-cache-dir \
    torch==2.1.0 torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    onnx==1.15.0 \
    onnxconverter-common \
    safetensors==0.4.1 \
    numpy==1.24.3 \
    einops==0.7.0

FROM openvino/ubuntu20_runtime:latest AS cpp-inference

USER root

WORKDIR /workspace

RUN apt update && apt install -y \
    build-essential \
    cmake \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/cpp/build

CMD ["/bin/bash"]
