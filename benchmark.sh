#!/bin/bash
# ACT Model C++ Benchmark Script
# Usage: ./benchmark.sh [OPTIONS]

set -e

# Show help if no arguments provided
if [ $# -eq 0 ]; then
    echo "Error: No arguments provided"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --model PATH           ONNX model path (default: output/act_model.onnx)"
    echo "  --device DEVICE        Device to use: CPU or GPU (default: CPU)"
    echo "  --iterations NUM       Number of iterations (default: 100)"
    echo "  --build-only           Only build the C++ project, don't run benchmark"
    echo "  --help                 Show this help message"
    exit 1
fi

# Default values
MODEL_PATH="$(pwd)/output/act_model.onnx"
BUILD_DIR="$(pwd)/cpp/build"
DOCKER_IMAGE="act-inference:latest"
DEVICE="CPU"
NUM_ITERATIONS="100"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --iterations)
            NUM_ITERATIONS="$2"
            shift 2
            ;;
        --build-only)
            BUILD_ONLY=1
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model PATH           ONNX model path (default: output/act_model.onnx)"
            echo "  --device DEVICE        Device to use: CPU or GPU (default: CPU)"
            echo "  --iterations NUM       Number of iterations (default: 100)"
            echo "  --build-only           Only build the C++ project, don't run benchmark"
            echo "  --help                 Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Convert to absolute paths
if [[ ! "$MODEL_PATH" = /* ]]; then
    MODEL_PATH="$(pwd)/$MODEL_PATH"
fi
if [[ ! "$BUILD_DIR" = /* ]]; then
    BUILD_DIR="$(pwd)/$BUILD_DIR"
fi

# Get the directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPP_DIR="$SCRIPT_DIR/cpp"

# Validate paths
if [ ! -d "$CPP_DIR" ]; then
    echo "Error: C++ source directory not found: $CPP_DIR"
    exit 1
fi

# Create build directory if it doesn't exist
mkdir -p "$BUILD_DIR"

docker build -t "$DOCKER_IMAGE" --target cpp-inference "$SCRIPT_DIR"

echo "=========================================="
echo "C++ Project Build"
echo "=========================================="
echo "Source: $CPP_DIR"
echo "Build: $BUILD_DIR"
echo "=========================================="
echo ""

# Build the C++ project
docker run --rm \
    -v "$CPP_DIR:/workspace/cpp:ro" \
    -v "$BUILD_DIR:/workspace/cpp/build" \
    -w /workspace/cpp/build \
    "$DOCKER_IMAGE" \
    bash -c "cmake .. && make"

echo ""
echo "=========================================="
echo "Build completed!"
echo "=========================================="

# Exit if build-only flag is set
if [ -n "$BUILD_ONLY" ]; then
    exit 0
fi

# Validate model file for benchmark
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found: $MODEL_PATH"
    echo "Please export the model first using export_model.sh"
    exit 1
fi

# Get model directory and filename
MODEL_DIR="$(dirname "$MODEL_PATH")"
MODEL_FILE="$(basename "$MODEL_PATH")"

echo ""
echo "=========================================="
echo "OpenVINO Benchmark"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Device: $DEVICE"
echo "Iterations: $NUM_ITERATIONS"
echo "=========================================="
echo ""

# Run benchmark
docker run --rm \
    -v "$BUILD_DIR:/workspace/cpp/build:ro" \
    -v "$MODEL_DIR:/workspace/models:ro" \
    -w /workspace/cpp/build \
    "$DOCKER_IMAGE" \
    ./benchmark_cpu_gpu \
        "/workspace/models/$MODEL_FILE" \
        "$DEVICE" \
        "$NUM_ITERATIONS"

echo ""
echo "=========================================="
echo "Benchmark completed!"
echo "=========================================="
