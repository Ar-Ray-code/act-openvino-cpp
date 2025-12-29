#!/bin/bash
# ACT Model ONNX Exporter Script
# Usage: ./export_model.sh [OPTIONS]

set -e

# Show help if no arguments provided
if [ $# -eq 0 ]; then
    echo "Error: No arguments provided"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --checkpoint DIR        Checkpoint directory (default: models/checkpoints/030000/pretrained_model)"
    echo "  --output-dir DIR        Output directory (default: conversion/output)"
    echo "  --output-file FILE      Output ONNX filename (default: act_model.onnx)"
    echo "  --fp16                  Export in FP16 precision"
    echo "  --opset-version VER     ONNX opset version (default: 14)"
    echo "  --help                  Show this help message"
    exit 1
fi

# Default values
CHECKPOINT_DIR="$(pwd)/models/checkpoints/030000/pretrained_model"
OUTPUT_DIR="$(pwd)/conversion/output"
OUTPUT_FILE="act_model.onnx"
FP16=""
OPSET_VERSION="14"
DOCKER_IMAGE="act-conversion:latest"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --output-file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --fp16)
            FP16="--fp16"
            shift
            ;;
        --opset-version)
            OPSET_VERSION="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --checkpoint DIR        Checkpoint directory (default: models/checkpoints/030000/pretrained_model)"
            echo "  --output-dir DIR        Output directory (default: conversion/output)"
            echo "  --output-file FILE      Output ONNX filename (default: act_model.onnx)"
            echo "  --fp16                  Export in FP16 precision"
            echo "  --opset-version VER     ONNX opset version (default: 14)"
            echo "  --help                  Show this help message"
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
if [[ ! "$CHECKPOINT_DIR" = /* ]]; then
    CHECKPOINT_DIR="$(pwd)/$CHECKPOINT_DIR"
fi
if [[ ! "$OUTPUT_DIR" = /* ]]; then
    OUTPUT_DIR="$(pwd)/$OUTPUT_DIR"
fi

# Validate checkpoint directory
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Error: Checkpoint directory not found: $CHECKPOINT_DIR"
    exit 1
fi

if [ ! -f "$CHECKPOINT_DIR/config.json" ]; then
    echo "Error: config.json not found in checkpoint directory: $CHECKPOINT_DIR"
    exit 1
fi

if [ ! -f "$CHECKPOINT_DIR/model.safetensors" ]; then
    echo "Error: model.safetensors not found in checkpoint directory: $CHECKPOINT_DIR"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Get the directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONVERSION_DIR="$SCRIPT_DIR/"

# Check if conversion scripts exist
if [ ! -d "$CONVERSION_DIR/" ]; then
    echo "Error: Conversion scripts not found at: $CONVERSION_DIR/"
    exit 1
fi

docker build -t "$DOCKER_IMAGE" --target python-base "$SCRIPT_DIR"

# Display configuration
echo "=========================================="
echo "ACT Model ONNX Export Configuration"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Output Dir: $OUTPUT_DIR"
echo "Output File: $OUTPUT_FILE"
echo "FP16: ${FP16:-disabled}"
echo "Opset Version: $OPSET_VERSION"
echo "=========================================="
echo ""

# Run Docker container
docker run --rm \
    -v "$CHECKPOINT_DIR:/workspace/checkpoint:ro" \
    -v "$OUTPUT_DIR:/workspace/output" \
    -v "$CONVERSION_DIR:/workspace/conversion:ro" \
    -w /workspace \
    "$DOCKER_IMAGE" \
    python conversion/scripts/export_standalone.py \
        --checkpoint /workspace/checkpoint \
        --output "/workspace/output/$OUTPUT_FILE" \
        --opset-version "$OPSET_VERSION" \
        $FP16

echo ""
echo "=========================================="
echo "Export completed!"
echo "Output: $OUTPUT_DIR/$OUTPUT_FILE"
echo "=========================================="
