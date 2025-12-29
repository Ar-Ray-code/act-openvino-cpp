#!/bin/bash
# ONNX to OpenVINO IR Converter Script
# Usage: ./convert_to_openvino.sh [OPTIONS]

set -e

# Show help if no arguments provided
if [ $# -eq 0 ]; then
    echo "Error: No arguments provided"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --input PATH            Input ONNX model path (required)"
    echo "  --output-dir DIR        Output directory (default: same as input)"
    echo "  --fp16                  Compress model to FP16"
    echo "  --help                  Show this help message"
    exit 1
fi

# Default values
INPUT_MODEL=""
OUTPUT_DIR=""
FP16=""
DOCKER_IMAGE="act-inference:latest"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input)
            INPUT_MODEL="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --fp16)
            FP16="--fp16"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Convert ONNX model to OpenVINO IR format"
            echo ""
            echo "Options:"
            echo "  --input PATH            Input ONNX model path (required)"
            echo "  --output-dir DIR        Output directory (default: same as input)"
            echo "  --fp16                  Compress model to FP16"
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

# Validate input
if [ -z "$INPUT_MODEL" ]; then
    echo "Error: --input is required"
    echo "Use --help for usage information"
    exit 1
fi

# Convert to absolute paths
if [[ ! "$INPUT_MODEL" = /* ]]; then
    INPUT_MODEL="$(pwd)/$INPUT_MODEL"
fi

# Set output directory
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="$(dirname "$INPUT_MODEL")"
fi

if [[ ! "$OUTPUT_DIR" = /* ]]; then
    OUTPUT_DIR="$(pwd)/$OUTPUT_DIR"
fi

# Validate input file
if [ ! -f "$INPUT_MODEL" ]; then
    echo "Error: Input model not found: $INPUT_MODEL"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Get model directory and filename
MODEL_DIR="$(dirname "$INPUT_MODEL")"
MODEL_FILE="$(basename "$INPUT_MODEL")"
MODEL_NAME="${MODEL_FILE%.onnx}"

# Build Docker image if it doesn't exist
if ! docker image inspect "$DOCKER_IMAGE" > /dev/null 2>&1; then
    echo "Docker image '$DOCKER_IMAGE' not found. Building..."
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    docker build -t "$DOCKER_IMAGE" --target cpp-inference "$SCRIPT_DIR"
fi

echo "=========================================="
echo "ONNX to OpenVINO IR Conversion"
echo "=========================================="
echo "Input: $INPUT_MODEL"
echo "Output Dir: $OUTPUT_DIR"
echo "FP16: ${FP16:-disabled}"
echo "=========================================="
echo ""

# Get the directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONVERSION_SCRIPT="$SCRIPT_DIR/scripts/convert_onnx_to_openvino.py"

# Check if conversion script exists
if [ ! -f "$CONVERSION_SCRIPT" ]; then
    echo "Error: Conversion script not found: $CONVERSION_SCRIPT"
    exit 1
fi

# Run conversion using OpenVINO Python API
docker run --rm \
    -v "$MODEL_DIR:/workspace/input:ro" \
    -v "$OUTPUT_DIR:/workspace/output" \
    -v "$CONVERSION_SCRIPT:/workspace/convert.py:ro" \
    -w /workspace \
    "$DOCKER_IMAGE" \
    python3 /workspace/convert.py \
        --input "/workspace/input/$MODEL_FILE" \
        --output "/workspace/output/${MODEL_NAME}.xml" \
        $FP16

echo ""
echo "=========================================="
echo "Conversion completed!"
echo "Output files:"
echo "  ${OUTPUT_DIR}/${MODEL_NAME}.xml"
echo "  ${OUTPUT_DIR}/${MODEL_NAME}.bin"
echo "=========================================="
