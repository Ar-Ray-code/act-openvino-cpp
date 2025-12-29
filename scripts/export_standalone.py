#!/usr/bin/env python3
"""
Standalone ACT Model ONNX Exporter

This script exports ACT model to ONNX without requiring the full lerobot library.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file

from act_model_standalone import ACTModel


def load_config(config_path: Path) -> dict:
    """Load model configuration from JSON"""
    with open(config_path) as f:
        return json.load(f)


def load_model_weights(model: nn.Module, weights_path: Path) -> nn.Module:
    """Load model weights from safetensors file"""
    print(f"Loading weights from: {weights_path}")
    state_dict = load_file(str(weights_path))

    # Map lerobot keys to standalone model keys
    new_state_dict = {}
    for key, value in state_dict.items():
        # Remove "model." prefix if present
        if key.startswith("model."):
            new_key = key[6:]  # Remove "model."
        else:
            new_key = key
        new_state_dict[new_key] = value

    # Load weights
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

    if missing:
        print(f"⚠ Missing keys: {len(missing)}")
        if len(missing) <= 10:
            for key in missing:
                print(f"  - {key}")
    if unexpected:
        print(f"⚠ Unexpected keys: {len(unexpected)}")
        if len(unexpected) <= 10:
            for key in unexpected:
                print(f"  - {key}")

    return model


def export_to_onnx(
    model: nn.Module,
    config: dict,
    output_path: Path,
    opset_version: int = 14,
    fp16: bool = False,
):
    """Export model to ONNX"""
    print("\nPreparing dummy inputs...")

    # Get input dimensions from config
    state_dim = config["input_features"]["observation.state"]["shape"][0]
    img_shape = config["input_features"]["observation.images.front"]["shape"]

    # Create dummy inputs (always FP32 for CPU compatibility)
    batch_size = 1
    dummy_state = torch.randn(batch_size, state_dim)
    dummy_image = torch.randn(batch_size, *img_shape)

    print(f"  State shape: {dummy_state.shape}")
    print(f"  Image shape: {dummy_image.shape}")

    # Set model to eval mode
    model.eval()

    # Test forward pass
    print("\nTesting forward pass...")
    with torch.no_grad():
        try:
            output = model(dummy_state, dummy_image)
            print(f"✓ Output shape: {output.shape}")
        except Exception as e:
            print(f"✗ Forward pass failed: {e}")
            raise

    # Export to ONNX
    print(f"\nExporting to ONNX...")
    print(f"  Output path: {output_path}")
    print(f"  Opset version: {opset_version}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_state, dummy_image),
            str(output_path),
            input_names=["observation.state", "observation.images.front"],
            output_names=["actions"],
            opset_version=opset_version,
            do_constant_folding=True,
            export_params=True,
            verbose=False,
        )

    print(f"✓ Model exported successfully")

    # Convert to FP16 if requested
    if fp16:
        print("\nConverting ONNX model to FP16...")
        try:
            from onnxconverter_common import float16
            import onnx

            onnx_model = onnx.load(str(output_path))

            # Use onnxconverter-common for proper FP16 conversion
            # This handles mixed precision correctly for OpenVINO
            onnx_model_fp16 = float16.convert_float_to_float16(
                onnx_model,
                keep_io_types=True,  # Keep input/output as FP32 for compatibility
                disable_shape_infer=False
            )

            # Save FP16 model
            onnx.save(onnx_model_fp16, str(output_path))
            print("✓ Model converted to FP16")
        except ImportError:
            print("⚠ onnxconverter-common not available")
            print("  Installing it would enable proper FP16 conversion:")
            print("  pip install onnxconverter-common")
            print("  Model remains in FP32 format")
        except Exception as e:
            print(f"⚠ FP16 conversion error: {e}")
            print("  Model remains in FP32 format")

    # Verify the model
    try:
        import onnx
        print("\nVerifying ONNX model...")
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model verification passed")

        # Print model info
        print("\n" + "=" * 80)
        print("ONNX Model Information")
        print("=" * 80)
        print(f"IR Version: {onnx_model.ir_version}")
        print(f"Producer: {onnx_model.producer_name} {onnx_model.producer_version}")
        print(f"Opset Version: {onnx_model.opset_import[0].version}")

        print("\nInputs:")
        for inp in onnx_model.graph.input:
            shape = [
                dim.dim_value if dim.dim_value > 0 else dim.dim_param
                for dim in inp.type.tensor_type.shape.dim
            ]
            print(f"  {inp.name}: {shape}")

        print("\nOutputs:")
        for out in onnx_model.graph.output:
            shape = [
                dim.dim_value if dim.dim_value > 0 else dim.dim_param
                for dim in out.type.tensor_type.shape.dim
            ]
            print(f"  {out.name}: {shape}")

        print(f"\nTotal nodes: {len(onnx_model.graph.node)}")

        # Count nodes by type
        node_types = {}
        for node in onnx_model.graph.node:
            node_types[node.op_type] = node_types.get(node.op_type, 0) + 1

        print("\nTop 15 node types:")
        for op_type, count in sorted(node_types.items(), key=lambda x: x[1], reverse=True)[:15]:
            print(f"  {op_type:30s}: {count:4d}")

        # Get model size
        model_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"\nModel size: {model_size_mb:.2f} MB")

    except ImportError:
        print("⚠ onnx package not available, skipping verification")
    except Exception as e:
        print(f"⚠ Verification error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Export ACT model to ONNX (standalone)")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/checkpoints/030000/pretrained_model"),
        help="Path to checkpoint directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("conversion/output/act_model.onnx"),
        help="Output ONNX file path",
    )
    parser.add_argument(
        "--opset-version", type=int, default=14, help="ONNX opset version"
    )
    parser.add_argument(
        "--fp16", action="store_true", help="Export model in FP16 precision"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("ACT Model ONNX Exporter (Standalone)")
    print("=" * 80)

    # Load config
    config_path = args.checkpoint / "config.json"
    print(f"\nLoading config from: {config_path}")
    config = load_config(config_path)

    print(f"\nModel Configuration:")
    print(f"  Type: {config.get('type', 'N/A')}")
    print(f"  Vision Backbone: {config.get('vision_backbone', 'N/A')}")
    print(f"  Chunk Size: {config.get('chunk_size', 'N/A')}")
    print(f"  Latent Dim: {config.get('latent_dim', 'N/A')}")
    print(f"  Transformer Dim: {config.get('dim_model', 'N/A')}")
    print(f"  Encoder Layers: {config.get('n_encoder_layers', 'N/A')}")
    print(f"  Decoder Layers: {config.get('n_decoder_layers', 'N/A')}")

    # Create model
    print("\nCreating model...")
    model = ACTModel(config)
    print(f"✓ Model created")

    # Load weights
    weights_path = args.checkpoint / "model.safetensors"
    model = load_model_weights(model, weights_path)
    print(f"✓ Weights loaded")

    # Export to ONNX
    export_to_onnx(model, config, args.output, args.opset_version, args.fp16)

    print("\n" + "=" * 80)
    print("Export completed successfully!")
    if args.fp16:
        print("Model exported in FP16 precision")
    print("=" * 80)


if __name__ == "__main__":
    main()
