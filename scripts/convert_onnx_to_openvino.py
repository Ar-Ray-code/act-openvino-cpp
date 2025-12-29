#!/usr/bin/env python3
"""
Convert ONNX model to OpenVINO IR format
"""

import argparse
from pathlib import Path
import openvino as ov


def main():
    parser = argparse.ArgumentParser(description="Convert ONNX to OpenVINO IR")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input ONNX model path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output XML file path",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Compress model to FP16",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("ONNX to OpenVINO IR Conversion")
    print("=" * 80)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"FP16: {args.fp16}")
    print("=" * 80)
    print()

    # Read ONNX model
    print(f"Reading ONNX model: {args.input}")
    core = ov.Core()
    model = core.read_model(args.input)

    print(f"Model loaded successfully")
    print(f"  Inputs: {len(model.inputs)}")
    for inp in model.inputs:
        print(f"    - {inp.any_name}: {inp.shape}")
    print(f"  Outputs: {len(model.outputs)}")
    for out in model.outputs:
        print(f"    - {out.any_name}: {out.shape}")

    # Compress to FP16 if requested
    if args.fp16:
        print("\nCompressing model to FP16 (keeping I/O as FP32)...")
        from openvino.runtime import Type
        from openvino.runtime.passes import Manager, ConvertFP32ToFP16
        from openvino.preprocess import PrePostProcessor

        # Run FP16 conversion pass
        pass_manager = Manager()
        pass_manager.register_pass(ConvertFP32ToFP16())
        pass_manager.run_passes(model)

        # Use PrePostProcessor to add FP32<->FP16 conversion at I/O boundaries
        ppp = PrePostProcessor(model)

        # For each input, expect FP32 from user and convert to model's FP16
        for i, inp in enumerate(model.inputs):
            if inp.element_type == Type.f16:
                ppp.input(i).tensor().set_element_type(Type.f32)
                ppp.input(i).preprocess().convert_element_type(Type.f16)

        # For each output, convert from model's FP16 to FP32 for user
        for i, out in enumerate(model.outputs):
            if out.element_type == Type.f16:
                ppp.output(i).postprocess().convert_element_type(Type.f32)
                ppp.output(i).tensor().set_element_type(Type.f32)

        model = ppp.build()
        print("  ✓ Model compressed to FP16 (I/O kept as FP32)")

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Serialize model to IR format
    print(f"\nSaving OpenVINO IR to: {args.output}")
    ov.serialize(model, str(args.output))
    print("  ✓ Model saved")

    # Get bin file path
    bin_path = args.output.with_suffix('.bin')

    print(f"  ✓ XML file: {args.output}")
    print(f"  ✓ BIN file: {bin_path}")

    # Get file sizes
    xml_size_mb = args.output.stat().st_size / (1024 * 1024)
    bin_size_mb = bin_path.stat().st_size / (1024 * 1024)

    print(f"\nFile sizes:")
    print(f"  XML: {xml_size_mb:.2f} MB")
    print(f"  BIN: {bin_size_mb:.2f} MB")
    print(f"  Total: {xml_size_mb + bin_size_mb:.2f} MB")

    print("\n" + "=" * 80)
    print("Conversion completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
