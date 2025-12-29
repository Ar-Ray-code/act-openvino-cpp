
# act-openvino-cpp

Conversion Script for ACT's OpenVINO C++ Implementation

## Prerequisites

- Docker
- ACT checkpoint at `<path to checkpoint's dir path>`

I used [huggingface/lerobot](https://github.com/huggingface/lerobot) to create the model.

Required files in the checkpoint directory:
- `config.json`
- `model.safetensors`

Run all commands from the repo root.

## Quick commands

FP32:

```bash
./export_model.sh --checkpoint <path to checkpoint's dir path> --output-dir ./output --output-file act_model_fp32.onnx
./convert_to_openvino.sh --input ./output/act_model_fp32.onnx
./benchmark.sh --model ./output/act_model_fp32.xml --device CPU --iterations 100
```

FP16:

```bash
./export_model.sh --checkpoint <path to checkpoint's dir path> --output-dir ./output --output-file act_model_fp16.onnx
./convert_to_openvino.sh --input ./output/act_model_fp16.onnx --fp16
./benchmark.sh --model ./output/act_model_fp16.xml --device CPU --iterations 100
```

## Scripts

### export_model.sh

Exports an ACT checkpoint to ONNX using a Dockerized Python environment.

```bash
./export_model.sh --checkpoint <path to checkpoint's dir path> --output-dir ./output --output-file act_model.onnx
```

Options:
- `--checkpoint DIR` - checkpoint directory
- `--output-dir DIR` - output directory for the ONNX file
- `--output-file FILE` - ONNX filename
- `--fp16` - export in FP16 precision
- `--opset-version VER` - ONNX opset version (default: 14)

Note: the script requires arguments; run `./export_model.sh --help` for usage.

### convert_to_openvino.sh

Converts an ONNX model to OpenVINO IR (`.xml` and `.bin`) via a Dockerized OpenVINO Python environment.

```bash
./convert_to_openvino.sh --input ./output/act_model.onnx
```

Options:
- `--input PATH` - input ONNX model path (required)
- `--output-dir DIR` - output directory (default: same as input)
- `--fp16` - compress model to FP16

### benchmark.sh

Builds the C++ benchmark and runs it against an OpenVINO IR model.

```bash
./benchmark.sh --model ./output/act_model.xml --device CPU --iterations 100
```

Options:
- `--model PATH` - OpenVINO IR `.xml` path
- `--device DEVICE` - `CPU` or `GPU`
- `--iterations NUM` - number of iterations
- `--build-only` - build only, skip benchmark

## Reference

- [huggingface/lerobot](https://github.com/huggingface/lerobot)

```bibtex
@article{zhao2023learning,
  title={Learning fine-grained bimanual manipulation with low-cost hardware},
  author={Zhao, Tony Z and Kumar, Vikash and Levine, Sergey and Finn, Chelsea},
  journal={arXiv preprint arXiv:2304.13705},
  year={2023}
}

@misc{cadene2024lerobot,
    author = {Cadene, Remi and Alibert, Simon and Soare, Alexander and Gallouedec, Quentin and Zouitine, Adil and Palma, Steven and Kooijmans, Pepijn and Aractingi, Michel and Shukor, Mustafa and Aubakirova, Dana and Russi, Martino and Capuano, Francesco and Pascal, Caroline and Choghari, Jade and Moss, Jess and Wolf, Thomas},
    title = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
    howpublished = "\url{https://github.com/huggingface/lerobot}",
    year = {2024}
}
```