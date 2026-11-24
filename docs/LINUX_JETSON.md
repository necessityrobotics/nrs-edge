# nrs-edge on Linux and on NVIDIA Jetson

This guide gives the commands for Linux computers and for NVIDIA Jetson boards. Read the table first: it says what we tested and what we did not.

## What is tested

| Platform | State |
|---|---|
| Windows 11, x86-64: PyTorch (CPU and CUDA), ONNX Runtime (CPU), TensorRT 11.4 FP16 on an RTX 5090 | Tested by us. |
| Linux x86-64, CPU (Ubuntu 24.04; Python 3.10 and 3.12) | Tested automatically on every change: install, weights download, both run paths, comparison with our reference. See the "linux" workflow of this repository. |
| Linux ARM64, CPU (Ubuntu 24.04; Python 3.10 and 3.12) | The same automatic test. ARM64 is the processor type of a Jetson. |
| Linux with an NVIDIA GPU (PyTorch CUDA, ONNX Runtime CUDA) | **Not tested.** |
| NVIDIA Jetson (Orin, Thor): TensorRT engine, speed, accuracy | **Not tested.** Section 3 is written for it. It has not run on a device. |

CAUTION: We have no speed number and no accuracy number from a Jetson. If you run section 3, please send us the result files (section 3.5).

## 1. Linux computer, CPU (x86-64 or ARM64)

You need Python 3.10 or newer, `git`, and about 1 GB of disk space for the Python packages.

```bash
git clone https://github.com/necessityrobotics/nrs-edge.git
cd nrs-edge
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install huggingface_hub
hf download Pyassahu/nrs-edge nrs-edge.safetensors nrs-edge.safetensors.sha256 nrs-edge_w640h400_fp32.onnx nrs-edge_w640h400_fp32.onnx.sha256 --local-dir .
sha256sum -c nrs-edge.safetensors.sha256
sha256sum -c nrs-edge_w640h400_fp32.onnx.sha256
```

Check the install. It must print `nrs-edge install check: PASS`:

```bash
python tests/check_examples.py
```

Run one pair:

```bash
python infer.py --onnx nrs-edge_w640h400_fp32.onnx --left examples/pair_01_left.png --right examples/pair_01_right.png --out_dir out --fx 471.049 --baseline 0.07514
```

## 2. Linux computer with an NVIDIA GPU (not tested)

- **PyTorch:** install the CUDA build of PyTorch for your system from https://pytorch.org, then add `--device cuda` to the `infer.py --weights` command.
- **ONNX Runtime:** install `onnxruntime-gpu` in place of `onnxruntime`, then add `--provider cuda` to the `infer.py --onnx` command.

Run `python tests/check_examples.py` first. It uses the CPU and tells you if the install is correct.

## 3. NVIDIA Jetson (not tested on a device)

The fast path on a Jetson is a TensorRT engine in FP16. A TensorRT engine runs only with the TensorRT version and the GPU that built it. So you build the engine on the Jetson itself. The script `device/build_trt_jetson.sh` does this, measures the speed, and checks the result against our reference.

The commands are the same for both JetPack lines:

| JetPack line | Boards | System Python |
|---|---|---|
| JetPack 6 | Jetson Orin family | Python 3.10 |
| JetPack 7 | Jetson Thor; Jetson Orin family from JetPack 7.2 | Python 3.12 |

A full JetPack install has TensorRT, the `tensorrt` Python package and the tool `trtexec` (in `/usr/src/tensorrt/bin`). The script uses these. It needs no `sudo` and no network.

### 3.1 Get the files

```bash
git clone https://github.com/necessityrobotics/nrs-edge.git
cd nrs-edge
for f in nrs-edge_w640h400_fp32.onnx nrs-edge_w640h400_fp32.onnx.sha256 nrs-edge_w640h400_fp16.onnx nrs-edge_w640h400_fp16.onnx.sha256; do
  wget -q "https://huggingface.co/Pyassahu/nrs-edge/resolve/main/$f"
done
sha256sum -c nrs-edge_w640h400_fp32.onnx.sha256
sha256sum -c nrs-edge_w640h400_fp16.onnx.sha256
```

### 3.2 Python packages for the self-test

The self-test needs the system Python of JetPack (it has `tensorrt`), plus `numpy` and OpenCV (`cv2`). Check them:

```bash
python3 -c "import tensorrt, numpy, cv2; print(tensorrt.__version__, numpy.__version__, cv2.__version__)"
```

If `numpy` or `cv2` is missing, install the Ubuntu packages:

```bash
sudo apt install python3-numpy python3-opencv
```

### 3.3 Set the power mode yourself

The script does not change the power mode, the clocks or the fan. Set the mode that you want to measure before you start (for example with `sudo nvpmodel -m <id>`). The script prints the mode that it finds.

### 3.4 Build, measure, check

```bash
bash device/build_trt_jetson.sh nrs-edge_w640h400_fp32.onnx
```

What the script does:

1. It prints the board, the JetPack version, the TensorRT version and the power mode.
2. It builds an FP16 engine from `nrs-edge_w640h400_fp16.onnx` with `trtexec --stronglyTyped`. This is the graph that we checked on our desktop. The build can take several minutes.
3. If that build fails (for example with an older TensorRT), it builds from the FP32 graph with `trtexec --fp16`. The script says which build it used.
4. It measures the speed two times for 60 seconds each: the network alone, and end to end with the copies to and from the GPU.
5. It runs the 3 bundled pairs through the engine and compares the disparity with our FP32 reference. It prints `PASS` or `FAIL`. The limits are a median difference of 0.05 px and a 99th-percentile difference of 0.5 px for each pair.

Everything is written into the folder `build_nrs-edge_w640h400/`.

Options:

| Command | Effect |
|---|---|
| `DURATION=30 bash device/build_trt_jetson.sh <model>` | shorter speed measurement |
| `SKIP_TIMING=1 bash device/build_trt_jetson.sh <model>` | no speed measurement |
| `SKIP_SELFTEST=1 bash device/build_trt_jetson.sh <model>` | no comparison with the reference |
| `TYPING=weak bash device/build_trt_jetson.sh <model>` | build from the FP32 graph with `--fp16` at once |
| `TRTEXEC=/path/to/trtexec bash device/build_trt_jetson.sh <model>` | use another `trtexec` |

For the other graph size, download `nrs-edge_w640h480_fp32.onnx` and `nrs-edge_w640h480_fp16.onnx` and add `SKIP_SELFTEST=1`: the reference of the self-test is for 640 x 400 pictures.

### 3.5 Send us the result

We have no Jetson result. Yours would be the first. Please open an issue in this repository, or write to necessityrobotics@gmail.com, with these files from `build_nrs-edge_w640h400/`:

- `system.txt` (board, versions, power mode, which build was used; it has no host name, address, serial number or user name),
- `timing.txt`,
- `selftest.json`.

### 3.6 Your own pictures on the Jetson

| Way | State | How |
|---|---|---|
| The TensorRT engine from section 3.4 | not tested on a device | the small Python example below |
| ONNX Runtime on the CPU | tested on Linux ARM64 (not on a Jetson); slow | section 1 with the `infer.py --onnx` command |
| PyTorch with CUDA | not tested | install NVIDIA's PyTorch build for your JetPack, then `infer.py --weights ... --device cuda` |

A TensorRT engine has a fixed input size. The engine from section 3.4 takes rectified pictures of 640 x 400 pixels:

```python
import sys
sys.path.insert(0, "device")
import numpy as np
from edge_io import read_rgb, to_nchw
from runtimes import TrtRunner

runner = TrtRunner("build_nrs-edge_w640h400/nrs-edge_w640h400_fp16.engine")
left, right = read_rgb("examples/pair_01_left.png"), read_rgb("examples/pair_01_right.png")
disparity = runner.run(to_nchw(left), to_nchw(right))      # float32, 400 x 640, pixels
np.save("pair_01_disp.npy", disparity)
print(float(np.median(disparity)))                         # 4.32 for this pair on our desktop
depth_m = 471.049 * 0.07514 / np.maximum(disparity, 1e-6)  # focal length [px] x baseline [m] / disparity
```

## 4. Limits

- The limits of the model are in `MODEL_CARD.md`: at most 192 px of disparity, no confidence output, a rectified pair is needed.
- The FP16 graphs were checked for accuracy on the CPU and, as a TensorRT engine, on one Windows desktop GPU. No accuracy result from a Jetson exists.
- Not for safety-critical use without an independent safety layer.
