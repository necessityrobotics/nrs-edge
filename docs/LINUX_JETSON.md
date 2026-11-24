# nrs-edge on Linux and on NVIDIA Jetson

This guide gives the commands for Linux computers and for NVIDIA Jetson boards. Read the table first: it says what we tested and what we did not.

## What is tested

| Platform | State |
|---|---|
| Windows 11, x86-64: PyTorch (CPU and CUDA), ONNX Runtime (CPU), a TensorRT 11.4 FP16 engine on an RTX 5090 | Tested by us on one computer. |
| Linux x86-64, CPU (Ubuntu 24.04; Python 3.10 and 3.12) | Tested automatically on every change: install, weights download, both run paths, comparison with our reference. See the "linux" workflow of this repository. |
| Linux ARM64, CPU (Ubuntu 24.04; Python 3.10 and 3.12) | The same automatic test. ARM64 is the processor type of a Jetson, but this is not a Jetson test: the test computers have no JetPack and no GPU. |
| Linux with an NVIDIA GPU (PyTorch CUDA, ONNX Runtime CUDA) | **Not tested.** |
| NVIDIA Jetson (Orin, Thor): TensorRT engine, speed, accuracy | **Not tested.** Section 3 is written for it. The script `device/build_trt_jetson.sh` has not built an engine on any computer yet. On the Linux test computers it stops at "trtexec not found", as it must. |

CAUTION: We have no speed number and no accuracy number from a Jetson. If you run section 3, please send us the result files (section 3.5).

## 1. Linux computer, CPU (x86-64 or ARM64)

The quickest way is the Python package. It uses ONNX Runtime and needs no PyTorch (about 300 MB of disk space). You need Python 3.10 or newer, `git` and the venv module (on Ubuntu: `sudo apt install git python3-venv`):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install "git+https://github.com/necessityrobotics/nrs-edge"
nrs-edge check
```

`nrs-edge check` must print `nrs-edge check: PASS`. The rest of this section is the path with a clone of the repository. Use it for the PyTorch path, the script `infer.py` and the device kit.

You need Python 3.10 or newer, `git`, and about 1 GB of disk space for the Python packages.

Install PyTorch and torchvision with one command. A torchvision from another source does not match your PyTorch (section 4).

```bash
git clone https://github.com/necessityrobotics/nrs-edge.git
cd nrs-edge
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
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

Run one pair with PyTorch, and with ONNX Runtime:

```bash
python infer.py --weights nrs-edge.safetensors --left examples/pair_01_left.png --right examples/pair_01_right.png --out_dir out --fx 471.049 --baseline 0.07514
python infer.py --onnx nrs-edge_w640h400_fp32.onnx --left examples/pair_01_left.png --right examples/pair_01_right.png --out_dir out
```

## 2. Linux computer with an NVIDIA GPU (not tested)

Do section 1 first, with these changes.

- **PyTorch:** install the CUDA build of PyTorch and torchvision with the command from https://pytorch.org in place of the CPU command. Then add `--device cuda`:

  ```bash
  python infer.py --weights nrs-edge.safetensors --left examples/pair_01_left.png --right examples/pair_01_right.png --out_dir out --device cuda
  ```

- **ONNX Runtime:** replace the CPU package with the GPU package. Then add `--provider cuda`. If you use the Python package (`pip install`), install it with `--no-deps` from then on: without it, pip installs the CPU `onnxruntime` again beside `onnxruntime-gpu`.

  ```bash
  pip uninstall -y onnxruntime
  pip install onnxruntime-gpu
  python infer.py --onnx nrs-edge_w640h400_fp32.onnx --left examples/pair_01_left.png --right examples/pair_01_right.png --out_dir out --provider cuda
  ```

`python tests/check_examples.py` uses the CPU. Run it first: it tells you if the install is correct.

## 3. NVIDIA Jetson (not tested on a device)

The fast path on a Jetson is a TensorRT engine in FP16. A TensorRT engine runs only with the TensorRT version and the GPU that built it. So you build the engine on the Jetson itself. The script `device/build_trt_jetson.sh` does this, measures the speed, and checks the result against our reference.

CAUTION: This script has not built an engine on any computer yet. Its self-test part (`device/selftest.py`, `device/runtimes.py`) ran on our Windows desktop with a TensorRT 11.4 engine. We also tested the logic of the script there with a stand-in for `trtexec`: the stand-in copied an engine that the desktop had built. That test covers the order of the steps, the fallback and the exit codes. It does not cover a real build or a real timing.

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

### 3.2 Python for the self-test

The self-test needs a Python that has the `tensorrt` package. On a Jetson this is the system Python of JetPack. It also needs `numpy` and OpenCV (`cv2`).

1. If a Python environment is active (for example the `.venv` of section 1), leave it: `deactivate`.
2. Check the packages:

   ```bash
   python3 -c "import tensorrt, numpy, cv2; print(tensorrt.__version__, numpy.__version__, cv2.__version__)"
   ```

3. If `numpy` or `cv2` is missing, install the Ubuntu packages:

   ```bash
   sudo apt install python3-numpy python3-opencv
   ```

### 3.3 Set the power mode yourself

The script does not change the power mode, the clocks or the fan. Set the mode that you want to measure before you start (for example with `sudo nvpmodel -m <id>`). The script prints the mode that it finds.

### 3.4 Build, measure, check

Run this from the repository folder:

```bash
bash device/build_trt_jetson.sh nrs-edge_w640h400_fp32.onnx
```

What the script does:

1. It prints the board, the JetPack version, the TensorRT version and the power mode.
2. It builds an FP16 engine from `nrs-edge_w640h400_fp16.onnx` with `trtexec --stronglyTyped`. Our desktop TensorRT engine was built from this same file. The build can take several minutes.
3. If that build fails (for example with an older TensorRT), it builds from the FP32 graph with `trtexec --fp16`. The script says which build it used.
4. It measures the speed two times for 60 seconds each: the network alone, and end to end with the copies to and from the GPU.
5. It runs the 3 bundled pairs through the engine and compares the disparity with our FP32 reference. It prints `SELFTEST PASS` or `SELFTEST FAIL`. The limits are a median difference of 0.05 px and a 99th-percentile difference of 0.5 px for each pair.

Everything is written into the folder `build_nrs-edge_w640h400/`. The exit code of the script is 0 when the self-test passed (or was skipped on purpose), 1 when the build or the self-test failed, and 2 when the self-test could not run.

Options (a variable with any value is "on"):

| Command | Effect |
|---|---|
| `DURATION=30 bash device/build_trt_jetson.sh <model>` | shorter speed measurement |
| `SKIP_TIMING=1 bash device/build_trt_jetson.sh <model>` | no speed measurement |
| `SKIP_SELFTEST=1 bash device/build_trt_jetson.sh <model>` | no comparison with the reference |
| `TYPING=weak bash device/build_trt_jetson.sh <model>` | build from the FP32 graph with `--fp16` at once |
| `TYPING=strong bash device/build_trt_jetson.sh <model>` | stop if the strongly typed build fails (no fallback) |
| `TRTEXEC=/path/to/trtexec bash device/build_trt_jetson.sh <model>` | use another `trtexec` |
| `PYTHON=/usr/bin/python3 bash device/build_trt_jetson.sh <model>` | the Python for the self-test (default `python3`) |
| `OUT=my_folder bash device/build_trt_jetson.sh <model>` | another output folder |

The other graph size (640 x 480). The reference of the self-test is for 640 x 400 pictures, so skip the self-test:

```bash
for f in nrs-edge_w640h480_fp32.onnx nrs-edge_w640h480_fp32.onnx.sha256 nrs-edge_w640h480_fp16.onnx nrs-edge_w640h480_fp16.onnx.sha256; do
  wget -q "https://huggingface.co/Pyassahu/nrs-edge/resolve/main/$f"
done
sha256sum -c nrs-edge_w640h480_fp32.onnx.sha256
sha256sum -c nrs-edge_w640h480_fp16.onnx.sha256
SKIP_SELFTEST=1 bash device/build_trt_jetson.sh nrs-edge_w640h480_fp32.onnx
```

### 3.5 Send us the result

We have no Jetson result. Yours would be the first. Please open an issue in this repository, or write to necessityrobotics@gmail.com, with these files from `build_nrs-edge_w640h400/`:

- `system.txt` (board, versions, power mode, which build was used; it has no host name, address, serial number or user name),
- `timing.txt`,
- `selftest.json` (it exists only when the comparison ran),
- `build.log` if the build failed.

### 3.6 Your own pictures on the Jetson

| Way | State | How |
|---|---|---|
| The TensorRT engine from section 3.4 | not tested on a device | the small Python example below |
| ONNX Runtime on the CPU | tested on Linux ARM64 (not on a Jetson); we expect it to be slow on a Jetson, but we did not measure it | section 1 with the `infer.py --onnx` command |
| PyTorch with CUDA | not tested | install NVIDIA's PyTorch build for your JetPack, then `infer.py --weights ... --device cuda` |

A TensorRT engine has a fixed input size. The engine from section 3.4 takes rectified pictures of exactly 640 x 400 pixels; the example does no padding and no resizing. Run it from the repository folder, with the system Python:

```python
import sys
sys.path.insert(0, "device")
import numpy as np
from edge_io import depth_from_disparity, read_rgb, to_nchw
from runtimes import TrtRunner

runner = TrtRunner("build_nrs-edge_w640h400/nrs-edge_w640h400_fp16.engine")
left, right = read_rgb("examples/pair_01_left.png"), read_rgb("examples/pair_01_right.png")
disparity = runner.run(to_nchw(left), to_nchw(right))      # float32, 400 x 640, pixels
np.save("pair_01_disp.npy", disparity)
print(float(np.median(disparity)))                         # our FP32 reference for this pair: 4.32
depth_m = depth_from_disparity(disparity, 471.049, 0.07514)  # focal length [px], baseline [m]; NaN where disparity <= 0
```

## 4. Known problems

| Message | Cause | What to do |
|---|---|---|
| `RuntimeError: operator torchvision::nms does not exist` | PyTorch and torchvision come from different sources and do not match. This happens when only `torch` was installed and another package then installed torchvision. Our first Linux test run had this fault. | Install both again with one command, for example `pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cpu`. The ONNX path (`infer.py --onnx`) does not need PyTorch. |
| `trtexec not found` | TensorRT is not installed, or `trtexec` is in another place. | On a Jetson, install the full JetPack, or give the path: `TRTEXEC=/path/to/trtexec bash device/build_trt_jetson.sh <model>`. |
| `self-test skipped: the tensorrt Python package is not importable` | The Python that ran the self-test is not the system Python of JetPack (for example a virtual environment). | `deactivate`, or `PYTHON=/usr/bin/python3 bash device/build_trt_jetson.sh <model>`. |

## 5. Limits

- The limits of the model are in `MODEL_CARD.md`: at most 192 px of disparity, no confidence output, a rectified pair is needed.
- FP16: the benchmark accuracy numbers of the model card are FP32. On our Windows desktop, a strongly typed TensorRT 11.4 FP16 engine passed `device/selftest.py` (3 pairs, against our FP32 output). This is a check of 3 pictures, not a benchmark result. No FP16 result from a Jetson exists.
- Not for safety-critical use without an independent safety layer.
