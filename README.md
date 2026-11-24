# nrs-edge

`nrs-edge` is a small stereo depth network by Necessity Robotics. It takes a rectified stereo pair and gives the disparity of the left image. With the focal length and the baseline of your rig it also gives metric depth.

- **Network:** LiteAnyStereo v2, size S (6.04 M parameters), the authors' MIT code, unmodified.
- **Weights:** ours. No weights of another group and no ImageNet weights were used.
- **Licence:** Apache-2.0.
- **Weights download:** Hugging Face, [`Pyassahu/nrs-edge`](https://huggingface.co/Pyassahu/nrs-edge).
- **Model card:** `MODEL_CARD.md` (every scored candidate, gates, training data, limits).
- **Speed:** no Jetson speed was measured. Real-time use on a Jetson-class board is our aim, not a result.
- **Platforms:** Windows (x86-64) and Linux (x86-64 and ARM64). The Linux install and both run paths are tested automatically on the CPU. For NVIDIA Jetson there is a guide and a TensorRT build script; they have not run on a device. See [`docs/LINUX_JETSON.md`](docs/LINUX_JETSON.md).

## Accuracy of these weights (our harness, FP32; lower is better)

| Model | Middlebury-H bad-2 | ETH3D bad-1 | KITTI-15 bad-3 | Booster-Q bad-2 |
|---|---|---|---|---|
| **nrs-edge 0.1.0** | **12.71** | **5.09** | **5.57** | **28.63** |
| LAS2-S, authors' weights (Stretch bar model) | 11.29 | 4.11 | 4.01 | 20.60 |
| RT-IGEV, Scene Flow weights, 8 iterations (Floor bar model) | 11.51 | 6.48 | 6.29 | 23.48 |

**Pre-registered gates of these weights:** Floor FAIL (2 of 4), Near PASS. Level reached: Near, not Floor.

The KITTI value is domain-exposed (driving images were in the training data). See `MODEL_CARD.md` for the protocol, all candidates and the gate results.

## Quick start (one command)

You need Python 3.10 or newer, and `git`.

```bash
pip install "git+https://github.com/necessityrobotics/nrs-edge"
nrs-edge check
nrs-edge run --left left.png --right right.png --out_dir out --fx 471.049 --baseline 0.07514
```

- `nrs-edge check` downloads the weights (about 24 MB, no login, checked by SHA-256), runs the three bundled pairs and compares them with our reference. It must print `nrs-edge check: PASS`.
- This path uses ONNX Runtime. It needs no PyTorch. Its graphs take pictures of 640 x 400 and 640 x 480: a smaller picture is padded; a larger picture is resized to exactly the graph size (the aspect ratio is not kept, and detail is lost).
- In Python: `import nrs_edge; disparity = nrs_edge.load()("left.png", "right.png")`. A numpy array must be uint8 in RGB order (not the BGR order of `cv2.imread`), or a grey picture.
- For any picture size, use the PyTorch path. First install PyTorch **and torchvision** with the one command from https://pytorch.org, then: `pip install "nrs-edge[torch] @ git+https://github.com/necessityrobotics/nrs-edge"` and `nrs-edge run --backend torch ...`.
- If you already have your own OpenCV (`cv2`) or `onnxruntime-gpu`, add `--no-deps` to the pip command and install `numpy` and `huggingface_hub` yourself. Without `--no-deps`, pip installs `opencv-python` and the CPU `onnxruntime` beside your packages, and that can break them.
- The accuracy numbers above are for the PyTorch path at full picture size. The ONNX path resizes pictures larger than 640 x 480; it has no accuracy numbers of its own.
- The package is not on PyPI yet. When it is, the first line becomes `pip install nrs-edge`.

## Install from a clone (the script `infer.py`)

Use this path to read the code, to run the device kit, or to work without the Python package.

You need Python 3.10 or newer.

1. Install PyTorch **and torchvision** for your system with the command from https://pytorch.org (CUDA or CPU build). Install both with that one command: `timm` needs torchvision, and a torchvision from another source does not match your PyTorch (error `operator torchvision::nms does not exist`).
2. Install the other packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the weights into this folder:

   ```bash
   pip install huggingface_hub
   hf download Pyassahu/nrs-edge nrs-edge.safetensors nrs-edge.safetensors.sha256 nrs-edge_w640h400_fp32.onnx nrs-edge_w640h400_fp32.onnx.sha256 --local-dir .
   ```

4. Check the download. On Linux, each line must end with `OK`:

   ```bash
   sha256sum -c nrs-edge.safetensors.sha256
   sha256sum -c nrs-edge_w640h400_fp32.onnx.sha256
   ```

   On Windows (PowerShell), this line must print `True`:

   ```powershell
   (Get-FileHash nrs-edge.safetensors -Algorithm SHA256).Hash -eq (Get-Content nrs-edge.safetensors.sha256).Split(' ')[0]
   ```

5. Check the install (optional). It runs the three example pairs through both paths and compares them with our reference. It must print `nrs-edge install check: PASS`:

   ```bash
   python tests/check_examples.py
   ```

## Run

PyTorch (CPU, or `--device cuda`):

```bash
python infer.py --weights nrs-edge.safetensors --left examples/pair_01_left.png --right examples/pair_01_right.png --out_dir out --fx 471.049 --baseline 0.07514
```

ONNX Runtime (CPU):

```bash
python infer.py --onnx nrs-edge_w640h400_fp32.onnx --left examples/pair_01_left.png --right examples/pair_01_right.png --out_dir out
```

- A new run replaces the output files of the same picture in `--out_dir`. Use another `--out_dir` to keep an earlier result.
- `--fx` is the rectified focal length in pixels and `--baseline` the baseline in metres. The values above are for the example pairs (our OAK-D Lite). Without them you get disparity only.
- The ONNX graphs have fixed sizes: `w640h480` (480 x 640) and `w640h400` (400 x 640). A smaller pair is padded; a larger pair is resized, which loses detail. The `_fp16` graphs are for TensorRT FP16 (strongly typed) builds.

**Outputs** in `--out_dir`, named after the left image:

| File | Content |
|---|---|
| `<name>_disp.npy` | float32 disparity of the left image, pixels |
| `<name>_disp16.png` | 16-bit PNG, value = disparity x 256 (KITTI convention), 0 = no value |
| `<name>_disp_color.png` | colour preview, 0 to 192 px |
| `<name>_depth.npy` | float32 depth in metres, NaN where disparity <= 0 (with `--fx` and `--baseline`) |
| `<name>_depth16.png` | 16-bit PNG, depth in millimetres, 0 = no value or above 65.535 m (with `--fx` and `--baseline`) |

## Known limits

- At most 192 px of disparity. Nearer objects are out of range.
- No confidence output. Glass, mirrors, blank walls and sky get a value that can be wrong.
- The input must be rectified. A vertical offset between the images makes the result worse.
- Not for safety-critical use without an independent safety layer.

## Folder contents

| Path | What |
|---|---|
| `infer.py` | inference script, PyTorch and ONNX Runtime |
| `third_party/LiteAnyStereo/` | the authors' code, commit 8c97bd4, unmodified files (MIT) |
| `examples/` | three rectified pairs from our own OAK-D Lite (640x400) |
| `MODEL_CARD.md`, `ATTRIBUTION.md`, `NOTICE`, `LICENSE`, `LICENSES/` | card, credits, licence texts |
| `src/nrs_edge/`, `pyproject.toml` | the Python package: command `nrs-edge`, module `nrs_edge` (`_infer.py` is `infer.py` with one line replaced by a comment; the weights are downloaded from Hugging Face) |
| `PYPI_README.md`, `.github/workflows/publish.yml` | the text of the PyPI page and the upload workflow, for later; the package is not on PyPI yet, so use the `git+` command above |
| `docs/LINUX_JETSON.md` | commands for Linux and for NVIDIA Jetson; what is tested and what is not |
| `tests/check_examples.py` | install check: both paths against our reference |
| `device/` | device kit: `build_trt_jetson.sh` (TensorRT engine, timing, self-test on a Jetson), `selftest.py`, the reference data |
| `.github/workflows/linux.yml` | the automatic Linux test (x86-64 and ARM64, CPU) |
| `CHANGELOG.md` | what changed in this repository; the weights are unchanged |
| `MANIFEST.sha256` | SHA-256 of every file in this folder |

## Citation and credits

The network design and code are by Jing et al. (*Lite Any Stereo*, CVPR 2026; *Lite Any Stereo V2*, arXiv 2606.24457). Please cite their work. Data credits: `ATTRIBUTION.md`.

Contact: necessityrobotics@gmail.com
