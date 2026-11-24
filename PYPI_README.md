# nrs-edge

`nrs-edge` is a small stereo depth network by Necessity Robotics (6.04 M parameters). It takes a **rectified** stereo pair and gives the disparity of the left image. With the focal length and the baseline of your rig it also gives metric depth.

- Weights licence: Apache-2.0. The weights are ours (`nrs-edge 0.1.0`). Network design: LiteAnyStereo v2, size S, by Jing et al. (MIT code, unmodified).
- Code, model card, Linux and Jetson guide: https://github.com/necessityrobotics/nrs-edge
- Weights download: https://huggingface.co/Pyassahu/nrs-edge

## Install and check

```bash
pip install nrs-edge
nrs-edge check
```

If pip does not find `nrs-edge`, the package is not on PyPI yet. Then use: `pip install "git+https://github.com/necessityrobotics/nrs-edge"` (this needs `git`).

`nrs-edge check` downloads the weights (about 24 MB, no login, checked by SHA-256), runs three bundled pairs and compares the result with our reference. It must print `nrs-edge check: PASS`.

## Use

Command line:

```bash
nrs-edge run --left left.png --right right.png --out_dir out --fx 471.049 --baseline 0.07514
```

`--fx` is the rectified focal length in pixels and `--baseline` the baseline in metres. Without them you get disparity only.

Python:

```python
import nrs_edge

model = nrs_edge.load()                                   # ONNX Runtime on the CPU
disparity = model("left.png", "right.png")                # float32, H x W, pixels
depth = nrs_edge.depth_from_disparity(disparity, fx_px=471.049, baseline_m=0.07514)   # metres; NaN = no value
```

A picture can be a file path or a numpy array. An array must be uint8 in **RGB** order (not the BGR order of `cv2.imread`), or a grey picture (H x W).

The default path uses ONNX Runtime and needs no PyTorch. Its graphs take pictures of 640 x 400 and 640 x 480. A smaller picture is padded. A larger picture is resized to exactly the graph size: the aspect ratio is not kept, and detail is lost.

For any picture size, use the PyTorch path. First install PyTorch **and torchvision** with the one command from https://pytorch.org (a torchvision from another source does not match your PyTorch). Then:

```bash
pip install "nrs-edge[torch]"
nrs-edge run --backend torch --left left.png --right right.png
```

If you already have your own OpenCV (`cv2`) or `onnxruntime-gpu`, install with `pip install --no-deps nrs-edge` and install `numpy` and `huggingface_hub` yourself. Without `--no-deps`, pip installs `opencv-python` and the CPU `onnxruntime` beside your packages, and that can break them.

## What you must know

- **Accuracy.** On our harness (FP32; share of wrong pixels, lower is better): Middlebury-H bad-2 12.71 %, ETH3D bad-1 5.09 %, KITTI 2015 bad-3 5.57 % (not zero-shot: driving images were in the training data), Booster-Q bad-2 28.63 %. These numbers are for the PyTorch path at full picture size. The ONNX path resizes pictures larger than 640 x 480; it has no accuracy numbers of its own. We fixed pass marks before the tests: the weights passed the "Near" mark and failed the "Floor" mark. The model card has every number and every scored candidate.
- **Limits.** At most 192 px of disparity. No confidence output: glass, mirrors, blank walls and sky get a value that can be wrong. The input must be rectified. Not for safety-critical use without an independent safety layer.
- **Platforms.** Tested on Windows (x86-64) and on Linux (x86-64 and ARM64), on the CPU. For NVIDIA Jetson the repository has a guide and a TensorRT build script; they have not run on a device.
- **Versions.** The package version is the version of the code. The weights are `nrs-edge 0.1.0`.

Contact: necessityrobotics@gmail.com
