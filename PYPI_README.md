# nrs-edge

`nrs-edge` is a small stereo depth network by Necessity Robotics (6.04 M parameters). It takes a **rectified** stereo pair and gives the disparity of the left image. With the focal length and the baseline of your rig it also gives metric depth.

- Weights: ours, `nrs-edge 0.1.0`, Apache-2.0. Network design: Lite Any Stereo V2-S by Jing et al. (MIT code, unmodified).
- Code, model card, Linux and Jetson guide: https://github.com/necessityrobotics/nrs-edge
- Weights: https://huggingface.co/Pyassahu/nrs-edge

## Install and check

```bash
pip install nrs-edge
nrs-edge check
```

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
disparity = model("left.png", "right.png")                # float32, H x W, pixels; numpy arrays work too
depth = nrs_edge.depth_from_disparity(disparity, fx_px=471.049, baseline_m=0.07514)   # metres; NaN = no value
```

The default path uses ONNX Runtime and needs no PyTorch. Its graphs have fixed sizes: pictures up to 640 x 400 or 640 x 480 are padded; a larger picture is resized, which loses detail. For any picture size, use the PyTorch path:

```bash
pip install "nrs-edge[torch]"
nrs-edge run --backend torch --left left.png --right right.png
```

## What you must know

- **Accuracy.** On our harness (FP32; share of wrong pixels, lower is better): Middlebury-H bad-2 12.71 %, ETH3D bad-1 5.09 %, KITTI 2015 bad-3 5.57 % (not zero-shot: driving images were in the training data), Booster-Q bad-2 28.63 %. We fixed pass marks before the tests: the weights passed the "Near" mark and failed the "Floor" mark. The model card has every number and every scored candidate.
- **Limits.** At most 192 px of disparity. No confidence output: glass, mirrors, blank walls and sky get a value that can be wrong. The input must be rectified. Not for safety-critical use without an independent safety layer.
- **Platforms.** Tested on Windows (x86-64) and on Linux (x86-64 and ARM64), on the CPU. For NVIDIA Jetson the repository has a guide and a TensorRT build script; they have not run on a device.
- **Versions.** The package version is the version of the code. The weights are `nrs-edge 0.1.0`.

Contact: necessityrobotics@gmail.com
