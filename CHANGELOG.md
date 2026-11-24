# Changes

The weights are `nrs-edge 0.1.0`. They have not changed since the release: `nrs-edge.safetensors` has the SHA-256
`aff7dfb82cbdba0cbfaa943bcbebc8a0583d276a3477317401e14a054054a254`.

## Repository update 1 (2026-11-25): Linux and Jetson

No change to the weights, the ONNX files, `infer.py`, the model card or any number.

Added:

- `docs/LINUX_JETSON.md`: commands for Linux (x86-64 and ARM64) and for NVIDIA Jetson, with a table of what is tested and what is not.
- `.github/workflows/linux.yml`: an automatic test on Linux x86-64 and ARM64 (CPU, Python 3.10 and 3.12). It installs the packages as the README says, downloads the weights without login, checks the hashes, runs both inference paths and compares the result with our reference.
- `tests/check_examples.py`: the same check for your own computer.
- `device/`: the device kit. `build_trt_jetson.sh` builds a TensorRT FP16 engine on a Jetson, measures the speed and runs `selftest.py`, which compares the output with our FP32 reference (`device/selftest_data/`). The kit ran on our Windows desktop with TensorRT 11.4. It has not run on a Jetson.

Changed in `README.md`:

- a "Platforms" line and a link to the guide;
- a one-line hash check for Windows;
- a note that a new run replaces the output files of the same picture;
- the install step now says: install PyTorch and torchvision with one command. The first Linux test run showed that
  a torchvision from another source does not match (error `operator torchvision::nms does not exist`). The ONNX path
  was not affected.

## Release 0.1.0 (2026-11-25)

First public release: weights (safetensors and ONNX), `infer.py`, model card, attribution.
