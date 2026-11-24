# Changes

The weights are `nrs-edge 0.1.0`. They have not changed since the release: `nrs-edge.safetensors` has the SHA-256
`aff7dfb82cbdba0cbfaa943bcbebc8a0583d276a3477317401e14a054054a254`.

## Repository update 2 (2026-11-25): Python package 0.1.1

No change to the weights, the ONNX files, `infer.py`, the model card or any number.

Added:

- `pyproject.toml` and `src/nrs_edge/`: a Python package. One command installs it: `pip install "git+https://github.com/necessityrobotics/nrs-edge"`. It gives the command `nrs-edge` (`check`, `run`, `download`, `info`) and the module `nrs_edge` (`nrs_edge.load()`).
- The default path uses ONNX Runtime and needs no PyTorch. The PyTorch path is the extra `nrs-edge[torch]`.
- The weights are downloaded from Hugging Face at one fixed revision on the first use, without login, and are checked by SHA-256.
- `src/nrs_edge/_infer.py` is `infer.py` with one line replaced by a comment; the package runs the same code as the script. As a library it does not change `sys.path`, and it raises an error where the script ends the program.
- The automatic Linux test now also installs the package with pip, runs `nrs-edge check` and runs `tests/package_behaviour.py` (picture formats, graph sizes, error handling) on x86-64 and ARM64.
- `.github/workflows/publish.yml` for a later upload to PyPI. The package is not on PyPI yet.

The package version 0.1.1 is the version of the code. The weights stay `nrs-edge 0.1.0`.

## Repository update 1 (2026-11-25): Linux and Jetson

No change to the weights, the ONNX files, `infer.py`, the model card or any number.

Added:

- `docs/LINUX_JETSON.md`: commands for Linux (x86-64 and ARM64) and for NVIDIA Jetson, with a table of what is tested and what is not.
- `.github/workflows/linux.yml`: an automatic test on Linux x86-64 and ARM64 (CPU, Python 3.10 and 3.12). It installs the packages as the README says, downloads the weights without login, checks the hashes, runs both inference paths and compares the result with our reference.
- `tests/check_examples.py`: the same check for your own computer.
- `device/`: the device kit. `build_trt_jetson.sh` builds a TensorRT FP16 engine on a Jetson, measures the speed and runs `selftest.py`, which compares the output with our FP32 reference (`device/selftest_data/`). `selftest.py` and `runtimes.py` ran on our Windows desktop with a TensorRT 11.4 engine. `build_trt_jetson.sh` has not built an engine on any computer yet; on the Linux test computers it stops at "trtexec not found", as it must. We tested the logic of the script (order of the steps, fallback, exit codes) on the desktop with a stand-in for `trtexec`. Nothing has run on a Jetson.

Changed in `README.md`:

- a "Platforms" line and a link to the guide;
- a one-line hash check for Windows;
- a note that a new run replaces the output files of the same picture;
- the install step now says: install PyTorch and torchvision with one command. The first Linux test run showed that
  a torchvision from another source does not match (error `operator torchvision::nms does not exist`). The ONNX path
  was not affected.

## Release 0.1.0 (2026-11-25)

First public release: weights (safetensors and ONNX), `infer.py`, model card, attribution.
