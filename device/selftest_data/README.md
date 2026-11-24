# Self-test data for nrs-edge

Three rectified stereo pairs from our own OAK-D Lite, 640x400, 8-bit grayscale PNG (the same pictures as in `examples/`),
and our reference for them.

- `pairs.json`: how the pairs were made (camera calibration, rectification, a check of the vertical offset) and the
  SHA-256 of every picture. `selftest.py` checks the pictures before it runs. The calibration file and the recordings
  that it names are not published.
- `reference/pair_0N_disp.npy`: the disparity of the released weights, PyTorch FP32 on a CPU, float32, 400x640, pixels.
- `reference/manifest.json`: the fingerprint of the weights that made the reference. `selftest.py` refuses an ONNX file
  with other weights. A TensorRT engine has no fingerprint, so the self-test cannot check which weights it holds.

`../selftest.py` runs the pairs through a model file (an ONNX graph or a TensorRT engine) and compares the result with
the reference. Limits for each pair: median difference 0.05 px, 99th-percentile difference 0.5 px.

Rectified camera of these pairs: fx = 471.049 px, baseline 0.07514 m.

Licence: Apache-2.0, as the scripts. Recorded by Necessity Robotics with its own camera.
