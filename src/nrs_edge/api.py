# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics
"""Python interface of nrs-edge. The model runners are the ones of the repository script infer.py (`_infer.py`)."""
import os

from . import _infer
from .weights import fetch

MAX_DISP = _infer.MAX_DISP


def _as_rgb(x):
    """A picture as float32 RGB, H x W x 3, values 0 to 255. x: a file path, or a numpy array (H x W or H x W x 3)."""
    import numpy as np
    if isinstance(x, (str, os.PathLike)):
        return _infer.read_rgb(x)
    a = np.asarray(x)
    if a.ndim == 2:
        a = np.repeat(a[:, :, None], 3, axis=2)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError("a picture must be H x W (grey) or H x W x 3 (RGB); got the shape %s" % (a.shape,))
    return np.ascontiguousarray(a, dtype=np.float32)


class Model:
    """A loaded nrs-edge model. Call it with a rectified left and right picture; it returns the disparity of the left
    picture: float32, H x W, pixels. `last_input` says how the pair was fitted to the network (padded or resized)."""

    def __init__(self, runner, backend, description):
        self._runner = runner
        self.backend = backend
        self.description = description
        self.last_input = None

    def __call__(self, left, right):
        L, R = _as_rgb(left), _as_rgb(right)
        if L.shape != R.shape:
            raise ValueError("left %s and right %s differ in size" % (L.shape, R.shape))
        disp, how = self._runner.run(L, R)
        self.last_input = how
        return disp

    def __repr__(self):
        return "nrs_edge.Model(%s)" % self.description


def load(backend="onnx", size=400, provider="cpu", device="cpu", threads=0, quiet=False):
    """Load the released weights.

    backend "onnx" (default): ONNX Runtime. `size` 400 or 480 selects the graph for pictures up to 640 x 400 or
        640 x 480. A smaller picture is padded; a larger one is resized (resizing loses detail).
        `provider`: "cpu", "cuda" or "tensorrt" (the last two need a GPU build of onnxruntime).
    backend "torch": PyTorch, any picture size (padded to a multiple of 32). `device`: "cpu" or "cuda".
        Needs the extra packages: pip install "nrs-edge[torch]".
    """
    if backend == "onnx":
        if size not in (400, 480):
            raise ValueError("size must be 400 or 480")
        r = _infer.OnnxRunner(fetch("onnx%d" % size, quiet=quiet), provider, threads)
    elif backend == "torch":
        try:
            import safetensors  # noqa: F401
            import timm  # noqa: F401
            import torch  # noqa: F401
        except ImportError as e:
            raise RuntimeError('the PyTorch path needs extra packages: pip install "nrs-edge[torch]" (%s)' % e) from e
        r = _infer.TorchRunner(fetch("torch", quiet=quiet), device, threads)
    else:
        raise ValueError('backend must be "onnx" or "torch"')
    return Model(r, backend, r.name)


def depth_from_disparity(disparity, fx_px, baseline_m):
    """Depth in metres: focal length [px] x baseline [m] / disparity [px]. NaN where the disparity is not above 0."""
    import numpy as np
    d = np.asarray(disparity, dtype=np.float32)
    depth = np.full(d.shape, np.nan, dtype=np.float32)
    ok = np.isfinite(d) & (d > 0)
    depth[ok] = np.float32(fx_px * baseline_m) / d[ok]
    return depth
