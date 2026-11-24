# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics
"""Python interface of nrs-edge. The model runners are the ones of the repository script infer.py (`_infer.py`)."""
import importlib
import os

from . import _infer
from .weights import fetch

MAX_DISP = _infer.MAX_DISP
SIZES = (400, 480)            # the ONNX graphs take pictures of 640 x 400 and 640 x 480 (width x height)


def _build_torch(weights, device):
    """The network with the released weights. The authors' code is imported as a part of this package, so that the
    user's `sys.path` is not changed and a module named `core` of the user is not touched."""
    models = importlib.import_module(__package__ + ".third_party.LiteAnyStereo.core.models")
    model = models.build_model("las2", fnet_pretrained=False, model_size="s", max_disp=MAX_DISP)
    sd, meta = _infer.load_weights(weights)
    models.load_model_weights(model, sd, strict=True)
    return model.to(device).eval(), meta


_infer.build_torch = _build_torch        # the script puts its folder on sys.path; a library must not


def auto_size(h, w):
    """The ONNX graph for a picture of h x w pixels: the smallest graph that holds it. For a larger picture, the graph
    whose shape (400 x 640 or 480 x 640) is nearer to the shape of the picture, because that picture is resized."""
    if w <= 640 and h <= 400:
        return 400
    if w <= 640 and h <= 480:
        return 480
    return 400 if abs(h / w - 400 / 640) <= abs(h / w - 480 / 640) else 480


def _as_rgb(x):
    """A picture as float32 RGB, H x W x 3, values 0 to 255. x: a file path, or a numpy array (see Model.__call__)."""
    import numpy as np
    if isinstance(x, (str, os.PathLike)):
        return _infer.read_rgb(x)
    a = np.asarray(x)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[:, :, 0]
    if a.ndim == 3 and a.shape[2] == 4:
        a = a[:, :, :3]                                  # the alpha channel is not used
    if a.ndim == 2:
        a = np.repeat(a[:, :, None], 3, axis=2)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError("a picture must be H x W (grey) or H x W x 3 (RGB); got the shape %s" % (a.shape,))
    if a.dtype == np.uint8:
        return np.ascontiguousarray(a, dtype=np.float32)
    if a.dtype.kind == "f":
        if a.size and float(np.nanmax(a)) <= 1.0 and float(np.nanmax(a)) > 0.0:
            raise ValueError("a float picture must have values 0 to 255; this one has values 0 to 1 (multiply it by 255)")
        return np.ascontiguousarray(a, dtype=np.float32)
    raise ValueError("a picture array must be uint8, or float with values 0 to 255; got %s (convert a 16-bit picture "
                     "to 8 bit first)" % a.dtype)


class Model:
    """A loaded nrs-edge model.

    disparity = model(left, right)

    left, right: a RECTIFIED stereo pair of the same size. Each one is a file path, or a numpy array:
        H x W (grey) or H x W x 3 in **RGB** order (not the BGR order of cv2.imread; for a grey picture the order does
        not matter), uint8, or float with values 0 to 255.
    Returns the disparity of the left picture: float32, H x W, pixels.
    `last_input` says how the last pair was fitted to the network (padded or resized). Use one Model in one thread.
    """

    def __init__(self, runners, backend, size, make):
        self._runners = runners            # {graph size or "torch": runner}
        self._make = make
        self.backend = backend
        self.size = size
        self.last_input = None

    @property
    def description(self):
        return "; ".join(r.name for r in self._runners.values())

    def _runner(self, h, w):
        if self.backend == "torch":
            return self._runners["torch"]
        s = auto_size(h, w) if self.size == "auto" else self.size
        if s not in self._runners:
            self._runners[s] = self._make(s)
        return self._runners[s]

    def __call__(self, left, right):
        L, R = _as_rgb(left), _as_rgb(right)
        if L.shape != R.shape:
            raise ValueError("left %s and right %s differ in size" % (L.shape, R.shape))
        disp, how = self._runner(L.shape[0], L.shape[1]).run(L, R)
        self.last_input = how
        return disp

    def __repr__(self):
        return "nrs_edge.Model(%s)" % self.description


def load(backend="onnx", size="auto", provider="cpu", device="cpu", threads=0, quiet=False):
    """Load the released weights.

    backend "onnx" (default): ONNX Runtime. The graphs have fixed sizes: 640 x 400 and 640 x 480 (width x height).
        `size` "auto" (default) takes, for each pair, the smallest graph that holds the picture; 400 or 480 fixes it.
        A smaller picture is padded. A larger picture is resized to exactly the graph size: the aspect ratio is not
        kept and detail is lost. The 640 x 400 graph is loaded now; the other one when a picture needs it.
        `provider`: "cpu", "cuda" or "tensorrt" (the last two need a GPU build of onnxruntime).
    backend "torch": PyTorch, any picture size (padded to a multiple of 32). `device`: "cpu" or "cuda".
        Needs the extra packages: pip install "nrs-edge[torch]". This path sets the PyTorch thread count (if
        `threads` is given) and switches TF32 off on CUDA for the whole process.
    """
    if backend == "onnx":
        if size != "auto" and size not in SIZES:
            raise ValueError('size must be "auto", 400 or 480')

        def make(s):
            try:
                return _infer.OnnxRunner(fetch("onnx%d" % s, quiet=quiet), provider, threads)
            except SystemExit as e:                     # the script ends the program here; a library must not
                raise RuntimeError(str(e)) from None
        first = 400 if size == "auto" else size
        return Model({first: make(first)}, backend, size, make)
    if backend == "torch":
        try:
            import safetensors  # noqa: F401
            import timm  # noqa: F401
            import torch  # noqa: F401
        except ImportError as e:
            raise RuntimeError('the PyTorch path needs extra packages: install PyTorch and torchvision with the one command '
                               'from https://pytorch.org, then pip install "nrs-edge[torch]" (%s)' % e) from e
        return Model({"torch": _infer.TorchRunner(fetch("torch", quiet=quiet), device, threads)}, backend, None, None)
    raise ValueError('backend must be "onnx" or "torch"')


def depth_from_disparity(disparity, fx_px, baseline_m):
    """Depth in metres: focal length [px] x baseline [m] / disparity [px]. NaN where the disparity is not above 0."""
    import numpy as np
    d = np.asarray(disparity, dtype=np.float32)
    depth = np.full(d.shape, np.nan, dtype=np.float32)
    ok = np.isfinite(d) & (d > 0)
    depth[ok] = np.float32(fx_px * baseline_m) / d[ok]
    return depth
