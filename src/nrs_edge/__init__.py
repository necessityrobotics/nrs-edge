# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics
"""nrs-edge: a small stereo depth network by Necessity Robotics.

    import nrs_edge
    model = nrs_edge.load()                 # ONNX Runtime on the CPU; the weights are downloaded on the first call
    disparity = model("left.png", "right.png")          # float32, H x W, pixels (a RECTIFIED pair)
    depth = nrs_edge.depth_from_disparity(disparity, fx_px=471.049, baseline_m=0.07514)   # metres, NaN = no value

Command line: `nrs-edge run`, `nrs-edge check`, `nrs-edge download`, `nrs-edge info`.
The package version is the version of this code. The weights are nrs-edge 0.1.0.
"""
__version__ = "0.1.1"
WEIGHTS_VERSION = "0.1.0"

from .api import Model, depth_from_disparity, load  # noqa: E402,F401
from .weights import fetch, files  # noqa: E402,F401

__all__ = ["Model", "load", "depth_from_disparity", "fetch", "files", "__version__", "WEIGHTS_VERSION"]
