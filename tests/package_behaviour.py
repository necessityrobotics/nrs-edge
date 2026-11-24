# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics
"""Behaviour test of the INSTALLED Python package nrs-edge: picture formats, graph sizes, error handling, and that the
package does not change sys.path or a module named `core` of the user.

    python tests/package_behaviour.py examples            # ONNX path
    python tests/package_behaviour.py examples torch      # PyTorch path as well (needs the extra packages)

Run it in a folder that you can write to: with "torch" it makes a small folder `core` there. Exit code 0 = all passed.
"""
import os
import sys

import cv2
import numpy as np

import nrs_edge

EX = sys.argv[1]
TORCH = len(sys.argv) > 2 and sys.argv[2] == "torch"
L, R = os.path.join(EX, "pair_02_left.png"), os.path.join(EX, "pair_02_right.png")
n_ok = n_bad = 0


def check(ok, what):
    global n_ok, n_bad
    n_ok += bool(ok)
    n_bad += not ok
    print(("PASS  " if ok else "FAIL  ") + what)


def raises(fn, exc, what):
    try:
        fn()
    except exc as e:
        check(True, what + " -> " + type(e).__name__ + ": " + str(e)[:90])
    except BaseException as e:
        check(False, what + " -> wrong error " + type(e).__name__ + ": " + str(e)[:90])
    else:
        check(False, what + " -> no error")


path_before = list(sys.path)
m = nrs_edge.load(quiet=True)
d = m(L, R)
check(d.dtype == np.float32 and d.shape == (400, 640) and abs(float(np.median(d)) - 5.20) < 0.01, "paths: 400 x 640, median 5.20 px")
g, g2 = cv2.imread(L, cv2.IMREAD_GRAYSCALE), cv2.imread(R, cv2.IMREAD_GRAYSCALE)
check(float(np.abs(m(g, g2) - d).max()) == 0.0, "grey uint8 arrays give the same result as the files")
rgb = np.repeat(g[:, :, None], 3, 2)
rgb2 = np.repeat(g2[:, :, None], 3, 2)
check(float(np.abs(m(rgb, rgb2) - d).max()) == 0.0, "RGB uint8 arrays give the same result")
rgba = np.concatenate([rgb, np.full(g.shape + (1,), 255, np.uint8)], 2)
rgba2 = np.concatenate([rgb2, np.full(g.shape + (1,), 255, np.uint8)], 2)
check(float(np.abs(m(rgba, rgba2) - d).max()) == 0.0, "RGBA arrays: the alpha channel is dropped")
check(float(np.abs(m(g[:, :, None], g2[:, :, None]) - d).max()) == 0.0, "H x W x 1 arrays are accepted")
check(float(np.abs(m(g.astype(np.float32), g2.astype(np.float32)) - d).max()) == 0.0, "float arrays 0 to 255 give the same result")
raises(lambda: m(g.astype(np.uint16) * 257, g2.astype(np.uint16) * 257), ValueError, "uint16 arrays are refused")
raises(lambda: m(g / 255.0, g2 / 255.0), ValueError, "float arrays 0 to 1 are refused")
raises(lambda: m(g, g2[:-2]), ValueError, "pairs of different size are refused")
raises(lambda: m(np.zeros((4, 5, 2), np.uint8), np.zeros((4, 5, 2), np.uint8)), ValueError, "a 2-channel array is refused")
# sizes
t480 = cv2.copyMakeBorder(g, 40, 40, 0, 0, cv2.BORDER_REPLICATE)
t480b = cv2.copyMakeBorder(g2, 40, 40, 0, 0, cv2.BORDER_REPLICATE)
d480 = m(t480, t480b)
check(d480.shape == (480, 640) and "RESIZED" not in m.last_input.upper(), "640 x 480 picture: padded graph, not resized (%s)" % m.last_input)
check("w640h480" in m.description and "w640h400" in m.description, "the second graph was loaded when it was needed")
big = cv2.resize(g, (1280, 800))
big2 = cv2.resize(g2, (1280, 800))
db = m(big, big2)
check(db.shape == (800, 1280) and "640x400" in m.last_input, "1280 x 800 picture: resized to the 640 x 400 graph (%s)" % m.last_input)
check(abs(float(np.median(db)) / float(np.median(d)) - 2.0) < 0.1, "its disparity is scaled back to the picture width (median %.2f px)" % float(np.median(db)))
f = nrs_edge.api.auto_size
check((f(400, 640), f(480, 640), f(800, 1280), f(960, 1280), f(360, 1280)) == (400, 480, 400, 480, 400), "auto size rule")
raises(lambda: nrs_edge.load(provider="cuda", quiet=True), RuntimeError, "provider cuda on a CPU onnxruntime: RuntimeError, not an exit")
raises(lambda: nrs_edge.load(size=512), ValueError, "a wrong size is refused")
raises(lambda: nrs_edge.fetch("nothing.onnx"), ValueError, "an unknown file name is refused")
z = nrs_edge.depth_from_disparity(d, 471.049, 0.07514)
check(abs(float(np.nanmedian(z)) - 6.81) < 0.01, "depth: median 6.81 m")
check(sys.path == path_before, "sys.path is unchanged after the ONNX path")
if TORCH:
    os.makedirs("core", exist_ok=True)                      # a module named `core` of the user, in the working folder
    open(os.path.join("core", "__init__.py"), "w").write("USER_CORE = True\n")
    sys.path.insert(0, os.getcwd())
    import core
    path_before = list(sys.path)
    mt = nrs_edge.load(backend="torch", quiet=True)
    dt = mt(L, R)
    check(float(np.abs(dt - d).max()) < 1e-3, "PyTorch path agrees with the ONNX path (max difference %.1e px)" % float(np.abs(dt - d).max()))
    check(sys.path == path_before, "sys.path is unchanged after the PyTorch path")
    check(getattr(sys.modules.get("core"), "USER_CORE", False) is True, "the user's own module `core` is untouched")
    dbt = mt(big, big2)
    check(dbt.shape == (800, 1280) and "padded" in mt.last_input, "PyTorch path takes a 1280 x 800 picture without resizing (%s)" % mt.last_input)
else:
    raises(lambda: nrs_edge.load(backend="torch", quiet=True), RuntimeError, "PyTorch path without the extra packages: clear error")
print("RESULT: %d passed, %d failed" % (n_ok, n_bad))
sys.exit(1 if n_bad else 0)
