#!/usr/bin/env python3
"""nrs-edge device self-test: runs the 3 bundled rectified OAK-D Lite pairs through the shipped model and compares
the disparity with our stored FP32 reference. Prints median / p99 / max |delta| (px) per pair and PASS or FAIL.

  python3 selftest.py --engine build_nrs-edge_w640h400/nrs-edge_w640h400_fp16.engine   # TensorRT (build_trt_jetson.sh)
  python3 selftest.py --onnx nrs-edge_w640h400_fp32.onnx [--provider cpu|cuda|tensorrt|auto]
  python3 selftest.py --from_dir out/                         # pair_0N_disp.npy written by another runtime
                                                              # (640x400 float32)

Needs numpy and OpenCV; plus onnxruntime for --onnx, or the tensorrt Python package for --engine.
No network, no sudo; writes nothing unless --json is given. Exit code 0 = PASS, 1 = FAIL, 2 = could not run.
"""
import argparse
import hashlib
import json
import os
import platform
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True                       # write nothing into this folder (no __pycache__)
try:
    from edge_io import Fit, abs_diff_stats, read_rgb, to_nchw  # noqa: E402
except ImportError as _e:                    # for example OpenCV (cv2) is missing: "could not run", not "FAIL"
    print("nrs-edge self-test could not run: %s" % _e)
    sys.exit(2)

# ---------------- PASS / FAIL thresholds: |delta disparity| in px, device output vs our FP32 reference ----------------
# Set for a TensorRT FP16 engine on the device. Every pair must meet both.
MEDIAN_MAX_PX = 0.05
P99_MAX_PX = 0.5
# The max is printed for information only (one pixel at an occlusion edge can move a lot in FP16).
# -----------------------------------------------------------------------------------------------------------------------


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def device_lines():
    lines = ["python %s on %s %s" % (platform.python_version(), platform.system(), platform.machine())]
    for p, tag in (("/proc/device-tree/model", "board"), ("/etc/nv_tegra_release", "l4t")):
        try:
            lines.append("%s: %s" % (tag, open(p, "rb").read().replace(b"\0", b"").decode(errors="replace")
                                     .splitlines()[0][:100]))
        except OSError:
            pass
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--engine", help="TensorRT engine file")
    g.add_argument("--onnx", help="ONNX model (ONNX Runtime)")
    g.add_argument("--from_dir", help="folder with pair_0N_disp.npy (640x400 float32) from another runtime")
    ap.add_argument("--provider", default="cpu", choices=["cpu", "cuda", "tensorrt", "auto"])
    ap.add_argument("--data", default=os.path.join(HERE, "selftest_data"))
    ap.add_argument("--reference", default=None, help="default: <data>/reference")
    ap.add_argument("--median_max", type=float, default=MEDIAN_MAX_PX)
    ap.add_argument("--p99_max", type=float, default=P99_MAX_PX)
    ap.add_argument("--json", default=None, help="also write the result here")
    a = ap.parse_args()

    ref_dir = a.reference or os.path.join(a.data, "reference")
    if a.json and os.path.exists(a.json):
        os.remove(a.json)                                                # no stale result from an earlier run
    print("== nrs-edge self-test ==")
    for line in device_lines():
        print("  " + line)
    try:
        pmeta = json.load(open(os.path.join(a.data, "pairs.json"), encoding="utf-8"))
        man = json.load(open(os.path.join(ref_dir, "manifest.json"), encoding="utf-8"))
    except OSError as e:
        print("  cannot read the bundled data / reference: %s" % e)
        return 2
    if not man.get("release_reference", False):
        print("  !! REFERENCE IS '%s': this run only tests the tool, not a release model !!" % man.get("KIND"))
    fp_ref = man["model"]["weights_fingerprint"]

    runner = None
    try:
        if a.engine:
            from runtimes import TrtRunner
            runner = TrtRunner(a.engine)
        elif a.onnx:
            from runtimes import OrtRunner
            runner = OrtRunner(a.onnx, a.provider)
    except Exception as e:
        print("  cannot load the model: %s: %s" % (type(e).__name__, e))
        return 2
    if runner is not None:
        print("  runtime: " + runner.describe())
        fp = runner.metadata.get("nrs.weights_fingerprint")
        if fp is not None and fp != fp_ref:
            print("  FAIL: the model's weights (%s...) are not the ones the reference was made from (%s...)"
                  % (fp[:12], fp_ref[:12]))
            return 1
        if fp is None:
            print("  note: no weights fingerprint in this model file (TensorRT engine); the reference is for %s..."
                  % fp_ref[:12])
    try:
        shown = os.path.relpath(ref_dir, HERE)
    except ValueError:                                                   # another drive on Windows
        shown = ref_dir
    print("  reference: %s (%s)" % (shown, man.get("KIND")))
    print("  thresholds per pair: median <= %.3g px, p99 <= %.3g px" % (a.median_max, a.p99_max))

    results, ok_all = [], True
    for p in pmeta["pairs"]:
        lp, rp = os.path.join(a.data, p["left"]), os.path.join(a.data, p["right"])
        if sha256_file(lp) != p["left_sha256"] or sha256_file(rp) != p["right_sha256"]:
            print("  FAIL: bundled image changed or corrupted: %s" % p["id"])
            return 1
        try:
            ref = np.load(os.path.join(ref_dir, p["id"] + "_disp.npy"))
            if runner is None:
                out = np.load(os.path.join(a.from_dir, p["id"] + "_disp.npy")).astype(np.float32)
                if out.shape != ref.shape:
                    print("  FAIL: %s output shape %s, reference %s" % (p["id"], out.shape, ref.shape))
                    return 1
        except OSError as e:
            print("  cannot read a disparity file: %s" % e)
            return 2
        if runner is not None:
            L, R = read_rgb(lp), read_rgb(rp)
            fit = Fit(L.shape[0], L.shape[1], *runner.input_hw)
            if fit.mode != "pad":
                print("  FAIL: model input %s cannot hold the %dx%d pairs" % (runner.input_hw, L.shape[1], L.shape[0]))
                return 1
            try:
                out = fit.invert(runner.run(to_nchw(fit.apply(L)), to_nchw(fit.apply(R))))
            except Exception as e:
                print("  the model failed on %s: %s: %s" % (p["id"], type(e).__name__, e))
                return 2
        if not np.all(np.isfinite(out)):
            s = {"median_px": float("inf"), "p99_px": float("inf"), "max_px": float("inf"),
                 "non_finite": int((~np.isfinite(out)).sum())}
        else:
            s = abs_diff_stats(out, ref)
        ok = s["median_px"] <= a.median_max and s["p99_px"] <= a.p99_max
        ok_all &= ok
        s.update({"id": p["id"], "pass": ok})
        results.append(s)
        f = lambda x: ("%.4f" % x) if (x >= 1e-3 or x == 0) else ("%.1e" % x)
        print("  %s: |delta| median %s  p99 %s  max %s px  -> %s"
              % (p["id"], f(s["median_px"]), f(s["p99_px"]), f(s["max_px"]), "PASS" if ok else "FAIL"))
    verdict = "PASS" if ok_all else "FAIL"
    print("SELFTEST %s%s" % (verdict, "" if man.get("release_reference") else "  (test reference, not the reference of a release)"))
    if a.json:
        json.dump({"verdict": verdict, "pairs": results, "runtime": runner.describe() if runner else a.from_dir,
                   "reference_kind": man.get("KIND"), "thresholds": {"median": a.median_max, "p99": a.p99_max},
                   "device": device_lines()}, open(a.json, "w", encoding="utf-8"), indent=1)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
