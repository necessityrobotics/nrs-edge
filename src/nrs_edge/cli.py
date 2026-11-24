# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics
"""The `nrs-edge` command.

  nrs-edge check                                   run the 3 bundled pairs and compare with our reference (PASS / FAIL)
  nrs-edge run --left L.png --right R.png          disparity of one RECTIFIED pair, written into ./out
  nrs-edge run --left L.png --right R.png --fx 471.049 --baseline 0.07514        depth in metres as well
  nrs-edge download [--all]                        get the weight files now; print where they are
  nrs-edge info                                    versions, links, limits

The weights are downloaded on the first use (about 24 MB for the default graph), without login, and checked by SHA-256.
"""
import argparse
import json
import os
import platform
import sys
import time

from . import WEIGHTS_VERSION, __version__, _infer
from .api import load
from .weights import KINDS, fetch, files, table

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
MEDIAN_MAX_PX = 0.05          # limits of `nrs-edge check`, for each pair: |disparity - reference|
P99_MAX_PX = 0.5
GITHUB = "https://github.com/necessityrobotics/nrs-edge"


def _add_model_args(p):
    p.add_argument("--backend", default="onnx", choices=["onnx", "torch"],
                   help='onnx (default; ONNX Runtime) or torch (needs: pip install "nrs-edge[torch]")')
    p.add_argument("--size", default="auto", choices=["auto", "400", "480"],
                   help="ONNX graph: 640 x 400 or 640 x 480 (auto: the smallest that fits the picture)")
    p.add_argument("--provider", default="cpu", choices=["cpu", "cuda", "tensorrt"], help="ONNX Runtime provider")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="PyTorch device")
    p.add_argument("--threads", type=int, default=0, help="CPU threads (0 = library default)")


def _auto_size(h, w):
    """The ONNX graph for a picture: the smallest graph that holds it; for a larger picture, the graph whose shape
    (400 x 640 or 480 x 640) is nearer to the shape of the picture, because that picture is resized."""
    if w <= 640 and h <= 400:
        return 400
    if w <= 640 and h <= 480:
        return 480
    return 400 if abs(h / w - 400 / 640) <= abs(h / w - 480 / 640) else 480


def cmd_run(a):
    if (a.fx is None) != (a.baseline is None):
        sys.exit("--fx and --baseline go together")
    import numpy as np
    L, R = _infer.read_rgb(a.left), _infer.read_rgb(a.right)
    if L.shape != R.shape:
        sys.exit("left %s and right %s differ in size" % (L.shape, R.shape))
    size = _auto_size(L.shape[0], L.shape[1]) if a.size == "auto" else int(a.size)
    try:
        model = load(a.backend, size, a.provider, a.device, a.threads)
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        sys.exit("nrs-edge: %s" % e)
    print("model: %s; weights nrs-edge %s; package %s" % (model.description, WEIGHTS_VERSION, __version__))
    t0 = time.perf_counter()
    disp = model(L, R)
    dt = time.perf_counter() - t0
    print("input: %s; inference %.3f s" % (model.last_input, dt))
    print("disparity: min %.2f  median %.2f  max %.2f px" % (disp.min(), float(np.median(disp)), disp.max()))
    stem = os.path.splitext(os.path.basename(a.left))[0]
    written, info = _infer.write_outputs(disp, a.out_dir, stem, a.fx, a.baseline)
    if info:
        print("depth: median %.2f m, %.1f %% of pixels valid" % (info["depth_median_m"], 100 * info["valid_share"]))
    print("wrote: " + ", ".join(written) + " in " + a.out_dir)
    return 0


def cmd_check(a):
    import numpy as np
    man = json.load(open(os.path.join(DATA, "reference", "manifest.json"), encoding="utf-8"))
    print("nrs-edge check: package %s, weights nrs-edge %s, Python %s on %s %s" % (
        __version__, WEIGHTS_VERSION, platform.python_version(), platform.system(), platform.machine()))
    backends = ["onnx", "torch"] if a.backend == "both" else [a.backend]
    ok_all = True
    for b in backends:
        try:
            model = load(b, 400, a.provider, a.device, a.threads)
        except Exception as e:
            print("  %s: could not load: %s" % (b, e))
            return 2
        print("  model: " + model.description)
        for p in man["pairs"]:
            ref = np.load(os.path.join(DATA, "reference", p["file"]))
            disp = model(os.path.join(DATA, p["id"] + "_left.png"), os.path.join(DATA, p["id"] + "_right.png"))
            d = np.abs(disp - ref)
            med, p99, mx = float(np.median(d)), float(np.percentile(d, 99)), float(d.max())
            ok = med <= MEDIAN_MAX_PX and p99 <= P99_MAX_PX
            ok_all &= ok
            print("  %s: |difference to our reference| median %.1e  p99 %.1e  max %.1e px  -> %s" % (
                p["id"], med, p99, mx, "PASS" if ok else "FAIL"))
    print("nrs-edge check: %s (limits for each pair: median %.2f px, 99th percentile %.1f px)" % (
        "PASS" if ok_all else "FAIL", MEDIAN_MAX_PX, P99_MAX_PX))
    return 0 if ok_all else 1


def cmd_download(a):
    names = files() if a.all else [KINDS.get(n, n) for n in (a.file or ["onnx400"])]
    for n in names:
        print(fetch(n))
    return 0


def cmd_info(a):
    t = table()
    print("nrs-edge package %s; weights nrs-edge %s (Apache-2.0)" % (__version__, WEIGHTS_VERSION))
    print("by Necessity Robotics; network design: Lite Any Stereo V2-S by Jing et al. (MIT code, unmodified)")
    print("code, model card, Linux and Jetson guide: " + GITHUB)
    print("weights: https://huggingface.co/%s (revision %s)" % (t["repo"], t["revision"][:12]))
    print("files: " + ", ".join("%s (%.0f MB)" % (n, v["bytes"] / 1e6) for n, v in sorted(t["files"].items())))
    print("limits: a RECTIFIED pair is needed; at most 192 px of disparity; no confidence output; not for safety-critical")
    print("        use without an independent safety layer. The model card has the accuracy numbers and every limit.")
    if os.environ.get("NRS_EDGE_DIR"):
        print("NRS_EDGE_DIR is set: weight files are read from " + os.environ["NRS_EDGE_DIR"])
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="nrs-edge", description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split("\n", 1)[1])
    ap.add_argument("--version", action="version", version="nrs-edge %s (weights nrs-edge %s)" % (__version__, WEIGHTS_VERSION))
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="disparity (and depth) of one rectified stereo pair")
    p.add_argument("--left", required=True)
    p.add_argument("--right", required=True)
    p.add_argument("--out_dir", default="out")
    p.add_argument("--fx", type=float, default=None, help="rectified focal length, pixels")
    p.add_argument("--baseline", type=float, default=None, help="stereo baseline, metres")
    _add_model_args(p)
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("check", help="run the 3 bundled pairs and compare with our reference")
    p.add_argument("--backend", default="onnx", choices=["onnx", "torch", "both"])
    p.add_argument("--provider", default="cpu", choices=["cpu", "cuda", "tensorrt"])
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--threads", type=int, default=0)
    p.set_defaults(fn=cmd_check)
    p = sub.add_parser("download", help="get weight files now and print their paths")
    p.add_argument("file", nargs="*", help="file names or kinds (%s); default onnx400" % ", ".join(KINDS))
    p.add_argument("--all", action="store_true", help="every released file")
    p.set_defaults(fn=cmd_download)
    p = sub.add_parser("info", help="versions, links, limits")
    p.set_defaults(fn=cmd_info)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
