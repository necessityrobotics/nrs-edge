#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics
"""Check an nrs-edge install: runs the 3 bundled pairs through infer.py (PyTorch and ONNX Runtime, CPU) and compares
each disparity with our stored FP32 reference (device/selftest_data/reference).

    python tests/check_examples.py                 # both paths
    python tests/check_examples.py --skip_torch    # ONNX Runtime only
    python tests/check_examples.py --skip_onnx     # PyTorch only

Run it from the repository folder, after the weights are downloaded (README, Install). It writes into ./out_check/ only.
Exit code 0 = PASS, 1 = FAIL, 2 = could not run. On GitHub Actions the result is also printed as an annotation.
"""
import argparse
import os
import platform
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "device", "selftest_data")
PAIRS = ("pair_01", "pair_02", "pair_03")


def run(cmd):
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def annotate(kind, title, text):
    if os.environ.get("GITHUB_ACTIONS") == "true":
        esc = lambda s: s.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")  # noqa: E731
        print("::%s title=%s::%s" % (kind, esc(title).replace(":", "%3A").replace(",", "%2C"), esc(text[:8000])))


def one_path(name, model_args, out):
    """infer.py on the 3 pairs, then the self-test comparison of the written disparities with the reference."""
    shutil.rmtree(out, ignore_errors=True)
    tmp = os.path.join(out, "as_selftest")
    os.makedirs(tmp)
    log = []
    for p in PAIRS:
        rc, txt = run([sys.executable, "infer.py"] + model_args + [
            "--left", os.path.join(DATA, p + "_left.png"), "--right", os.path.join(DATA, p + "_right.png"),
            "--out_dir", os.path.join(out, "infer")])
        log.append(txt.strip())
        if rc != 0:
            return 2, "\n".join(log)
        shutil.copyfile(os.path.join(out, "infer", p + "_left_disp.npy"), os.path.join(tmp, p + "_disp.npy"))
    rc, txt = run([sys.executable, os.path.join("device", "selftest.py"), "--from_dir", tmp])
    log.append(txt.strip())
    return rc, "\n".join(log)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="nrs-edge.safetensors")
    ap.add_argument("--onnx", default="nrs-edge_w640h400_fp32.onnx")
    ap.add_argument("--skip_torch", action="store_true")
    ap.add_argument("--skip_onnx", action="store_true")
    a = ap.parse_args()
    where = "%s %s, Python %s" % (platform.system(), platform.machine(), platform.python_version())
    print("nrs-edge install check on " + where)
    todo = []
    if not a.skip_torch:
        todo.append(("PyTorch, CPU", ["--weights", a.weights, "--device", "cpu"], a.weights))
    if not a.skip_onnx:
        todo.append(("ONNX Runtime, CPU", ["--onnx", a.onnx], a.onnx))
    worst = 0
    lines = []
    for name, args, need in todo:
        if not os.path.isfile(os.path.join(ROOT, need)):
            print("\n== %s: %s is not in this folder. Download the weights first (README, Install)." % (name, need))
            worst = max(worst, 2)
            lines.append("%s: could not run (%s missing)" % (name, need))
            continue
        rc, txt = one_path(name, args, os.path.join(ROOT, "out_check", name.split(",")[0].replace(" ", "_").lower()))
        print("\n== %s\n%s" % (name, txt))
        worst = max(worst, rc)
        res = {0: "PASS", 1: "FAIL"}.get(rc, "could not run")
        last = [ln for ln in txt.splitlines() if "median" in ln.lower()][-3:]
        lines.append("%s: %s%s" % (name, res, ("\n  " + "\n  ".join(s.strip() for s in last)) if last else ""))
    verdict = {0: "PASS", 1: "FAIL"}.get(worst, "COULD NOT RUN")
    summary = "nrs-edge install check: %s (%s)\n%s" % (verdict, where, "\n".join(lines))
    print("\n" + summary)
    annotate("notice" if worst == 0 else "error", "nrs-edge install check (%s %s, Python %s)" % (
        platform.system(), platform.machine(), platform.python_version()), summary)
    return worst


if __name__ == "__main__":
    sys.exit(main())
