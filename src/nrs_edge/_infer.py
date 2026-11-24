#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics
"""nrs-edge inference: disparity, and optionally metric depth, from one rectified stereo pair.

  PyTorch (CPU or CUDA):
    python infer.py --weights nrs-edge.safetensors --left L.png --right R.png --out_dir out [--device cuda]
  ONNX Runtime (CPU by default):
    python infer.py --onnx nrs-edge_w640h400_fp32.onnx --left L.png --right R.png --out_dir out
  Depth in metres as well:
    python infer.py ... --fx 471.049 --baseline 0.07514

Input: a RECTIFIED left/right pair (rows aligned), same size, any 8-bit image that OpenCV reads. Grayscale images
are read as three equal channels.
  * PyTorch: the pair is padded to a multiple of 32 (edge replication, centred) and the output is cropped back.
  * ONNX: the graph has a fixed size. A smaller pair is padded to it; a larger pair is resized to it and the
    disparity is scaled back to the input width (resizing loses detail).

Outputs in --out_dir, named after the left image:
  <name>_disp.npy       float32 HxW disparity of the left image, pixels
  <name>_disp16.png     16-bit PNG, value = round(disparity x 256) (the KITTI convention); 0 = no value
  <name>_disp_color.png colour preview (0 to 192 px)
  <name>_depth.npy      float32 HxW depth in metres, NaN where disparity <= 0     (with --fx and --baseline)
  <name>_depth16.png    16-bit PNG, depth in millimetres, 0 = no value or above 65.535 m  (with --fx and --baseline)

The network code is the LiteAnyStereo authors' MIT code in third_party/LiteAnyStereo (unmodified).
"""
import argparse
import json
import os
import sys
import time

# (infer.py of the repository sets sys.dont_write_bytecode here; a library must not. This file is generated from infer.py: do not edit it.)
HERE = os.path.dirname(os.path.abspath(__file__))
LAS_DIR = os.path.join(HERE, "third_party", "LiteAnyStereo")
MAX_DISP = 192


def read_rgb(path):
    import cv2
    import numpy as np
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32)


def centred_pad(h, w, H, W):
    ph, pw = H - h, W - w
    return ph // 2, ph - ph // 2, pw // 2, pw - pw // 2


class Fit:
    """Maps a pair to a fixed graph size (pad if it fits, else resize) and the output back."""

    def __init__(self, h, w, H, W):
        self.h, self.w, self.H, self.W = h, w, H, W
        self.mode = "pad" if (h <= H and w <= W) else "resize"
        self.pad = centred_pad(h, w, H, W) if self.mode == "pad" else (0, 0, 0, 0)

    def apply(self, img):
        import cv2
        import numpy as np
        if self.mode == "pad":
            t, b, l, r = self.pad
            return np.pad(img, ((t, b), (l, r), (0, 0)), mode="edge")
        return cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)

    def invert(self, d):
        import cv2
        import numpy as np
        if self.mode == "pad":
            t, _, l, _ = self.pad
            return d[t:t + self.h, l:l + self.w]
        return cv2.resize(d, (self.w, self.h), interpolation=cv2.INTER_LINEAR) * np.float32(self.w / float(self.W))

    def describe(self):
        if self.mode == "pad":
            return "%dx%d padded to %dx%d (edge replication)" % (self.w, self.h, self.W, self.H) if any(self.pad) \
                else "%dx%d, no padding" % (self.w, self.h)
        return "RESIZED %dx%d -> %dx%d; disparity scaled back" % (self.w, self.h, self.W, self.H)


def nchw(img):
    import numpy as np
    return np.ascontiguousarray(img.transpose(2, 0, 1)[None], dtype=np.float32)


def load_weights(path):
    """State dict from .safetensors (release format) or a .pth state dict; returns (state, metadata)."""
    import torch
    if path.endswith(".safetensors"):
        from safetensors import safe_open
        from safetensors.torch import load_file
        with safe_open(path, "pt") as f:
            meta = f.metadata() or {}
        meta = json.loads(meta["nrs"]) if "nrs" in meta else meta
        return load_file(path), meta
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    return sd, {}


def build_torch(weights, device):
    import torch
    if LAS_DIR not in sys.path:
        sys.path.insert(0, LAS_DIR)
    from core.models import build_model, load_model_weights
    model = build_model("las2", fnet_pretrained=False, model_size="s", max_disp=MAX_DISP)
    sd, meta = load_weights(weights)
    load_model_weights(model, sd, strict=True)
    return model.to(device).eval(), meta


class TorchRunner:
    def __init__(self, weights, device="cpu", threads=0):
        import torch
        if threads:
            torch.set_num_threads(threads)
        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False            # keep FP32 FP32
            torch.backends.cudnn.allow_tf32 = False
        self.device = device
        self.model, self.metadata = build_torch(weights, device)
        self.name = "PyTorch %s on %s, %s" % (torch.__version__, device, os.path.basename(weights))

    def run(self, L, R):
        import numpy as np
        import torch
        h, w = L.shape[:2]
        H, W = -(-h // 32) * 32, -(-w // 32) * 32
        t, b, l, r = centred_pad(h, w, H, W)
        x = [torch.nn.functional.pad(torch.from_numpy(nchw(im)).to(self.device), (l, r, t, b), mode="replicate")
             for im in (L, R)]
        with torch.no_grad():
            d = self.model(x[0], x[1], max_disp=MAX_DISP, test_mode=True)
        d = d[0, 0, t:t + h, l:l + w].float().cpu().numpy()
        return np.ascontiguousarray(d, dtype=np.float32), "padded to %dx%d (multiple of 32)" % (W, H)


class OnnxRunner:
    PROVIDERS = {"cpu": ["CPUExecutionProvider"], "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
                 "tensorrt": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]}

    def __init__(self, path, provider="cpu", threads=0):
        import onnxruntime as ort
        want = self.PROVIDERS[provider]
        if want[0] not in ort.get_available_providers():
            raise SystemExit("%s is not in this onnxruntime build (%s)" % (want[0], ort.get_available_providers()))
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(path, so, providers=want)
        self.hw = tuple(int(x) for x in self.sess.get_inputs()[0].shape[2:])
        self.metadata = dict(self.sess.get_modelmeta().custom_metadata_map)
        self.name = "onnxruntime %s, %s, %s" % (ort.__version__, self.sess.get_providers()[0], os.path.basename(path))

    def run(self, L, R):
        import numpy as np
        fit = Fit(L.shape[0], L.shape[1], *self.hw)
        d = self.sess.run(["disparity"], {"left": nchw(fit.apply(L)), "right": nchw(fit.apply(R))})[0][0, 0]
        return np.ascontiguousarray(fit.invert(d), dtype=np.float32), fit.describe()


def colourise(v, vmax):
    import cv2
    import numpy as np
    x = np.clip(np.nan_to_num(v, nan=0.0) / max(vmax, 1e-6), 0, 1)
    out = cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    out[~np.isfinite(v)] = 0
    return out


def write_outputs(disp, out_dir, stem, fx=None, baseline=None, vmax=MAX_DISP):
    import cv2
    import numpy as np
    os.makedirs(out_dir, exist_ok=True)
    names = ["_disp.npy", "_disp16.png", "_disp_color.png", "_depth.npy", "_depth16.png"]
    for n in names:
        p = os.path.join(out_dir, stem + n)
        if os.path.exists(p):
            os.remove(p)                                             # no stale outputs from an earlier run
    np.save(os.path.join(out_dir, stem + "_disp.npy"), disp)
    d16 = np.clip(np.round(np.where(np.isfinite(disp) & (disp > 0), disp, 0) * 256.0), 0, 65535).astype(np.uint16)
    cv2.imwrite(os.path.join(out_dir, stem + "_disp16.png"), d16)
    cv2.imwrite(os.path.join(out_dir, stem + "_disp_color.png"), colourise(disp, vmax))
    written = names[:3]
    info = {}
    if fx is not None:
        depth = np.full(disp.shape, np.nan, np.float32)
        ok = np.isfinite(disp) & (disp > 0)
        depth[ok] = np.float32(fx * baseline) / disp[ok]
        np.save(os.path.join(out_dir, stem + "_depth.npy"), depth)
        mm = np.where(ok, np.round(depth * 1000.0), 0)
        mm[mm > 65535] = 0
        cv2.imwrite(os.path.join(out_dir, stem + "_depth16.png"), mm.astype(np.uint16))
        written += names[3:]
        if ok.any():
            info = {"depth_median_m": float(np.median(depth[ok])), "valid_share": float(ok.mean())}
    return [stem + n for n in written], info


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n", 1)[1])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--weights", help="nrs-edge.safetensors (PyTorch path)")
    g.add_argument("--onnx", help="an nrs-edge ONNX graph (ONNX Runtime path)")
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--fx", type=float, default=None, help="rectified focal length, pixels")
    ap.add_argument("--baseline", type=float, default=None, help="stereo baseline, metres")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="PyTorch device")
    ap.add_argument("--provider", default="cpu", choices=["cpu", "cuda", "tensorrt"], help="ONNX Runtime provider")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = library default)")
    ap.add_argument("--json", default=None, help="write run information to this file")
    a = ap.parse_args()
    if (a.fx is None) != (a.baseline is None):
        ap.error("--fx and --baseline go together")
    L, R = read_rgb(a.left), read_rgb(a.right)
    if L.shape != R.shape:
        sys.exit("left %s and right %s differ in size" % (L.shape, R.shape))
    runner = TorchRunner(a.weights, a.device, a.threads) if a.weights else OnnxRunner(a.onnx, a.provider, a.threads)
    md = runner.metadata
    print("model: %s; version %s" % (runner.name, md.get("version", md.get("nrs.version", "?"))))
    t0 = time.perf_counter()
    disp, how = runner.run(L, R)
    dt = time.perf_counter() - t0
    print("input: %s; inference %.3f s" % (how, dt))
    import numpy as np
    print("disparity: min %.2f  median %.2f  max %.2f px" % (disp.min(), float(np.median(disp)), disp.max()))
    stem = os.path.splitext(os.path.basename(a.left))[0]
    written, info = write_outputs(disp, a.out_dir, stem, a.fx, a.baseline)
    if info:
        print("depth: median %.2f m, %.1f %% of pixels valid" % (info["depth_median_m"], 100 * info["valid_share"]))
    print("wrote: " + ", ".join(written) + " in " + a.out_dir)
    if a.json:
        json.dump({"runner": runner.name, "input": how, "seconds": dt, "left": os.path.basename(a.left),
                   "outputs": written, "metadata": md,
                   "disparity": {"min": float(disp.min()), "max": float(disp.max())}, **info},
                  open(a.json, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
