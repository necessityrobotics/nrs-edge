"""Image and output helpers shared by the nrs-edge tools (numpy + OpenCV only; no torch, so they run on a Jetson).

Conventions
  * Model input: two rectified images, RGB, float32, 0-255, NCHW. The model normalises internally. Grayscale
    images (OAK-D Lite) are replicated to 3 channels.
  * Model output: disparity of the left image in pixels, float32, shape (1, 1, H, W).
  * Size: an input smaller than the model input is padded with edge replication, centred (the authors'
    InputPadder 'sintel' mode), and the output is cropped back. A larger input is resized to the model size and
    the disparity is scaled back to the input width (disparity is in pixels of the image it refers to).
  * Depth (optional): depth_m = fx_px * baseline_m / disparity_px for disparity > 0, NaN elsewhere.
"""
import cv2
import numpy as np


def read_rgb(path):
    """HxWx3 float32 RGB in 0-255. Grayscale files are replicated to 3 channels."""
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32)


def to_nchw(img):
    return np.ascontiguousarray(img.transpose(2, 0, 1)[None], dtype=np.float32)


def centred_pad(h, w, H, W):
    """(top, bottom, left, right) to pad an h x w image to H x W, split like InputPadder(mode='sintel')."""
    ph, pw = H - h, W - w
    assert ph >= 0 and pw >= 0
    return ph // 2, ph - ph // 2, pw // 2, pw - pw // 2


class Fit:
    """Maps an input pair to the model's fixed input size and the model output back to the input size."""

    def __init__(self, h, w, H, W):
        self.h, self.w, self.H, self.W = h, w, H, W
        self.mode = "pad" if (h <= H and w <= W) else "resize"
        self.pad = centred_pad(h, w, H, W) if self.mode == "pad" else (0, 0, 0, 0)

    def apply(self, img_hwc):
        if self.mode == "pad":
            t, b, l, r = self.pad
            return np.pad(img_hwc, ((t, b), (l, r), (0, 0)), mode="edge")
        return cv2.resize(img_hwc, (self.W, self.H), interpolation=cv2.INTER_AREA)

    def invert(self, disp_hw):
        if self.mode == "pad":
            t, _, l, _ = self.pad
            return disp_hw[t:t + self.h, l:l + self.w]
        scale = self.w / float(self.W)
        return cv2.resize(disp_hw, (self.w, self.h), interpolation=cv2.INTER_LINEAR) * np.float32(scale)

    def describe(self):
        if self.mode == "pad" and not any(self.pad):
            return "%dx%d, no padding needed" % (self.w, self.h)
        if self.mode == "pad":
            return "padded %dx%d -> %dx%d (top/bottom/left/right %s, edge replication)" % (
                self.w, self.h, self.W, self.H, self.pad)
        return ("RESIZED %dx%d -> %dx%d; disparity scaled back by %.4f. Resizing changes the image; prefer an input "
                "at or below the model size" % (self.w, self.h, self.W, self.H, self.w / float(self.W)))


def depth_from_disparity(disp, fx_px, baseline_m):
    depth = np.full(disp.shape, np.nan, dtype=np.float32)
    ok = np.isfinite(disp) & (disp > 0)
    depth[ok] = np.float32(fx_px * baseline_m) / disp[ok]
    return depth


def colourise(values, vmin=None, vmax=None, invalid=None):
    """TURBO colour map, BGR uint8 (for cv2.imwrite). Range defaults to the 1st-99th percentile of valid values."""
    v = values.astype(np.float32)
    ok = np.isfinite(v) if invalid is None else (np.isfinite(v) & ~invalid)
    if not ok.any():
        return np.zeros(v.shape + (3,), np.uint8)
    lo = np.percentile(v[ok], 1) if vmin is None else vmin
    hi = np.percentile(v[ok], 99) if vmax is None else vmax
    x = np.clip((np.where(ok, v, lo) - lo) / max(hi - lo, 1e-6), 0, 1)
    out = cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    out[~ok] = 0
    return out


def abs_diff_stats(a, b):
    d = np.abs(a.astype(np.float64) - b.astype(np.float64)).ravel()
    return {"median_px": float(np.median(d)), "p99_px": float(np.percentile(d, 99)), "max_px": float(d.max()),
            "mean_px": float(d.mean()), "n": int(d.size)}
