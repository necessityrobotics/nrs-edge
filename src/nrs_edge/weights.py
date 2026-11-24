# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics
"""The released weight files of nrs-edge: where they are, their SHA-256, and how to get them.

Files come from the Hugging Face repository `Pyassahu/nrs-edge`, at one fixed revision. Every file is checked against
the SHA-256 stored in this package (`_files.json`) before it is used. No login is needed.

To work without a network, put the files into one folder and set the environment variable NRS_EDGE_DIR to it. One way
to fill that folder on a computer with a network:
    hf download Pyassahu/nrs-edge nrs-edge_w640h400_fp32.onnx nrs-edge_w640h480_fp32.onnx --local-dir <folder>
"""
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_TABLE = None

KINDS = {
    "onnx400": "nrs-edge_w640h400_fp32.onnx",       # ONNX graph for pictures up to 640 x 400
    "onnx480": "nrs-edge_w640h480_fp32.onnx",       # ONNX graph for pictures up to 640 x 480
    "onnx400_fp16": "nrs-edge_w640h400_fp16.onnx",  # for TensorRT FP16 builds
    "onnx480_fp16": "nrs-edge_w640h480_fp16.onnx",
    "torch": "nrs-edge.safetensors",                # PyTorch weights
}


def table():
    """{"repo": ..., "revision": ..., "files": {name: {"sha256": ..., "bytes": ...}}}"""
    global _TABLE
    if _TABLE is None:
        with open(os.path.join(_HERE, "_files.json"), encoding="utf-8") as f:
            _TABLE = json.load(f)
    return _TABLE


def files():
    """Names of the released weight files."""
    return sorted(table()["files"])


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _ok(path, name):
    return sha256_file(path) == table()["files"][name]["sha256"]


def fetch(name_or_kind, quiet=False):
    """Local path of one released file (downloaded on the first call, then read from the cache). The file is checked.

    name_or_kind: a file name from files(), or one of "onnx400", "onnx480", "onnx400_fp16", "onnx480_fp16", "torch".
    Messages go to stderr; quiet=True prints none.
    """
    name = KINDS.get(name_or_kind, name_or_kind)
    t = table()
    if name not in t["files"]:
        raise ValueError("unknown file %r; known: %s; kinds: %s" % (name_or_kind, ", ".join(files()), ", ".join(KINDS)))
    want = t["files"][name]
    local = os.environ.get("NRS_EDGE_DIR")
    if local:
        p = os.path.join(os.path.expanduser(local), name)
        if not os.path.isfile(p):
            raise FileNotFoundError("NRS_EDGE_DIR is set to %s, but %s is not in it" % (local, name))
        if not _ok(p, name):
            raise RuntimeError("%s is not the released file (its SHA-256 is not %s). Replace it." % (p, want["sha256"]))
        return p
    try:
        from huggingface_hub import hf_hub_download, try_to_load_from_cache
    except ImportError as e:                                        # pragma: no cover
        raise RuntimeError("the package huggingface_hub is needed to download the weights (pip install huggingface_hub), "
                           "or set NRS_EDGE_DIR to a folder that holds the files") from e
    cached = try_to_load_from_cache(repo_id=t["repo"], filename=name, revision=t["revision"])
    if isinstance(cached, str) and os.path.isfile(cached) and _ok(cached, name):
        return cached
    if not quiet:
        print("nrs-edge: downloading %s (%.0f MB) from huggingface.co/%s; this happens one time"
              % (name, want["bytes"] / 1e6, t["repo"]), file=sys.stderr)
    p = hf_hub_download(repo_id=t["repo"], filename=name, revision=t["revision"])
    if not _ok(p, name):                                            # a damaged file in the cache: get it again, one time
        p = hf_hub_download(repo_id=t["repo"], filename=name, revision=t["revision"], force_download=True)
        if not _ok(p, name):
            raise RuntimeError("%s does not have the SHA-256 of the released file (%s) after a second download"
                               % (p, want["sha256"]))
    return p
