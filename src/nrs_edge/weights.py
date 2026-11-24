# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics
"""The released weight files of nrs-edge: where they are, their SHA-256, and how to get them.

Files come from the Hugging Face repository `Pyassahu/nrs-edge`, at one fixed revision. Every file is checked against
the SHA-256 stored in this package (`_files.json`) before it is used. No login is needed.

To work without a network, put the files into one folder and set the environment variable NRS_EDGE_DIR to it.
"""
import hashlib
import json
import os

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


def _check(path, name):
    want = table()["files"][name]
    got = sha256_file(path)
    if got != want["sha256"]:
        raise RuntimeError("%s has the SHA-256 %s; the released file has %s. Delete the file and try again."
                           % (path, got, want["sha256"]))
    return path


def fetch(name_or_kind, quiet=False):
    """Local path of one released file (downloaded on the first call, then read from the cache). The file is checked.

    name_or_kind: a file name from files(), or one of "onnx400", "onnx480", "onnx400_fp16", "onnx480_fp16", "torch".
    """
    name = KINDS.get(name_or_kind, name_or_kind)
    t = table()
    if name not in t["files"]:
        raise ValueError("unknown file %r; known: %s; kinds: %s" % (name_or_kind, ", ".join(files()), ", ".join(KINDS)))
    local = os.environ.get("NRS_EDGE_DIR")
    if local:
        p = os.path.join(local, name)
        if not os.path.isfile(p):
            raise FileNotFoundError("NRS_EDGE_DIR is set to %s, but %s is not in it" % (local, name))
        return _check(p, name)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:                                        # pragma: no cover
        raise RuntimeError("the package huggingface_hub is needed to download the weights (pip install huggingface_hub), "
                           "or set NRS_EDGE_DIR to a folder that holds the files") from e
    if not quiet:
        mb = t["files"][name]["bytes"] / 1e6
        print("nrs-edge: %s (%.0f MB) from huggingface.co/%s; it is downloaded one time and then read from the cache"
              % (name, mb, t["repo"]))
    p = hf_hub_download(repo_id=t["repo"], filename=name, revision=t["revision"])
    return _check(p, name)
