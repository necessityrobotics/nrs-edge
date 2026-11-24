"""Model runners for the nrs-edge tools: ONNX Runtime and TensorRT (Python API). Same interface:

    r = OrtRunner("nrs-edge.onnx", provider="cpu")     # or TrtRunner("nrs-edge_fp16.engine")
    r.input_hw                                          # (H, W) the graph takes
    disp = r.run(left_nchw, right_nchw)                 # float32 (H, W) disparity in pixels
    r.describe(); r.metadata                            # strings for logs; ONNX metadata (empty for an engine)

TrtRunner needs only the `tensorrt` Python package (JetPack ships it) and the CUDA runtime library, which it calls
through ctypes (no pycuda / cuda-python / torch). TrtRunner ran on a Windows desktop (TensorRT 11.4, RTX 5090). It has
not run on a Jetson.
"""
import ctypes
import glob
import os
import sys

import numpy as np

PROVIDERS = {
    "cpu": ["CPUExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "tensorrt": [("TensorrtExecutionProvider", {"trt_fp16_enable": True}), "CUDAExecutionProvider",
                 "CPUExecutionProvider"],
}


class OrtRunner:
    def __init__(self, path, provider="cpu", threads=0):
        import onnxruntime as ort
        avail = ort.get_available_providers()
        if provider == "auto":
            provider = ("tensorrt" if "TensorrtExecutionProvider" in avail else
                        "cuda" if "CUDAExecutionProvider" in avail else "cpu")
        want = PROVIDERS[provider]
        first = want[0][0] if isinstance(want[0], tuple) else want[0]
        if first not in avail:
            raise RuntimeError("%s is not in this onnxruntime build (available: %s)" % (first, avail))
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(path, so, providers=want)
        ins = {i.name: i.shape for i in self.sess.get_inputs()}
        assert set(ins) == {"left", "right"}, ins
        self.input_hw = tuple(int(x) for x in ins["left"][2:])
        self.metadata = dict(self.sess.get_modelmeta().custom_metadata_map)
        self.version = ort.__version__
        self.path = path

    def run(self, left, right):
        return self.sess.run(["disparity"], {"left": left, "right": right})[0][0, 0]

    def describe(self):
        return "onnxruntime %s, providers %s, %s" % (self.version, self.sess.get_providers(), os.path.basename(self.path))


def _load_cudart():
    if sys.platform == "win32":
        roots = [os.environ.get("CUDA_PATH", "")] + glob.glob(r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*")
        names = [p for r in roots if r for p in sorted(glob.glob(os.path.join(r, "bin", "**", "cudart64_*.dll"),
                                                                 recursive=True), reverse=True)]
        names += ["cudart64_13.dll", "cudart64_12.dll", "cudart64_110.dll"]
    else:
        names = ["libcudart.so", "libcudart.so.13", "libcudart.so.12", "libcudart.so.11.0"]
        names += sorted(glob.glob("/usr/local/cuda*/lib64/libcudart.so*"), reverse=True)
        names += sorted(glob.glob("/usr/lib/aarch64-linux-gnu/libcudart.so*"), reverse=True)
    for n in names:
        try:
            return ctypes.CDLL(n)
        except OSError:
            continue
    raise RuntimeError("CUDA runtime library (libcudart) not found; tried %s" % names[:6])


class TrtRunner:
    H2D, D2H = 1, 2

    def __init__(self, engine_path):
        import tensorrt as trt
        self.trt, self.version, self.path = trt, trt.__version__, engine_path
        cu = self.cu = _load_cudart()
        cu.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        cu.cudaFree.argtypes = [ctypes.c_void_p]
        cu.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        cu.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        cu.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        cu.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("could not deserialise %s (built with another TensorRT version or GPU?)" % engine_path)
        self.ctx = self.engine.create_execution_context()
        self.io = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            if any(d < 0 for d in shape):
                raise RuntimeError("dynamic shape %s for %s; build a fixed-size engine" % (shape, name))
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            nbytes = int(np.prod(shape)) * dtype.itemsize
            ptr = ctypes.c_void_p()
            self._ck(cu.cudaMalloc(ctypes.byref(ptr), nbytes), "cudaMalloc")
            self.ctx.set_tensor_address(name, ptr.value)
            self.io[name] = {"shape": shape, "dtype": dtype, "ptr": ptr, "nbytes": nbytes,
                             "input": self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT}
        assert {n for n, m in self.io.items() if m["input"]} == {"left", "right"}, list(self.io)
        self.input_hw = tuple(int(x) for x in self.io["left"]["shape"][2:])
        self.stream = ctypes.c_void_p()
        self._ck(cu.cudaStreamCreate(ctypes.byref(self.stream)), "cudaStreamCreate")
        self.metadata = {}

    @staticmethod
    def _ck(err, what):
        if err != 0:
            raise RuntimeError("%s failed with CUDA error %d" % (what, err))

    def run(self, left, right):
        for name, arr in (("left", left), ("right", right)):
            m = self.io[name]
            a = np.ascontiguousarray(arr, dtype=m["dtype"]).reshape(m["shape"])
            self._ck(self.cu.cudaMemcpy(m["ptr"], a.ctypes.data, m["nbytes"], self.H2D), "cudaMemcpy H2D")
        if not self.ctx.execute_async_v3(self.stream.value):
            raise RuntimeError("execute_async_v3 failed")
        self._ck(self.cu.cudaStreamSynchronize(self.stream), "cudaStreamSynchronize")
        m = self.io["disparity"]
        out = np.empty(m["shape"], m["dtype"])
        self._ck(self.cu.cudaMemcpy(out.ctypes.data, m["ptr"], m["nbytes"], self.D2H), "cudaMemcpy D2H")
        return out.astype(np.float32).reshape(m["shape"][-2:])

    def describe(self):
        return "TensorRT %s engine %s" % (self.version, os.path.basename(self.path))

    def __del__(self):
        try:
            for m in self.io.values():
                self.cu.cudaFree(m["ptr"])
            self.cu.cudaStreamDestroy(self.stream)
        except Exception:
            pass
