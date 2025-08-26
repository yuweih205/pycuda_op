# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os, sys

# ---------- 1) 自动设置编译架构 ----------
# 若外部未指定，则默认用 "native"（按当前机器 GPU 自动探测）
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "native")

def gencode_flags():
    """
    若外部设置了 TORCH_CUDA_ARCH_LIST（含我们默认的 'native'），
    让 PyTorch 自己生成 -gencode；这里返回空即可。
    如果你想在无 GPU 的 CI 上固定多架构，可把上面的 setdefault 去掉，
    然后在此返回固定列表，比如 8.0/8.9/9.0。
    """
    if os.getenv("TORCH_CUDA_ARCH_LIST"):
        return []
    # 兜底（极少走到）：A100(8.0) + H100(9.0)
    return [
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_90,code=sm_90",
    ]

# ---------- 2) 自动定位 Torch 的 lib 目录并写入 rpath ----------
def detect_torch_libdir():
    try:
        import torch
        from torch.utils.cpp_extension import library_paths
        paths = library_paths()  # 通常返回 [<site-packages>/torch/lib]
        for p in paths:
            if os.path.isdir(p):
                return p
        # 兜底猜测
        guess = os.path.join(os.path.dirname(torch.__file__), "lib")
        return guess if os.path.isdir(guess) else None
    except Exception:
        return None

def rpath_flags():
    """在链接阶段注入 rpath，这样运行时无需 LD_LIBRARY_PATH 也能找到 libc10.so 等。"""
    if os.environ.get("SKIP_RPATH"):
        return []
    libdir = detect_torch_libdir()
    if sys.platform.startswith("linux"):
        flags = ["-Wl,-rpath,$ORIGIN"]  # 先相对自身
        if libdir:
            flags.append(f"-Wl,-rpath,{libdir}")  # 再加 torch/lib
        return flags
    elif sys.platform == "darwin":
        flags = ["-Wl,-rpath,@loader_path"]
        if libdir:
            flags.append(f"-Wl,-rpath,{libdir}")
        return flags
    return []

ext = CUDAExtension(
    name="pycuda_op._C",  # TORCH_EXTENSION_NAME -> "_C"
    sources=[
        "src/binding.cpp",
        "src/sum_pooling_fw.cu",
        "src/sum_pooling_bw.cu",
    ],
    extra_compile_args={
        "cxx": ["-O3"],
        "nvcc": ["-O3", "-lineinfo", "-U__CUDA_NO_BFLOAT16__", *gencode_flags()],
    },
    extra_link_args=rpath_flags(),  # 关键：带上 rpath
)

setup(
    name="pycuda_op",
    version="0.0.1",
    packages=["pycuda_op"],
    package_dir={"pycuda_op": "pycuda_op"},
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
    description="PyBind11-based CUDA ops",
)
