# ops_loader.py
import os
import torch
from torch.utils.cpp_extension import load

# 固定存放已编译 .so 的路径
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
SO_DIR = os.path.join(parent_dir, "compiled_ops")
SO_PATH = os.path.join(SO_DIR, "pycuda_op.so")

def pycuda_ops():
    if os.path.exists(SO_PATH):
        print(f"[OpsLoader] Loading precompiled ops from {SO_PATH}")
        torch.ops.load_library(SO_PATH)
        return torch.ops
    else:
        print("[OpsLoader] No precompiled ops found. Compiling...")
        os.makedirs(SO_DIR, exist_ok=True)
        ops = load(
            name="pycuda_op",
            sources=[
                os.path.join(os.path.dirname(__file__), "binding.cpp"),
                os.path.join(os.path.dirname(__file__), "sum_pooling_bw.cu"),
                os.path.join(os.path.dirname(__file__), "sum_pooling_fw.cu")
            ],
            extra_cflags=["-O3"],
            extra_cuda_cflags=[
                "-O3", "-lineinfo",
                "-U__CUDA_NO_BFLOAT16__",
                "-gencode=arch=compute_80,code=sm_80"  #根据机器类型改
            ],
            verbose=True
        )
        # 编译完成后，把生成的 .so 拷贝到固定位置
        import shutil
        compiled_path = ops.__file__
        shutil.copy2(compiled_path, SO_PATH)
        print(f"[OpsLoader] Compiled and saved to {SO_PATH}")
        return ops


if __name__ == "__main__":
    pycuda_ops()
