# pycuda_op/pycuda_op/__init__.py
import torch 
try:
    from . import _C

except ImportError:
    print("错误：无法导入 C++/CUDA 核心扩展模块 (_C.so)。")
    print("请确保你已经使用 'pip install .' 成功编译并安装了该包。")
    raise

sum_pooling_fw = _C.sum_pooling_fw
sum_pooling_bw = _C.sum_pooling_bw

__all__ = [
    "_C",
    "sum_pooling_fw",
    "sum_pooling_bw",
]