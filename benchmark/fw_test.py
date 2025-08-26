# test_sum_pooling.py
import os
import math
import time
import random
import torch

# ---------------------------------------------
# 1) 加载扩展（两种路径都支持）
# ---------------------------------------------
def load_ops():
    try:
        import pycuda_op as m  # 直接 import 已编译的 pybind11 模块
        return m
    except Exception:
        from ops_loader import pycuda_op
        return pycuda_op()

ops = load_ops()

# ---------------------------------------------
# 2) 合成数据（可控规模/分布）
#    dims: 每样本列数 ∈ [16, 128]
#    features: 每样本行数 ∈ [1, 10]
# ---------------------------------------------
def make_synth(batchsize=4096*1024, dim_lo=4, dim_hi=128, feat_lo=1, feat_hi=10, seed=123):
    g = random.Random(seed)
    dims = torch.empty(batchsize, dtype=torch.int32)
    feats = torch.empty(batchsize, dtype=torch.int32)
    for i in range(batchsize):
        dims[i]  = g.randint(dim_lo, dim_hi)
        feats[i] = g.randint(feat_lo, feat_hi)

    # 前缀和： offsets[i+1] = offsets[i] + feats[i] * dims[i]
    offsets = torch.zeros(batchsize + 1, dtype=torch.int32)
    output_offsets = torch.zeros(batchsize + 1, dtype=torch.int32)
    for i in range(batchsize):
        offsets[i+1]        = offsets[i]        + feats[i] * dims[i]
        output_offsets[i+1] = output_offsets[i] + dims[i]

    total_values = int(offsets[-1].item())
    out_total    = int(output_offsets[-1].item())

    # 生成 bf16 values（按扁平拼接）
    values = torch.randn(total_values, dtype=torch.bfloat16)

    # 目标输出缓冲区（fp32）
    out = torch.zeros(out_total, dtype=torch.bfloat16)

    # 为了 GPU 跑，拷到 cuda
    dev = torch.device("cuda")
    return {
        "dims": dims.to(dev),
        "feats": feats,  # 仅供 CPU 参考实现使用，可不搬到 GPU
        "offsets": offsets.to(dev),
        "output_offsets": output_offsets.to(dev),
        "values": values.to(dev),
        "out": out.to(dev),
        "meta": dict(B=batchsize,
                     total_values=total_values,
                     out_total=out_total)
    }

# ---------------------------------------------
# 3) 参考实现（CPU，便于校验）
# ---------------------------------------------
def ref_sum(values_bf16_cpu, offsets_cpu, dims_cpu, feats_cpu, output_offsets_cpu):
    out = torch.zeros(int(output_offsets_cpu[-1].item()), dtype=torch.float32)
    values_f = values_bf16_cpu.to(torch.float32)
    B = dims_cpu.numel()
    for s in range(B):
        dim = int(dims_cpu[s].item())
        sb  = int(offsets_cpu[s].item())
        base= int(output_offsets_cpu[s].item())
        F   = int(feats_cpu[s].item())
        # 将该样本的 F 行拼起来：每行长度 dim
        # 累加到 out[base:base+dim]
        for f in range(F):
            row = sb + f * dim
            out[base:base+dim] += values_f[row:row+dim]
    return out

# ---------------------------------------------
# 4) 计时工具
# ---------------------------------------------
def time_kernel(fn, warmup=10, iters=50):
    # 预热
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop  = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    stop.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(stop) / iters  # 单次平均 ms
    return ms

# ---------------------------------------------
# 5) 主流程：跑 kernel、校验、统计带宽
# ---------------------------------------------
def main():
    assert torch.cuda.is_available(), "需要有可用 CUDA 设备"
    # 建议设置：os.environ["CUDA_LAUNCH_BLOCKING"]="0"（event 计时足够了）

    B = 4096 * 128
    data = make_synth(batchsize=B, dim_lo=16, dim_hi=128, feat_lo=1, feat_hi=10, seed=20250813)

    values = data["values"]
    offsets = data["offsets"]
    dims = data["dims"]
    ooffs = data["output_offsets"]
    out = data["out"]

    # 记录下字节量（粗略核算带宽）
    total_values = data["meta"]["total_values"]
    out_total = data["meta"]["out_total"]
    bytes_read  = total_values * 2  # bf16 -> 2B
    bytes_write = out_total * 4    # fp32 -> 4B
    # 元数据读（offsets/dims/ooffs），体量小但可计入
    meta_bytes  = (offsets.numel() + ooffs.numel()) * 4 + dims.numel() * 4
    total_bytes = bytes_read + bytes_write + meta_bytes

    # 封装调用（注意参数顺序）
    def call_fw():
        # sum_pooling_fw(values_bf16, offsets_i32, dims_i32, output_f32, output_offsets_i32)
        ops.sum_pooling_fw(values, offsets, dims, out, ooffs)
        print(values.dtype, offsets.dtype, dims.dtype, out.dtype, ooffs.dtype)
# 期望：values/out=bfloat16；offsets/dims/ooffs=int32

    # 1) 预跑一次，确保 lazy init 完成
    call_fw()
    torch.cuda.synchronize()

    # 2) 计时
    ms = time_kernel(call_fw, warmup=5, iters=30)

    # 3) 校验（把 GPU 结果/输入拷回 CPU 对比）
    out_cpu = out.detach().cpu()
    ref_cpu = ref_sum(values.detach().cpu(), offsets.detach().cpu(), dims.detach().cpu(),
                      data["feats"], ooffs.detach().cpu())

    max_abs = (out_cpu - ref_cpu).abs().max().item()
    passed = max_abs <= 1e-3  # bf16→fp32 累加，阈值放宽到 1e-3 较合理

    # 4) 统计吞吐与带宽
    elems_accum = out_total  # 每个输出元素是若干行相加，但“计算量”与特征有关；这里只给 IO 视角
    gbps = total_bytes / (ms * 1e-3) / 1e9
    giBps = total_bytes / (ms * 1e-3) / (1024**3)

    dev = torch.cuda.get_device_name()
    cc  = torch.cuda.get_device_capability()
    print("="*80)
    print(f"Device: {dev} (CC {cc[0]}.{cc[1]})")
    print(f"B={B}, total_values={total_values}, out_total={out_total}")
    print(f"Avg kernel time: {ms:.3f} ms  |  total IO ~ {total_bytes/1e9:.3f} GB")
    print(f"Est. Bandwidth: {gbps:.2f} GB/s  ({giBps:.2f} GiB/s)")
    print(f"Max abs diff : {max_abs:.4e}  -> {'PASSED' if passed else 'FAILED'}")
    print("="*80)

    # 失败时给一点提示
    if not passed:
        # 打印前几个不一致位置
        diff = (out_cpu - ref_cpu).abs()
        idx = torch.nonzero(diff > 1e-3).flatten()[:10]
        print("First mismatches (idx, gpu, ref, err):")
        for i in idx.tolist():
            print(i, float(out_cpu[i].item()), float(ref_cpu[i].item()), float(diff[i].item()))
        raise SystemExit(1)

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    main()

