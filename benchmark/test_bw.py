# bw_test.py
import argparse
import math
import random
import torch

def make_problem(B=4096*128, dim_lo=168, dim_hi=168, feat_lo=1, feat_hi=10, seed=123):
    rng = random.Random(seed)
    dims = torch.empty(B, dtype=torch.int32)
    feats = torch.empty(B, dtype=torch.int32)
    for i in range(B):
        dims[i]  = rng.randint(dim_lo, dim_hi)
        feats[i] = rng.randint(feat_lo, feat_hi)

    offsets = torch.zeros(B + 1, dtype=torch.int32)
    ooffs   = torch.zeros(B + 1, dtype=torch.int32)
    for i in range(B):
        offsets[i+1] = offsets[i] + dims[i] * feats[i]
        ooffs[i+1]   = ooffs[i]   + dims[i]

    total_vals = int(offsets[-1].item())
    total_go   = int(ooffs[-1].item())
    return dims, feats, offsets, ooffs, total_vals, total_go

def cpu_ref(go_f32_cpu, dims_cpu, feats_cpu, offsets_cpu, ooffs_cpu, dtype='bf16'):
    B = dims_cpu.numel()
    total_vals = int(offsets_cpu[-1].item())
    if dtype == 'bf16':
        gv = torch.empty(total_vals, dtype=torch.bfloat16)
    else:
        gv = torch.empty(total_vals, dtype=torch.float32)

    for s in range(B):
        dim   = int(dims_cpu[s])
        F     = int(feats_cpu[s])
        sb    = int(offsets_cpu[s])
        gbase = int(ooffs_cpu[s])
        row_src = go_f32_cpu[gbase:gbase+dim]
        if dtype == 'bf16':
            row_src = row_src.to(torch.bfloat16)
        for f in range(F):
            start = sb + f * dim
            gv[start:start+dim] = row_src
    return gv

def time_cuda(fn, warmup=5, iters=30):
    # 预热
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    # 计时
    start = torch.cuda.Event(enable_timing=True)
    stop  = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / iters  # ms

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=4096*1024)
    ap.add_argument("--dim_lo", type=int, default=4)
    ap.add_argument("--dim_hi", type=int, default=168)
    ap.add_argument("--feat_lo", type=int, default=1)
    ap.add_argument("--feat_hi", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20250813)
    ap.add_argument("--dtype", choices=["bf16","f32"], default="bf16",
                    help="gv 的 dtype（go 固定 float32）")
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()

    assert torch.cuda.is_available()
    device = torch.device("cuda")

    # 尝试导入：优先直接 import，失败则走 ops_loader 动态编译
    try:
        import pycuda_op as ops
    except Exception:
        from ops_loader import pycuda_ops
        ops = pycuda_ops()

    # 1) 构造问题
    dims, feats, offs, ooffs, total_vals, total_go = make_problem(
        B=args.B, dim_lo=args.dim_lo, dim_hi=args.dim_hi,
        feat_lo=args.feat_lo, feat_hi=args.feat_hi, seed=args.seed
    )

    # 2) 构造张量
    go = torch.randn(total_go, dtype=torch.bfloat16, device=device)   # grad_output
    if args.dtype == "bf16":
        gv = torch.empty(total_vals, dtype=torch.bfloat16, device=device)  # grad_values
        check_tol = 2e-2
    else:
        gv = torch.empty(total_vals, dtype=torch.float32, device=device)
        check_tol = 1e-5

    dims_d  = dims.to(device)
    offs_d  = offs.to(device)
    ooffs_d = ooffs.to(device)

    # 3) 预跑一次（lazy init）
    ops.sum_pooling_bw(go, gv, offs_d, dims_d, ooffs_d)
    torch.cuda.synchronize()

    # 4) 计时
    ms = time_cuda(lambda: ops.sum_pooling_bw(go, gv, offs_d, dims_d, ooffs_d),
                   warmup=5, iters=args.iters)

    # 5) 粗略带宽（读 go + 写 gv + meta）
    bytes_read  = total_go   * 2.0
    bytes_write = total_vals * (2.0 if args.dtype == "bf16" else 4.0)
    meta_bytes  = (offs.numel() + ooffs.numel() + dims.numel()) * 4.0
    total_bytes = bytes_read + bytes_write + meta_bytes
    gbps  = total_bytes / (ms * 1e-3) / 1e9
    giBps = total_bytes / (ms * 1e-3) / (1024**3)

    # 6) 正确性
    gv_gpu = gv.detach().cpu()
    ref    = cpu_ref(go.detach().cpu(), dims, feats, offs, ooffs, dtype=args.dtype)
    max_abs = (gv_gpu.to(torch.float32) - ref.to(torch.float32)).abs().max().item()
    ok = max_abs <= check_tol

    # 7) 打印
    name = torch.cuda.get_device_name()
    cc   = torch.cuda.get_device_capability()
    print("="*80)
    print(f"Device: {name} (CC {cc[0]}.{cc[1]})")
    print(f"[BW] B={args.B}  dims=[{args.dim_lo},{args.dim_hi}]  feats=[{args.feat_lo},{args.feat_hi}]  dtype(gv)={args.dtype}")
    print(f"sizes: go={total_go} elems, values={total_vals} elems")
    print(f"Avg kernel time: {ms:.3f} ms  |  total IO ~ {total_bytes/1e9:.3f} GB")
    print(f"Est. Bandwidth: {gbps:.2f} GB/s  ({giBps:.2f} GiB/s)")
    print(f"Max abs diff: {max_abs:.4e} -> {'PASSED' if ok else 'FAILED'}")
    print("="*80)
    if not ok:
        # 打印前几个 mismatch
        diff = (gv_gpu.to(torch.float32) - ref.to(torch.float32)).abs()
        bad  = (diff > check_tol).nonzero(as_tuple=False).flatten()[:10].tolist()
        for i in bad:
            print(f"mismatch idx={i} gpu={float(gv_gpu.view(-1)[i])} ref={float(ref.view(-1)[i])} err={float(diff.view(-1)[i])}")
        raise SystemExit(1)

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    main()

