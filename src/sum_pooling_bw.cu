// // ops/row2rows_bwd/kernel.cu
// #include <torch/extension.h>
// #include <ATen/cuda/CUDAContext.h>
// #include <torch/extension.h>
// #include <cuda_fp16.h>
// #include <cuda_bf16.h>
// using at::Tensor;

// template<typename T> __device__ __forceinline__ T f2T(float x);
// template<> __device__ __forceinline__ float         f2T<float>(float x){ return x; }
// template<> __device__ __forceinline__ __half        f2T<__half>(float x){ return __float2half_rn(x); }
// template<> __device__ __forceinline__ __nv_bfloat16 f2T<__nv_bfloat16>(float x){ return __float2bfloat16(x); }

// // template<typename T, int WARP=32>
// // __global__ void bwd_row_to_rows_warp_fast(
// //     const float* __restrict__ go,  T* __restrict__ gv,
// //     const int*  __restrict__ offs, const int* __restrict__ dims,
// //     const int*  __restrict__ ooffs,int B)
// // {
// //     const int lane            = threadIdx.x & (WARP - 1);
// //     const int warps_per_block = blockDim.x / WARP;
// //     const int warp_in_block   = threadIdx.x / WARP;
// //     const int warp_global     = blockIdx.x * warps_per_block + warp_in_block;
// //     const int total_warps     = gridDim.x * warps_per_block;

// //     const int active_warps = min(total_warps, B);
// //     if (warp_global >= active_warps) return;

// //     for (int s = warp_global; s < B; s += active_warps) {
// //         const int dim  = dims[s];
// //         const int sb   = offs[s];
// //         const int se   = offs[s+1];
// //         const int F    = (se - sb) / dim;
// //         const int gob  = ooffs[s];

// //         for (int i = lane; i < dim; i += WARP) {
// //             const T v = f2T<T>(go[gob + i]);
// //             #pragma unroll
// //             for (int f = 0; f < F; ++f) {
// //                 gv[sb + f*dim + i] = v;
// //             }
// //         }
// //     }
// // }

// template<typename T, int WARP=32, int VEC=4>
// __global__ void bwd_row_to_rows_warp_ilp_cols(
//     const float* __restrict__ go,  T* __restrict__ gv,
//     const int*  __restrict__ offs, const int* __restrict__ dims,
//     const int*  __restrict__ ooffs, int B)
// {
//     const int lane            = threadIdx.x & (WARP - 1);
//     const int warps_per_block = blockDim.x / WARP;
//     const int warp_in_block   = threadIdx.x / WARP;
//     const int warp_global     = blockIdx.x * warps_per_block + warp_in_block;
//     const int total_warps     = gridDim.x * warps_per_block;

//     for (int s = warp_global; s < B; s += total_warps) {
//         const int sb  = offs[s];
//         const int se  = offs[s+1];
//         const int dim = dims[s];
//         const int F   = (se - sb) / dim;
//         const int gob = ooffs[s];

//         // 防御：空样本直接跳过（极端数据下）
//         if (dim <= 0 || F <= 0) continue;

//         // 列块：每线程一次处理 VEC 相邻列，形成连续写
//         for (int base = 0; base < dim; base += WARP * VEC) {
//             const int i0 = base + lane * VEC;

//             float reg[VEC];
//             #pragma unroll
//             for (int u = 0; u < VEC; ++u) {
//                 const int idx = i0 + u;
//                 reg[u] = (idx < dim) ? go[gob + idx] : 0.0f;
//             }

//             T val[VEC];
//             #pragma unroll
//             for (int u = 0; u < VEC; ++u) {
//                 val[u] = f2T<T>(reg[u]);
//             }

//             #pragma unroll 4
//             for (int f = 0; f < F; ++f) {
//                 const int row_base = sb + f * dim + i0;
//                 #pragma unroll
//                 for (int u = 0; u < VEC; ++u) {
//                     const int idx = i0 + u;
//                     if (idx < dim) {
//                         gv[row_base + u] = val[u];
//                     }
//                 }
//             }
//         }
//     }
// }


// // template<typename T>
// // static void launch_impl(const at::Tensor& go, at::Tensor& gv,
// //                         const at::Tensor& offs, const at::Tensor& dims, const at::Tensor& ooffs)
// // {
// //     const int B = dims.size(0);
// //     constexpr int WARP=32;
// //     const int warps_per_block = 8;              // 256 threads
// //     dim3 block(warps_per_block * WARP);

// //     int sm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
// //     auto cd=[](int a,int b){return (a+b-1)/b;};
// //     int target_warps = std::min(B, sm * 16);    // 8~16 warp/SM
// //     int blocks = std::max(1, cd(target_warps, warps_per_block));
// //     dim3 grid(blocks);

// //     auto stream = at::cuda::getCurrentCUDAStream();
// //     bwd_row_to_rows_warp_fast<T,WARP><<<grid, block, 0, stream>>>(
// //         go.data_ptr<float>(),
// //         reinterpret_cast<T*>(gv.data_ptr()),
// //         offs.data_ptr<int>(),
// //         dims.data_ptr<int>(),
// //         ooffs.data_ptr<int>(),
// //         B);
// //     TORCH_CHECK(cudaGetLastError()==cudaSuccess, "kernel launch failed");
// // }

// // void sum_pooling_bw_launcher(const at::Tensor& go, at::Tensor& gv,
// //                            const at::Tensor& offs, const at::Tensor& dims, const at::Tensor& ooffs)
// // {
// //     TORCH_CHECK(go.is_cuda()&&gv.is_cuda()&&offs.is_cuda()&&dims.is_cuda()&&ooffs.is_cuda(), "CUDA tensors required");
// //     TORCH_CHECK(go.scalar_type()==at::kFloat, "grad_output must be float32");
// //     TORCH_CHECK(offs.scalar_type()==at::kInt && dims.scalar_type()==at::kInt && ooffs.scalar_type()==at::kInt);

// //     switch (gv.scalar_type()) {
// //       case at::kFloat:    launch_impl<float>(go, gv, offs, dims, ooffs); break;
// //       case at::kHalf:     launch_impl<__half>(go, gv, offs, dims, ooffs); break;
// //       case at::kBFloat16: launch_impl<__nv_bfloat16>(go, gv, offs, dims, ooffs); break;
// //       default: TORCH_CHECK(false, "grad_values must be f32/f16/bf16");
// //     }
// // }

// template<typename T>
// static void launch_impl(const at::Tensor& go, at::Tensor& gv,
//                         const at::Tensor& offs, const at::Tensor& dims, const at::Tensor& ooffs)
// {
//     const int B = dims.size(0);
//     constexpr int WARP = 32;
//     const int warps_per_block = 16;          // 512 threads/block
//     dim3 block(warps_per_block * WARP);

//     int sm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
//     auto cd = [](int a,int b){return (a+b-1)/b;};
//     int target_warps = std::min(B, sm * 32); // 32 warp/SM 起步（可根据卡调 32~64）
//     int blocks = std::max(1, cd(target_warps, warps_per_block));
//     dim3 grid(blocks);

//     auto stream = at::cuda::getCurrentCUDAStream();
//     // VEC=4 的常用版本（dim 小尾巴会 if 判掉）
//     bwd_row_to_rows_warp_ilp_cols<T, WARP, 4><<<grid, block, 0, stream>>>(
//         go.data_ptr<float>(),
//         reinterpret_cast<T*>(gv.data_ptr()),
//         offs.data_ptr<int>(),
//         dims.data_ptr<int>(),
//         ooffs.data_ptr<int>(),
//         B
//     );
//     C10_CUDA_KERNEL_LAUNCH_CHECK();
// }

// void sum_pooling_bw_launcher(const at::Tensor& go, at::Tensor& gv,
//                            const at::Tensor& offs, const at::Tensor& dims, const at::Tensor& ooffs)
// {
//     TORCH_CHECK(go.is_cuda()&&gv.is_cuda()&&offs.is_cuda()&&dims.is_cuda()&&ooffs.is_cuda(), "CUDA tensors required");
//     TORCH_CHECK(go.scalar_type()==at::kFloat, "grad_output must be float32");
//     TORCH_CHECK(offs.scalar_type()==at::kInt && dims.scalar_type()==at::kInt && ooffs.scalar_type()==at::kInt);

//     switch (gv.scalar_type()) {
//       case at::kFloat:    launch_impl<float>(go, gv, offs, dims, ooffs); break;
//       case at::kHalf:     launch_impl<__half>(go, gv, offs, dims, ooffs); break;
//       case at::kBFloat16: launch_impl<__nv_bfloat16>(go, gv, offs, dims, ooffs); break;
//       default: TORCH_CHECK(false, "grad_values must be f32/f16/bf16");
//     }
// }

// ops/row2rows_bwd/kernel.cu
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
using at::Tensor;

// ---------------- 类型转换工具 ----------------
template<typename T> __device__ __forceinline__ T f2T(float x);
template<> __device__ __forceinline__ float         f2T<float>(float x){ return x; }
template<> __device__ __forceinline__ __half        f2T<__half>(float x){ return __float2half_rn(x); }
template<> __device__ __forceinline__ __nv_bfloat16 f2T<__nv_bfloat16>(float x){ return __float2bfloat16(x); }

template<typename S> __device__ __forceinline__ float T2f(S x);
template<> __device__ __forceinline__ float T2f<float>(float x){ return x; }
template<> __device__ __forceinline__ float T2f<__half>(__half x){ return __half2float(x); }
template<> __device__ __forceinline__ float T2f<__nv_bfloat16>(__nv_bfloat16 x){
    return __bfloat162float(x);
}
// ---------------- 主核函数（输入/输出类型可独立） ----------------
template<typename S, typename T, int WARP=32, int VEC=4>
__global__ void bwd_row_to_rows_warp_ilp_cols_typed(
    const S* __restrict__ go,  T* __restrict__ gv,
    const int*  __restrict__ offs, const int* __restrict__ dims,
    const int*  __restrict__ ooffs, int B)
{
    const int lane            = threadIdx.x & (WARP - 1);
    const int warps_per_block = blockDim.x / WARP;
    const int warp_in_block   = threadIdx.x / WARP;
    const int warp_global     = blockIdx.x * warps_per_block + warp_in_block;
    const int total_warps     = gridDim.x * warps_per_block;

    for (int s = warp_global; s < B; s += total_warps) {
        const int sb  = offs[s];
        const int se  = offs[s+1];
        const int dim = dims[s];
        const int F   = (se - sb) / dim;
        const int gob = ooffs[s];
        if (dim <= 0 || F <= 0) continue;

        for (int base = 0; base < dim; base += WARP * VEC) {
            const int i0 = base + lane * VEC;

            // 先读入为 float 做统一转换（避免不同 S 带来的分支）
            float reg[VEC];
            #pragma unroll
            for (int u = 0; u < VEC; ++u) {
                const int idx = i0 + u;
                reg[u] = (idx < dim) ? T2f<S>(go[gob + idx]) : 0.0f;
            }

            // 再转为目标类型 T，后面复用
            T val[VEC];
            #pragma unroll
            for (int u = 0; u < VEC; ++u) {
                val[u] = f2T<T>(reg[u]);
            }

            #pragma unroll 4
            for (int f = 0; f < F; ++f) {
                const int row_base = sb + f * dim + i0;
                #pragma unroll
                for (int u = 0; u < VEC; ++u) {
                    const int idx = i0 + u;
                    if (idx < dim) {
                        gv[row_base + u] = val[u];
                    }
                }
            }
        }
    }
}

// ---------------- 通用 launch（复用网格/块设置） ----------------
template<typename S, typename T>
static void launch_pair(const at::Tensor& go, at::Tensor& gv,
                        const at::Tensor& offs, const at::Tensor& dims, const at::Tensor& ooffs)
{
    constexpr int WARP = 32;
    const int B = dims.size(0);
    const int warps_per_block = 16;          // 512 threads/block
    dim3 block(warps_per_block * WARP);

    int sm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    auto cd = [](int a,int b){return (a+b-1)/b;};
    int target_warps = std::min(B, sm * 32); // 32 warp/SM，按卡可调 32~64
    int blocks = std::max(1, cd(target_warps, warps_per_block));
    dim3 grid(blocks);

    auto stream = at::cuda::getCurrentCUDAStream();
    bwd_row_to_rows_warp_ilp_cols_typed<S,T,WARP,4><<<grid, block, 0, stream>>>(
        reinterpret_cast<const S*>(go.data_ptr()),
        reinterpret_cast<T*>(gv.data_ptr()),
        offs.data_ptr<int>(),
        dims.data_ptr<int>(),
        ooffs.data_ptr<int>(),
        B
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------- 顶层 launcher ----------------
void sum_pooling_bw_launcher(const at::Tensor& go, at::Tensor& gv,
                             const at::Tensor& offs, const at::Tensor& dims, const at::Tensor& ooffs)
{
    TORCH_CHECK(go.is_cuda()&&gv.is_cuda()&&offs.is_cuda()&&dims.is_cuda()&&ooffs.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(offs.scalar_type()==at::kInt && dims.scalar_type()==at::kInt && ooffs.scalar_type()==at::kInt);

    // 现在支持 FP32 和 BF16 的 grad_output
    TORCH_CHECK(
        go.scalar_type()==at::kFloat || go.scalar_type()==at::kBFloat16,
        "grad_output must be float32 or bfloat16"
    );

    // 根据 go dtype × gv dtype 分派
    if (go.scalar_type() == at::kFloat) {
        switch (gv.scalar_type()) {
          case at::kFloat:    launch_pair<float,          float>(go, gv, offs, dims, ooffs); break;
          case at::kHalf:     launch_pair<float,          __half>(go, gv, offs, dims, ooffs); break;
          case at::kBFloat16: launch_pair<float, __nv_bfloat16>(go, gv, offs, dims, ooffs); break;
          default: TORCH_CHECK(false, "grad_values must be f32/f16/bf16");
        }
    } else { // go == BF16
        switch (gv.scalar_type()) {
          case at::kBFloat16: launch_pair<__nv_bfloat16, __nv_bfloat16>(go, gv, offs, dims, ooffs); break; // ★ BF16 -> BF16
          case at::kFloat:    launch_pair<__nv_bfloat16, float>(go, gv, offs, dims, ooffs); break;          // 可选：BF16 -> F32
          case at::kHalf:     launch_pair<__nv_bfloat16, __half>(go, gv, offs, dims, ooffs); break;         // 可选：BF16 -> F16
          default: TORCH_CHECK(false, "grad_values must be f32/f16/bf16");
        }
    }
}
