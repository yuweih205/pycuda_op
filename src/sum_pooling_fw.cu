// // sum_bf16_warp.cu  —— 前向：按样本逐列求和 (BF16 -> FP32)，pybind11 导出
// #include <torch/extension.h>
// #include <ATen/cuda/CUDAContext.h>
// #include <cuda_runtime.h>
// #include <cuda_bf16.h>
// using at::Tensor;

// // #define TCHK(c, msg) TORCH_CHECK((c), msg)
// // template<int ILP = 8, int WARP=32>
// // __global__ void sum_bf16_dim_per_sample_warp(
// //     const __nv_bfloat16* __restrict__ values, 
// //     const int* __restrict__ offsets,
// //     const int* __restrict__ dims,
// //     float* __restrict__ output,
// //     const int* __restrict__ output_offsets,
// //     int batchsize,
// //     int active_warps)
// // {
// //     constexpr int kWarp = WARP;
// //     const int warps_per_block = blockDim.x / kWarp;
// //     const int warp_in_block   = threadIdx.x / kWarp;
// //     const int warp_global     = blockIdx.x * warps_per_block + warp_in_block;
// //     const int lane            = threadIdx.x & (kWarp - 1);

// //     if (warp_global >= active_warps) return;
// //     for (int s = warp_global; s < batchsize; s += active_warps) {
// //         const int dim        = dims[s];
// //         const int out_base   = output_offsets[s];
// //         const int span_begin = offsets[s];
// //         const int span_end   = offsets[s + 1];
// //         const int width      = span_end - span_begin;
// //         const int features   = width / dim;

// //         for (int t = lane; t < dim; t += kWarp) {
// //             float accum = 0.f;

// //             int f = 0;
// // #pragma unroll
// //             for (; f + (ILP - 1) < features; f += ILP) {
// // #pragma unroll
// //                 for (int u = 0; u < ILP; ++u) {
// //                     const int row = span_begin + (f + u) * dim;
// //                     const __nv_bfloat16 v = values[row + t];
// //                     accum += __bfloat162float(v);
// //                 }
// //             }
// //             for (; f < features; ++f) {
// //                 const int row = span_begin + f * dim;
// //                 accum += __bfloat162float(values[row + t]);
// //             }
// //             output[out_base + t] = accum;
// //         }
// //     }
// // }

// // void sum_pooling_fw_launcher(const Tensor& values_bf16,
// //                              const Tensor& offsets_i32,
// //                              const Tensor& dims_i32,
// //                              Tensor&       output,
// //                              const Tensor& output_offsets_i32)
// // {
// //     TCHK(values_bf16.is_cuda(), "values must be CUDA");
// //     TCHK(offsets_i32.is_cuda() && dims_i32.is_cuda() && output_offsets_i32.is_cuda(), "offsets/dims/output_offsets must be CUDA");
// //     TCHK(output.is_cuda(), "output must be CUDA");

// //     TCHK(values_bf16.scalar_type() == at::kBFloat16, "values must be bfloat16");
// //     TCHK(output.scalar_type()  == at::kFloat,    "output must be float32");
// //     TCHK(offsets_i32.scalar_type() == at::kInt,      "offsets must be int32");
// //     TCHK(dims_i32.scalar_type()    == at::kInt,      "dims must be int32");
// //     TCHK(output_offsets_i32.scalar_type() == at::kInt, "output_offsets must be int32");

// //     const int B = dims_i32.size(0);
// //     TCHK(offsets_i32.size(0) == B + 1, "offsets length must be B+1");
// //     TCHK(output_offsets_i32.size(0) == B + 1, "output_offsets length must be B+1");

// //     const __nv_bfloat16* values_ptr = reinterpret_cast<const __nv_bfloat16*>(values_bf16.data_ptr<at::BFloat16>());
// //     const int* offsets_ptr      = offsets_i32.data_ptr<int>();
// //     const int* dims_ptr         = dims_i32.data_ptr<int>();
// //     const int* ooffs_ptr        = output_offsets_i32.data_ptr<int>();
// //     float* output_ptr           = output.data_ptr<float>();

// //     cudaDeviceProp prop{};
// //     cudaGetDeviceProperties(&prop, at::cuda::current_device());

// //     constexpr int WARP = 32;
// //     const int warps_per_block = 16;                 // 8 warp / block = 256 threads
// //     const dim3 block(warps_per_block * WARP);

// //     const int sm_count = prop.multiProcessorCount;
// //     const int target_warps = std::max(1, std::min(B, sm_count * 64));
// //     auto ceil_div = [](int a, int b){ return (a + b - 1) / b; };

// //     int blocks = std::max(1, ceil_div(target_warps, warps_per_block));
// //     const int launched_warps = blocks * warps_per_block;
// //     const int active_warps   = std::min(target_warps, launched_warps);

// //     const dim3 grid(blocks);
// //     const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

// //     sum_bf16_dim_per_sample_warp<8, WARP><<<grid, block, 0, stream>>>(
// //         values_ptr, offsets_ptr, dims_ptr, output_ptr, ooffs_ptr, B, active_warps);

// //     TORCH_CHECK(cudaPeekAtLastError() == cudaSuccess, "kernel launch failed");
// // }


// #define TCHK(c, msg) TORCH_CHECK((c), msg)
// template<int ILP=8, int WARP=32>
// __global__ void sum_bf16_dim_per_sample_warp_v2(
//     const __nv_bfloat16* __restrict__ values,
//     const int* __restrict__ offsets,
//     const int* __restrict__ dims,
//     float* __restrict__ output,
//     const int* __restrict__ output_offsets,
//     int batchsize)
// {
//     constexpr int kWarp = WARP;
//     const int warps_per_block = blockDim.x / kWarp;
//     const int warp_in_block   = threadIdx.x / kWarp;
//     const int warp_global     = blockIdx.x * warps_per_block + warp_in_block;
//     const int lane            = threadIdx.x & (kWarp - 1);

//     if (warp_global >= batchsize) return;
//     const int s = warp_global;

//     const int dim        = dims[s];
//     const int out_base   = output_offsets[s];
//     const int span_begin = offsets[s];
//     const int span_end   = offsets[s + 1];
//     const int width      = span_end - span_begin;
//     const int features   = width / dim;

//     for (int t = lane; t < dim; t += kWarp) {
//         float accum = 0.f;
//         int f = 0;
//         #pragma unroll
//         for (; f + (ILP - 1) < features; f += ILP) {
//             #pragma unroll
//             for (int u = 0; u < ILP; ++u) {
//                 const int row = span_begin + (f + u) * dim;
//                 accum += __bfloat162float(values[row + t]);
//             }
//         }
//         for (; f < features; ++f) {
//             const int row = span_begin + f * dim;
//             accum += __bfloat162float(values[row + t]);
//         }
//         output[out_base + t] = accum;
//     }
// }




// void sum_pooling_fw_launcher(const Tensor& values_bf16,
//                              const Tensor& offsets_i32,
//                              const Tensor& dims_i32,
//                              Tensor&       output,
//                              const Tensor& output_offsets_i32)
// {
//     TCHK(values_bf16.is_cuda() && values_bf16.is_contiguous() && values_bf16.scalar_type()==at::kBFloat16, "values must be CUDA contiguous bf16");
//     TCHK(output.is_cuda()  && output.is_contiguous()  && output.scalar_type()==at::kFloat,    "output must be CUDA contiguous f32");
//     TCHK(offsets_i32.is_cuda() && offsets_i32.is_contiguous() && offsets_i32.scalar_type()==at::kInt,      "offsets must be CUDA contiguous int32");
//     TCHK(dims_i32.is_cuda()    && dims_i32.is_contiguous()    && dims_i32.scalar_type()==at::kInt,        "dims must be CUDA contiguous int32");
//     TCHK(output_offsets_i32.is_cuda() && output_offsets_i32.is_contiguous() && output_offsets_i32.scalar_type()==at::kInt, "output_offsets must be CUDA contiguous int32");

//     const int dev = values_bf16.get_device();
//     TCHK(offsets_i32.get_device()==dev && dims_i32.get_device()==dev && output_offsets_i32.get_device()==dev && output.get_device()==dev,
//          "all tensors must be on the same CUDA device");

//     const int B = dims_i32.size(0);
//     TCHK(offsets_i32.size(0)==B+1 && output_offsets_i32.size(0)==B+1, "offsets/out_offsets length must be B+1");
//     TCHK(output.numel()==output_offsets_i32[B].item<int>(), "output size must equal output_offsets[B]");

//     // 可选：数据一致性（开发期打开）
//     // for (int i=0;i<B;++i) TCHK(((offsets_i32[i+1]-offsets_i32[i]) % dims_i32[i])==0, "width must be multiple of dim");

//     const __nv_bfloat16* values_ptr = reinterpret_cast<const __nv_bfloat16*>(values_bf16.data_ptr<at::BFloat16>());
//     const int* offsets_ptr = offsets_i32.data_ptr<int>();
//     const int* dims_ptr    = dims_i32.data_ptr<int>();
//     const int* ooffs_ptr   = output_offsets_i32.data_ptr<int>();
//     float* output_ptr      = output.data_ptr<float>();

//     // 2) warp-per-sample 发射
//     constexpr int WARP = 32;
//     const int warps_per_block = 16;                 // 512 线程
//     const dim3 block(warps_per_block * WARP);
//     const dim3 grid((B + warps_per_block - 1) / warps_per_block);

//     const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
//     sum_bf16_dim_per_sample_warp_v2<8, WARP><<<grid, block, 0, stream>>>(
//         values_ptr, offsets_ptr, dims_ptr, output_ptr, ooffs_ptr, B);

//     C10_CUDA_KERNEL_LAUNCH_CHECK(); // 替代单纯 Peek
// }

// sum_unified_warp.cu  —— 前向：按样本逐列求和 (BF16/FP32 -> FP32)
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
using at::Tensor;

#define TCHK(c, msg) TORCH_CHECK((c), msg)

// ---- 类型统一：Tin -> float ----
__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }

template<typename Tout>
__device__ __forceinline__ Tout from_float(float x);

template<>
__device__ __forceinline__ float from_float<float>(float x) { return x; }

template<>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float x) {
    return __float2bfloat16(x);
}
// ---- warp-per-sample + ILP 展开 ----
template<typename Tin, typename Tout,int ILP=8, int WARP=32>
__global__ void sum_dim_per_sample_warp_unified(
    const Tin*  __restrict__ values,          // 输入：FP32 或 BF16
    const int*  __restrict__ offsets,         // [B+1]
    const int*  __restrict__ dims,            // [B]
    Tout*      __restrict__ output,          // 输出：FP32
    const int*  __restrict__ output_offsets,  // [B+1]
    int B)
{
    constexpr int kWarp = WARP;
    const int warps_per_block = blockDim.x / kWarp;
    const int warp_in_block   = threadIdx.x / kWarp;
    const int warp_global     = blockIdx.x * warps_per_block + warp_in_block;
    const int lane            = threadIdx.x & (kWarp - 1);

    if (warp_global >= B) return;
    const int s = warp_global;

    const int dim        = dims[s];
    const int out_base   = output_offsets[s];
    const int span_begin = offsets[s];
    const int span_end   = offsets[s + 1];
    const int width      = span_end - span_begin;
    const int features   = (dim > 0) ? (width / dim) : 0;

    if (dim <= 0 || features <= 0) return;

    for (int t = lane; t < dim; t += kWarp) {
        float accum = 0.f;

        int f = 0;
        #pragma unroll
        for (; f + (ILP - 1) < features; f += ILP) {
            #pragma unroll
            for (int u = 0; u < ILP; ++u) {
                const int row = span_begin + (f + u) * dim;
                accum += to_float(values[row + t]);
            }
        }
        for (; f < features; ++f) {
            const int row = span_begin + f * dim;
            accum += to_float(values[row + t]);
        }
        output[out_base + t] = from_float<Tout> (accum);
    }
}

// ---- Launcher：自动按输入 dtype 分派 ----
void sum_pooling_fw_launcher(const Tensor& values,
                             const Tensor& offsets_i32,
                             const Tensor& dims_i32,
                             Tensor&       output,
                             const Tensor& output_offsets_i32)
{
    // 基本校验
    TCHK(values.is_cuda() && values.is_contiguous(), "values must be CUDA contiguous");
    TCHK(output.is_cuda() && output.is_contiguous() && (output.scalar_type()==at::kFloat || output.scalar_type()==at::kBFloat16 ), "output must be CUDA contiguous float32 or bf16");
    TCHK(offsets_i32.is_cuda() && offsets_i32.is_contiguous() && offsets_i32.scalar_type()==at::kInt, "offsets must be CUDA contiguous int32");
    TCHK(dims_i32.is_cuda()    && dims_i32.is_contiguous()    && dims_i32.scalar_type()==at::kInt,   "dims must be CUDA contiguous int32");
    TCHK(output_offsets_i32.is_cuda() && output_offsets_i32.is_contiguous() && output_offsets_i32.scalar_type()==at::kInt, "output_offsets must be CUDA contiguous int32");

    const int dev = values.get_device();
    TCHK(offsets_i32.get_device()==dev && dims_i32.get_device()==dev &&
         output_offsets_i32.get_device()==dev && output.get_device()==dev,
         "all tensors must be on the same CUDA device");

    const int B = dims_i32.size(0);
    TCHK(offsets_i32.size(0)==B+1 && output_offsets_i32.size(0)==B+1, "offsets/out_offsets length must be B+1");
    TCHK(output.numel()==output_offsets_i32[B].item<int>(), "output size must equal output_offsets[B]");

    const int* offsets_ptr = offsets_i32.data_ptr<int>();
    const int* dims_ptr    = dims_i32.data_ptr<int>();
    const int* ooffs_ptr   = output_offsets_i32.data_ptr<int>();

    constexpr int WARP = 32;
    const int warps_per_block = 16;                    // 512 threads/block
    const dim3 block(warps_per_block * WARP);
    const dim3 grid((B + warps_per_block - 1) / warps_per_block);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (values.scalar_type()==at::kFloat && output.scalar_type()==at::kBFloat16) {
        const float* vptr = values.data_ptr<float>();
        __nv_bfloat16* optr = reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>());
        // FP32进 → FP32算 → BF16出
        sum_dim_per_sample_warp_unified<float, __nv_bfloat16, 8, WARP>
            <<<grid, block, 0, stream>>>(vptr, offsets_ptr, dims_ptr, optr, ooffs_ptr, B);

    } else if (values.scalar_type()==at::kBFloat16 && output.scalar_type()==at::kBFloat16) {
        const __nv_bfloat16* vptr = reinterpret_cast<const __nv_bfloat16*>(values.data_ptr<at::BFloat16>());
        __nv_bfloat16* optr = reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>());
        // **BF16进 → FP32算 → BF16出
        sum_dim_per_sample_warp_unified<__nv_bfloat16, __nv_bfloat16, 8, WARP>
            <<<grid, block, 0, stream>>>(vptr, offsets_ptr, dims_ptr, optr, ooffs_ptr, B);

    } else if (values.scalar_type()==at::kBFloat16 && output.scalar_type()==at::kFloat) {
        const __nv_bfloat16* vptr = reinterpret_cast<const __nv_bfloat16*>(values.data_ptr<at::BFloat16>());
        float* optr = output.data_ptr<float>();
        // BF16进 → FP32算 → FP32出
        sum_dim_per_sample_warp_unified<__nv_bfloat16, float, 8, WARP>
            <<<grid, block, 0, stream>>>(vptr, offsets_ptr, dims_ptr, optr, ooffs_ptr, B);

    } else if (values.scalar_type()==at::kFloat && output.scalar_type()==at::kFloat) {
        const float* vptr = values.data_ptr<float>();
        float* optr = output.data_ptr<float>();
        // FP32进 → FP32算 → FP32出
        sum_dim_per_sample_warp_unified<float, float, 8, WARP>
            <<<grid, block, 0, stream>>>(vptr, offsets_ptr, dims_ptr, optr, ooffs_ptr, B);

    } else {
        TORCH_CHECK(false, "values/output must be float32 or bfloat16");
    }  
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
