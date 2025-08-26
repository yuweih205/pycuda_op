#include <torch/extension.h>
#include <pybind11/pybind11.h>
using at::Tensor;

void sum_pooling_bw_launcher(const Tensor& go, Tensor& gv,
                          const Tensor& offs, const Tensor& dims, const Tensor& ooffs);


void sum_pooling_fw_launcher(const Tensor& values_bf16, const Tensor& offsets_i32,
                             const Tensor& dims_i32, Tensor& output_f32,
                             const Tensor& output_offsets_i32);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//#ifdef SUMPOOL
    m.def("sum_pooling_fw", &sum_pooling_fw_launcher, "SumPooling forward (CUDA)");
    m.def("sum_pooling_bw", &sum_pooling_bw_launcher, "SumPooling backward (CUDA)");
//#endif  
  //增加新函数
}
