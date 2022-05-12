#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

extern void init_conv(int bbpw, int fbpw, int wbpw, int hbpw, int cbpw, int bbdw, int cbdw, int fdw, int hbdw, int wbdw, int hfdw, int wfbdw);
extern torch::Tensor dws_conv_torch(torch::Tensor input, torch::Tensor kernel_dw, torch::Tensor kernel_pw, int stride);

void init_conv_handler(int bbpw, int fbpw, int wbpw, int hbpw, int cbpw, int bbdw, int cbdw, int fdw, int hbdw, int wbdw, int hfdw, int wfbdw) {
    init_conv(bbpw, fbpw, wbpw, hbpw, cbpw, bbdw, cbdw, fdw, hbdw, wbdw, hfdw, wfbdw);
}

torch::Tensor dws_conv_handler(torch::Tensor input, torch::Tensor kernel_dw, torch::Tensor kernel_pw, int stride) {
    CHECK_INPUT(input);
    CHECK_INPUT(kernel_dw);
    CHECK_INPUT(kernel_pw);
    return dws_conv_torch(input, kernel_dw, kernel_pw, stride);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dws_conv", &dws_conv_handler, "DWS Convolution (CUDA)");
  m.def("init_conv", &init_conv_handler, "Initialize DWS Convolution (CUDA)");
}