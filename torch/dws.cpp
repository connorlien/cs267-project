#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <string.h>
#include <omp.h>
#include <torch/extension.h>
#include <iostream>

#define min(a, b) (((a) < (b)) ? (a) : (b))
#define row_major(i, j, num_rows) ((i) * (num_rows) + (j))
#define pad(B) (((B) % 8 != 0 ? (B) + 8 - ((B) % 8) : (B)))
#define AVX_BITS 512

int BATCH_BLOCK_PW;
int FILTER_BLOCK_PW;
int WIDTH_BLOCK_PW;
int HEIGHT_BLOCK_PW;
int CHANNEL_BLOCK_PW;

int BATCH_BLOCK_DW;
int CHANNEL_BLOCK_DW;
int FILTER_DW;
int HEIGHT_BLOCK_DW;
int WIDTH_BLOCK_DW;
int HEIGHT_FILTER_BLOCK_DW;
int WIDTH_FILTER_BLOCK_DW;

int VEC_SIZE;

static void dw_conv(float *X, float *F_DW, float *O, int B, int H_in, int W_in, int C_in, int H_f, int W_f, int N_dw, int H_out, int W_out, int stride_h, int stride_w)
{
    int mat_size = W_in * H_in;
    int f_size = W_f * H_f;
    int img_size = mat_size * C_in;

    int temp_out_img_size = W_out * H_out;
    int temp_out_size = temp_out_img_size * N_dw * C_in;

    #pragma omp parallel for collapse(4)
    for (int b = 0; b < B; b += 1)
    {
        for (int c = 0; c < C_in; c += 1)
        {
            for (int h = 0; h < H_out; h += 1)
            {
                for (int w = 0; w < W_out; w += 1)
                {
                    for (int h_f = 0; h_f < H_f; h_f += 1)
                    {
                        for (int w_f = 0; w_f < W_f; w_f += 1)
                        {
                            float *curr_img = X + b * img_size;
                            float *curr_out = O + b * temp_out_size;
                            float *curr_channel = curr_img + mat_size * c;
                            float *f_curr = F_DW + f_size * (c * N_dw) + row_major(h_f, w_f, W_f);
                            int h_curr = h_f + stride_h * h;
                            int w_curr = w_f + stride_w * w;
                            float *curr_inp = curr_channel + row_major(h_curr, w_curr, W_in);
                            float *curr_out_xy = curr_out + temp_out_img_size * (c * N_dw) + row_major(h, w, W_out);
                            *curr_out_xy = *curr_out_xy + *f_curr * *curr_inp;
                        }
                    }
                }
            }
        }
    }
}

static void pw_conv(float *X, float *F_1D, float *O, int B, int H_in, int W_in, int C_in, int C_out)
{
    int mat_size = W_in * H_in;
    int img_size = mat_size * C_in;
    int out_size = mat_size * C_out;

    #pragma omp parallel for collapse(4)
    for (int b = 0; b < B; b += 1)
    {
        for (int f = 0; f < C_out; f += 1)
        {
            for (int h = 0; h < H_in; h += 1)
            {
                for (int w = 0; w < W_in; w += 1)
                {
                    for (int c = 0; c < C_in; c += 1)
                    {
                        float *curr_img = X + b * img_size;
                        float *curr_out = O + b * out_size;
                        float *f_curr = F_1D + f * C_in + c;
                        float *o_curr = curr_out + mat_size * f + row_major(h, w, W_in);
                        float *inp_curr = curr_img + mat_size * c + row_major(h, w, W_in);
                        *o_curr += (*f_curr) * (*inp_curr);
                    }
                }
            }
        }
    }
}

void print_tensor(float *X, int size, const char *name)
{
    fprintf(stderr, "%s\n", name);
    for (int i = 0; i < size; i += 1)
    {
        fprintf(stderr, "%f ", X[i]);
    }
    fprintf(stderr, "\n");
}

void init_conv(int bbpw, int fbpw, int wbpw, int hbpw, int cbpw, int bbdw, int cbdw, int fdw, int hbdw, int wbdw, int hfdw, int wfbdw)
{
    BATCH_BLOCK_PW = bbpw;
    FILTER_BLOCK_PW = fbpw;
    WIDTH_BLOCK_PW = wbpw;
    HEIGHT_BLOCK_PW = hbpw;
    CHANNEL_BLOCK_PW = cbpw;

    BATCH_BLOCK_DW = bbdw;
    CHANNEL_BLOCK_DW = cbdw;
    FILTER_DW = fdw;
    HEIGHT_BLOCK_DW = hbdw;
    WIDTH_BLOCK_DW = wbdw;
    HEIGHT_FILTER_BLOCK_DW = hfdw;
    WIDTH_FILTER_BLOCK_DW = wfbdw;

    VEC_SIZE = AVX_BITS / sizeof(float);
}

void dws_conv(float *X, float *F_DW, float *F_1D, float *O, int B, int H_in, int W_in, int C_in, int H_f, int W_f, int N_dw, int H_out, int W_out, int C_out, int stride_h, int stride_w, float *depthwise_output)
{
    dw_conv(X, F_DW, depthwise_output, B, H_in, W_in, C_in, H_f, W_f, N_dw, H_out, W_out, stride_h, stride_w);
    pw_conv(depthwise_output, F_1D, O, B, H_out, W_out, C_in * N_dw, C_out);
}

torch::Tensor dws_conv_torch(torch::Tensor input, torch::Tensor kernel_dw, torch::Tensor kernel_pw, int stride)
{
    float *X = input.data_ptr<float>();
    float *F_DW = kernel_dw.data_ptr<float>();
    float *F_1D = kernel_pw.data_ptr<float>();

    int B = input.sizes()[0];
    int H_in = input.sizes()[2];
    int W_in = input.sizes()[3];
    int C_in = input.sizes()[1];
    int H_f = kernel_dw.sizes()[2];
    int W_f = kernel_dw.sizes()[3];
    int N_dw = 1;
    int H_out = floor((H_in - H_f) / stride + 1);
    int W_out = floor((W_in - W_f) / stride + 1);
    int C_out = C_in;

    float* O = (float *) calloc(B * C_out * W_out * H_out, sizeof(float));
    float* depthwise_output = (float *) calloc(B * W_out * H_out * C_in * N_dw, sizeof(float));

    dws_conv(X, F_DW, F_1D, O, B, H_in, W_in, C_in, H_f, W_f, N_dw, H_out, W_out, C_out, stride, stride, depthwise_output);
    
    torch::Tensor result = torch::from_blob(O, {B, C_out, H_out, W_out}, torch::dtype(torch::kFloat32)).clone();
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dws_conv", &dws_conv_torch, "DWS Convolution");
  m.def("init_conv", &init_conv, "Initialize DWS Convolution");
}