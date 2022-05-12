#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <torch/extension.h>

#define NUM_THREADS 32
#define min(a, b) (((a) < (b)) ? (a) : (b))
#define row_major(i, j, num_rows) ((i) * (num_rows) + (j))

__global__ void dw_conv_gpu(float *X, float *F_DW, float *O, int B, int H_in, int W_in, int C_in, int H_f, int W_f, int N_dw, int H_out, int W_out, int stride_h, int stride_w) {
	
	// Compute batch and channel for this thread
    int b = threadIdx.x + blockIdx.x * blockDim.x;
    int w = threadIdx.y + blockIdx.y * blockDim.y;
    int h = threadIdx.z + blockIdx.z * blockDim.z;

    if (b >= B || h >= H_out || w >= W_out) {
        return;
    }

    // Pre-computations
    int mat_size = W_in * H_in;
    int f_size = W_f * H_f;
    int img_size = mat_size * C_in;

    int temp_out_img_size = W_out * H_out;
    int temp_out_size = temp_out_img_size * N_dw * C_in;

    // PTRS TO IMG IN BATCH
    float *curr_img = X + b * img_size;
    float *curr_out = O + b * temp_out_size;

    // Filters are 2D
    for (int c = 0; c < C_in; c += 1)
    {
        float *curr_channel = curr_img + mat_size * c;
        for (int w_f = 0; w_f < W_f; w_f += 1)
        {
            for (int h_f = 0; h_f < H_f; h_f += 1)
            {
                // PTR TO CURRENT POSITION IN FILTER
                float *f_curr = F_DW + f_size * c + row_major(h_f, w_f, W_f);

                // PTR TO INPUT POSITION
                int h_curr = h_f + stride_h * h;
                int w_curr = w_f + stride_w * w;
                float *curr_inp = curr_channel + row_major(h_curr, w_curr, W_in);

                // PTR TO INPUT POSITION
                float *curr_out_xy = curr_out + temp_out_img_size * c + row_major(h, w, W_out);

                // CONVOLVE
                *curr_out_xy = *curr_out_xy + *f_curr * *curr_inp;
            }
        }
    }
}

__global__ void pw_conv_gpu(float *X, float *F_1D, float *O, int B, int H_in, int W_in, int C_in, int C_out)
{
    // Compute batch and channel for this thread
    int b = threadIdx.x + blockIdx.x * blockDim.x;
    int w = threadIdx.y + blockIdx.y * blockDim.y;
    int h = threadIdx.z + blockIdx.z * blockDim.z;

    if (b >= B || w >= W_in || h >= H_in) {
        return;
    }

    int mat_size = W_in * H_in;
    int img_size = mat_size * C_in;
    int out_size = mat_size * C_out;

    float *curr_img = X + b * img_size;
    float *curr_out = O + b * out_size;

    for (int f = 0; f < C_out; f += 1)
    {
        float *o_curr = curr_out + mat_size * f + row_major(h, w, W_in);
        for (int c = 0; c < C_in; c += 1)
        {
            float *f_curr = F_1D + f * C_in + c;
            float *inp_curr = curr_img + mat_size * c + row_major(h, w, W_in);
            *o_curr += (*f_curr) * (*inp_curr);
        }
    }
}

void init_conv(int bbpw, int fbpw, int wbpw, int hbpw, int cbpw, int bbdw, int cbdw, int fdw, int hbdw, int wbdw, int hfdw, int wfbdw) {
}

void dws_conv(float *X, float *F_DW, float *F_1D, float *O, int B, int H_in, int W_in, int C_in, int H_f, int W_f, int N_dw, int H_out, int W_out, int C_out, int stride_h, int stride_w, float* depthwise_output)
{
    dim3 dw_gridDim(B, W_out, H_out);
    dim3 dw_blockDim(NUM_THREADS, NUM_THREADS);
    dw_conv_gpu<<<dw_gridDim, dw_blockDim>>>(X, F_DW, depthwise_output, B, H_in, W_in, C_in, H_f, W_f, N_dw, H_out, W_out, stride_h, stride_w);
    
    dim3 pw_gridDim(B, W_out, H_out);
    dim3 pw_blockDim(NUM_THREADS, NUM_THREADS);
    pw_conv_gpu<<<pw_gridDim, pw_blockDim>>>(depthwise_output, F_1D, O, B, H_out, W_out, C_in * N_dw, C_out);
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
    
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0);

    torch::Tensor O = torch::zeros({B, C_out, W_out, H_out}, options);
    float* O_gpu = O.data_ptr<float>();

    torch::Tensor depthwise_output = torch::zeros({B, C_in, W_out, H_out}, options);
    float* depthwise_output_gpu = depthwise_output.data_ptr<float>();

    dws_conv(X, F_DW, F_1D, O_gpu, B, H_in, W_in, C_in, H_f, W_f, N_dw, H_out, W_out, C_out, stride, stride, depthwise_output_gpu);

    return O;
}
