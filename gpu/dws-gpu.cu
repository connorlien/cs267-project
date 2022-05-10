#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "dws-gpu.h"

#define NUM_THREADS 16
#define min(a, b) (((a) < (b)) ? (a) : (b))
#define row_major(i, j, num_rows) ((i) * (num_rows) + (j))

__global__ void dw_conv_gpu(float *X, float *F_DW, float *O, int B, int H_in, int W_in, int C_in, int H_f, int W_f, int N_dw, int H_out, int W_out, int stride_h, int stride_w) {
	
	// Compute batch and channel for this thread
    int b = threadIdx.x + blockIdx.x * blockDim.x;
    int c = threadIdx.y + blockIdx.y * blockDim.y;

    if (b >= B || c >= C_in) {
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

    // Do 2D Convolution channelwise
    float *curr_channel = curr_img + mat_size * c;

    // Filters are 2D
    for (int f = 0; f < N_dw; f += 1)
    {
        for (int w = 0; w < W_out; w += 1)
        {
            for (int h = 0; h < H_out; h += 1)
            {
                // MICROKERNEL - tile if needed.
                for (int w_f = 0; w_f < W_f; w_f += 1)
                {
                    for (int h_f = 0; h_f < H_f; h_f += 1)
                    {
                        // PTR TO CURRENT POSITION IN FILTER
                        float *f_curr = F_DW + f_size * (c * N_dw + f) + row_major(h_f, w_f, W_f);

                        // PTR TO INPUT POSITION
                        int h_curr = h_f + stride_h * h;
                        int w_curr = w_f + stride_w * w;
                        float *curr_inp = curr_channel + row_major(h_curr, w_curr, W_in);

                        // PTR TO INPUT POSITION
                        float *curr_out_xy = curr_out + temp_out_img_size * (c * N_dw + f) + row_major(h, w, W_out);

                        // CONVOLVE
                        *curr_out_xy = *curr_out_xy + *f_curr * *curr_inp;
                    }
                }
            }
        }
    }
}

__global__ void pw_conv_gpu(float *X, float *F_1D, float *O, int B, int H_in, int W_in, int C_in, int C_out)
{
    // Compute batch and channel for this thread
    int b = threadIdx.x + blockIdx.x * blockDim.x;
    int f = threadIdx.y + blockIdx.y * blockDim.y;

    if (b >= B || f >= C_out) {
        return;
    }

    int mat_size = W_in * H_in;
    int img_size = mat_size * C_in;
    int out_size = mat_size * C_out;

    float *curr_img = X + b * img_size;
    float *curr_out = O + b * out_size;

   
    for (int w = 0; w < W_in; w += 1)
    {
        for (int h = 0; h < H_in; h += 1)
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

void init_conv(int bbpw, int fbpw, int wbpw, int hbpw, int cbpw, int bbdw, int cbdw, int fdw, int hbdw, int wbdw, int hfdw, int wfbdw) {
}

void dws_conv(float *X, float *F_DW, float *F_1D, float *O, int B, int H_in, int W_in, int C_in, int H_f, int W_f, int N_dw, int H_out, int W_out, int C_out, int stride_h, int stride_w, float* depthwise_output)
{
    float* X_gpu;
    float* F_DW_gpu;
    float* F_1D_gpu;
    float* O_gpu;
    float* depthwise_output_gpu;

    cudaMalloc((void**) &X_gpu, B * C_in * W_in * H_in * sizeof(float));
    cudaMalloc((void**) &F_DW_gpu, N_dw * C_in * H_f * W_f * sizeof(float));
    cudaMalloc((void**) &F_1D_gpu, C_out * C_in * N_dw * sizeof(float));
    cudaMalloc((void**) &O_gpu, B * C_out * W_out * H_out * sizeof(float));
    cudaMalloc((void**) &depthwise_output_gpu, B * W_out * H_out * C_in * N_dw * sizeof(float));

    cudaMemcpy(X_gpu, X, B * C_in * W_in * H_in * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(F_DW_gpu, F_DW, N_dw * C_in * H_f * W_f * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(F_1D_gpu, F_1D, C_out * C_in * N_dw * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dw_gridDim(B, C_in);
    dim3 dw_blockDim(NUM_THREADS, NUM_THREADS);
    dw_conv_gpu<<<dw_gridDim, dw_blockDim>>>(X_gpu, F_DW_gpu, depthwise_output_gpu, B, H_in, W_in, C_in, H_f, W_f, N_dw, H_out, W_out, stride_h, stride_w);
    
    dim3 pw_gridDim(B, C_out);
    dim3 pw_blockDim(NUM_THREADS, NUM_THREADS);
    pw_conv_gpu<<<pw_gridDim, pw_blockDim>>>(depthwise_output_gpu, F_1D_gpu, O_gpu, B, H_out, W_out, C_in * N_dw, C_out);

    cudaMemcpy(O, O_gpu, B * C_out * W_out * H_out * sizeof(float), cudaMemcpyDeviceToHost);
}
