#include <stdio.h>
#include <stdlib.h>

#define NUM_THREADS 16
#define min(a, b) (((a) < (b)) ? (a) : (b))
#define row_major(i, j, num_rows) ((i) * (num_rows) + (j))

__global__ void dw_conv(double *X, double *F_DW, double *O, int B, int H_in, int W_in, int C_in, int H_f, int W_f, int N_dw, int H_out, int W_out, int stride_h, int stride_w) {
	
	// Compute batch and channel for this thread
    int b = threadIdx.x + blockIdx.x * blockDim.x; 
    int c = threadIdx.y + blockIdx.y * blockDim.y; 

    // Pre-computations
    int mat_size = W_in * H_in;
    int f_size = W_f * H_f;
    int img_size = mat_size * C_in;

    int temp_out_img_size = W_out * H_out;
    int temp_out_size = temp_out_img_size * N_dw * C_in;

    // PTRS TO IMG IN BATCH
    double *curr_img = X + b * img_size;
    double *curr_out = O + b * temp_out_size;

    // Do 2D Convolution channelwise
    double *curr_channel = curr_img + mat_size * c;

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
                        double *f_curr = F_DW + f_size * (c * N_dw + f) + row_major(h_f, w_f, W_f);

                        // PTR TO INPUT POSITION
                        int h_curr = h_f + stride_h * h;
                        int w_curr = w_f + stride_w * w;
                        double *curr_inp = curr_channel + row_major(h_curr, w_curr, W_in);

                        // PTR TO INPUT POSITION
                        double *curr_out_xy = curr_out + temp_out_img_size * (c * N_dw + f) + row_major(h, w, W_out);

                        // CONVOLVE
                        *curr_out_xy = *curr_out_xy + *f_curr * *curr_inp;
                    }
                }
            }
        }
    }
}

__global__ void pw_conv(double *X, double *F_1D, double *O, int B, int H_in, int W_in, int C_in, int C_out)
{
    // Compute batch and channel for this thread
    int b = threadIdx.x + blockIdx.x * blockDim.x; 

    int mat_size = W_in * H_in;
    int img_size = mat_size * C_in;
    int out_size = mat_size * C_out;

    double *curr_img = X + b * img_size;
    double *curr_out = O + b * out_size;

    for (int f = 0; f < C_out; f += 1)
    {
        for (int w = 0; w < W_in; w += 1)
        {
            for (int h = 0; h < H_in; h += 1)
            {
                double *o_curr = curr_out + mat_size * f + row_major(h, w, W_in);
                for (int c = 0; c < C_in; c += 1)
                {
                    double *f_curr = F_1D + f * C_in + c;
                    double *inp_curr = curr_img + mat_size * c + row_major(h, w, W_in);
                    *o_curr += (*f_curr) * (*inp_curr);
                }
            }
        }
    }
}

void print_tensor(double *X, int size, const char *name)
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

void dws_conv(double *X, double *F_DW, double *F_1D, double *O, int B, int H_in, int W_in, int C_in, int H_f, int W_f, int N_dw, int H_out, int W_out, int C_out, int stride_h, int stride_w, double* depthwise_output)
{
    particle_t* X_gpu;
    particle_t* F_DW_gpu;
    particle_t* F_1D_gpu;
    particle_t* O_gpu;
    particle_t* depthwise_output_gpu;

    cudaMalloc((void**) &X_gpu, B * C_in * W_in * H_in * sizeof(double));
    cudaMalloc((void**) &F_DW_gpu, N_dw * C_in * H_f * W_f * sizeof(double));
    cudaMalloc((void**) &F_1D_gpu, N_1d * C_in * N_dw * sizeof(double));
    cudaMalloc((void**) &O_gpu, B * C_out * W_out * H_out * sizeof(double));
    cudaMalloc((void**) &depthwise_output_gpu, B * W_out * H_out * C_in * N_dw * sizeof(double));

    cudaMemcpy(X_gpu, X, B * C_in * W_in * H_in * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(F_DW_gpu, F_DW, N_dw * C_in * H_f * W_f * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(F_1D_gpu, F_1D, N_1d * C_in * N_dw * sizeof(double), cudaMemcpyHostToDevice);

    dim3 dimGrid(B, C_in);
    dim3 dimBlock(NUM_THREADS, NUM_THREADS);
    dw_conv<<<dimGrid, dimBlock>>>(X_gpu, F_DW_gpu, depthwise_output_gpu, B, H_in, W_in, C_in, H_f, W_f, N_dw, H_out, W_out, stride_h, stride_w);
    pw_conv<<<B, NUM_THREADS>>>(depthwise_output_gpu, F_1D_gpu, O_gpu, B, H_out, W_out, C_in * N_dw, C_out);

    cudaMemcpy(O, O_gpu, B * C_out * W_out * H_out * sizeof(double), cudaMemcpyDeviceToHost);
}
