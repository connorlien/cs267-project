#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <cmath>
#include <cstring>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include "dws-gpu.h"

#define row_major(i, j, num_rows) ((i) * (num_rows) + (j))

// =================
// Helper Functions
// =================
// I/O routines
void save_tensor(std::ofstream& fsave, float *tensor, int size, const char *name) {
    fsave << name << std::endl;
    for (int i = 0; i < size; ++i) {
        fsave << tensor[i] << " ";
    }
    fsave << std::endl;
}


// Command Line Option Processing
int find_arg_idx(int argc, char** argv, const char* option) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], option) == 0) {
            return i;
        }
    }
    return -1;
}

int find_int_arg(int argc, char** argv, const char* option, int default_value) {
    int iplace = find_arg_idx(argc, argv, option);

    if (iplace >= 0 && iplace < argc - 1) {
        return std::stoi(argv[iplace + 1]);
    }

    return default_value;
}

char* find_string_option(int argc, char** argv, const char* option, char* default_value) {
    int iplace = find_arg_idx(argc, argv, option);

    if (iplace >= 0 && iplace < argc - 1) {
        return argv[iplace + 1];
    }

    return default_value;
}

// Tensor helper functions
void fill(float* p, int n, int seed) {
    static std::random_device rd;
    static std::default_random_engine gen(seed ? seed : rd());
    static std::uniform_real_distribution<> dis(-1.0, 1.0);
    for (int i = 0; i < n; ++i) {
        p[i] = dis(gen);
    }
}

__global__ void fillGpu(float* p, int n, int seed) {
    curandState state;
    curand_init(seed, 0, 0, &state);
    for (int i = 0; i < n; ++i) {
        p[i] = curand_uniform(&state);
    }
}

void fillDet(float* p, int n){
    for (int i = 0; i < n; ++i){
	    p[i] = i + 1.0;
    }
}

void fillConst(float* p, int n, float k){
    for (int i = 0; i < n; ++i) {
        p[i] = k;
	}
}

void fillOne(float* p, int n){
    fillConst(p, n, 1.0);
}

void fillZero(float* p, int n){
    fillConst(p, n, 0.0);
}

__global__ void fillZeroGpu(float* p, int n){
    for (int i = 0; i < n; ++i) {
        p[i] = 0;
	}
}

void printTensor(float* p, int B, int W, int H, int C) {
    int mat_size = H * W;
    int img_size = mat_size * C;

    for (int b = 0; b < B; b += 1) {
        fprintf(stderr, "Image %d \n", b);
        for (int c = 0; c < C; c += 1) {
            fprintf(stderr, "Channel %d \n", c);
            for (int w = 0; w < W; w += 1) {
                for (int h = 0; h < H; h += 1) {
                    float val = *(p + b * img_size + c * mat_size + row_major(h, w, W));
                    fprintf(stderr, "%f ", val);
                }
                fprintf(stderr, "\n");
            }
        }
    }
}

void benchmark(bool all_sizes = false) {
    std::vector<int> tensor_sizes;
    std::vector<int> kernel_sizes;

    // int bbpw, int fbpw, int wbpw, int hbpw, int cbpw, int bbdw, int cbdw, int fdw, int hbdw, int wbdw, int hfdw, int wfbd
    std::vector<int> bbpw;
    std::vector<int> fbpw;
    std::vector<int> wbpw;
    std::vector<int> hbpw;
    std::vector<int> cbpw;
    std::vector<int> bbdw;
    std::vector<int> cbdw;
    std::vector<int> fdw;
    std::vector<int> hbdw;
    std::vector<int> wbdw;
    std::vector<int> hfdw;
    std::vector<int> wfbd;

    if (all_sizes) {
        tensor_sizes.assign({ 
            4, 8, 16, 31,  32,  33,  63,  64,  65,  95,  96,  97,  127, 128, 129, 
            159, 160, 161, 191, 192, 193, 223, 224, 225, 255, 256, 257, 287, 288, 
            289, 319, 320, 321, 351, 352, 353, 383, 384, 385, 415, 416, 417, 447, 
            448, 449, 479, 480, 481, 511, 512
        });
        kernel_sizes.assign({2, 3, 5, 7});
        bbpw.assign({126, 71, 39, 24, 23, 13, 13, 12, 11, 11, 10, 10, 10, 10, 10, 10, 9, 9, 9, 9, 9, 9, 8, 8, 129, 92, 44, 24, 24, 13, 13, 12, 12, 11, 10, 10, 10, 10, 10, 10, 9, 9, 9, 9, 9, 9, 8, 8, 126, 50, 26, 26, 13, 13, 12, 12, 12, 10, 10, 10, 10, 10, 10, 9, 9, 9, 9, 9, 9, 8, 8, 129, 59, 27, 27, 13, 13, 12, 12, 12, 10, 10, 10, 10, 10, 10, 9, 9, 9, 9, 9, 9, 8, 8});
        fbpw.assign({4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4});
        wbpw.assign({2, 3, 4, 6, 6, 8, 8, 8, 8, 8, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 1, 3, 4, 6, 6, 8, 8, 8, 8, 8, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 2, 4, 5, 5, 8, 8, 8, 8, 8, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 1, 4, 5, 5, 8, 8, 8, 8, 8, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10});
        hbpw.assign({2, 3, 4, 6, 6, 8, 8, 8, 8, 8, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 1, 3, 4, 6, 6, 8, 8, 8, 8, 8, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 2, 4, 5, 5, 8, 8, 8, 8, 8, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 1, 4, 5, 5, 8, 8, 8, 8, 8, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10});
        cbpw.assign({9, 8, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 10, 8, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 9, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 10, 8, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6});
        bbdw.assign({118, 71, 42, 26, 25, 13, 13, 12, 11, 11, 10, 10, 9, 9, 9, 9, 9, 9, 9, 8, 8, 8, 8, 8, 129, 69, 37, 21, 21, 11, 11, 10, 10, 10, 9, 9, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 65, 30, 17, 17, 9, 9, 9, 9, 9, 8, 8, 8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 75, 27, 14, 14, 8, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6});
        cbdw.assign({3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2});
        fdw.assign({4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4});
        hbdw.assign({4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4});
        wbdw.assign({2, 3, 4, 5, 6, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 1, 3, 4, 5, 5, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 2, 3, 5, 5, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 1, 3, 4, 4, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7});
        hfdw.assign({2, 3, 4, 5, 6, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 1, 3, 4, 5, 5, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 2, 3, 5, 5, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 1, 3, 4, 4, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7});
        wfbd.assign({2, 3, 4, 5, 6, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 1, 3, 4, 5, 5, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 2, 3, 5, 5, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 1, 3, 4, 4, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7});
    } else {
        tensor_sizes.assign({
            4, 8, 16, 31,  32,  96,  97,  127, 128, 129, 191, 192, 229, 255, 256,
            257, 319, 320, 321, 417, 479, 480, 511, 512
        });

        kernel_sizes.assign({3});
        bbpw.assign({129, 92, 44, 24, 24, 13, 13, 12, 12, 11, 10, 10, 10, 10, 10, 10, 9, 9, 9, 9, 9, 9, 8, 8});
        fbpw.assign({4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4});
        wbpw.assign({1, 3, 4, 6, 6, 8, 8, 8, 8, 8, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10});
        hbpw.assign({1, 3, 4, 6, 6, 8, 8, 8, 8, 8, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10});
        cbpw.assign({10, 8, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6});
        bbdw.assign({129, 69, 37, 21, 21, 11, 11, 10, 10, 10, 9, 9, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 7});
        cbdw.assign({3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2});
        fdw.assign({4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4});
        hbdw.assign({1, 3, 4, 5, 5, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9});
        wbdw.assign({1, 3, 4, 5, 5, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9});
        hfdw.assign({4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3});
        wfbd.assign({4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3}); 
    }

    std::sort(tensor_sizes.begin(), tensor_sizes.end());
    std::sort(kernel_sizes.begin(), kernel_sizes.end());
    int nmax = tensor_sizes[tensor_sizes.size() - 1];
    int kmax = kernel_sizes[kernel_sizes.size() - 1];

    /* Set a seed. */
    int seed = 0;

    fprintf(stdout, "Starting benchmarking...\n");

    /* Set default variables and allocate space. */
    int B = 128;
    int C_in = 3;
    int W_in = 4;
    int H_in = 4;
    
    int C_out = 3;
    int W_out = 2;
    int H_out = 2;
    
    int N_dw = 3;
    int H_f = 2;
    int W_f = 2;

    int N_1d = C_out;
    int stride_h = 2;
    int stride_w = 2;

    float* X_gpu;
    float* F_DW_gpu;
    float* F_1D_gpu;
    float* O_gpu;
    float* depthwise_output_gpu;

    cudaMalloc((void**) &X_gpu, B * C_in * nmax * nmax * sizeof(float));
    cudaMalloc((void**) &F_DW_gpu, N_dw * C_in * kmax * kmax * sizeof(float));
    cudaMalloc((void**) &F_1D_gpu, N_1d * C_in * N_dw * sizeof(float));
    cudaMalloc((void**) &O_gpu, B * C_out * nmax * nmax * sizeof(float));
    cudaMalloc((void**) &depthwise_output_gpu, B * nmax * nmax * C_in * N_dw * sizeof(float));

    /* For each tensor size */
    int idx = 0;
    for (int k : kernel_sizes) {
        /* For each kernel size */
        for (int n : tensor_sizes) {
            init_conv(bbpw[idx], fbpw[idx], wbpw[idx], hbpw[idx], cbpw[idx],
            bbdw[idx], cbdw[idx], fdw[idx], hbdw[idx], wbdw[idx], hfdw[idx], wfbd[idx]);
            
            W_in = n;
            H_in = n;
            H_f = k;
            W_f = k;
            W_out = floor((n - k) / stride_w + 1);
            H_out = floor((n - k) / stride_h + 1);

            if (H_out == 0 || W_out == 0) {
                continue;
            }

            fillGpu<<<1, 1>>>(X_gpu, B * C_in * W_in * H_in, seed);
            fillGpu<<<1, 1>>>(F_DW_gpu, N_dw * C_in * H_f * W_f, seed);
            fillGpu<<<1, 1>>>(F_1D_gpu, N_1d * C_in * N_dw, seed);
            fillZeroGpu<<<1, 1>>>(depthwise_output_gpu, B * W_out * H_out * C_in * N_dw);

            /* Time a "sufficiently long" sequence of calls to reduce noise */
            double avg_time = 0.0, seconds = -1.0;
            double timeout = 0.1; // "sufficiently long" := at least 1/10 second.
            for (int n_iterations = 1; seconds < timeout; n_iterations *= 2) {
                /* Warm-up */
                dws_conv(X_gpu, F_DW_gpu, F_1D_gpu, O_gpu, B, H_in, W_in, C_in, H_f, W_f, N_dw, H_out, W_out, C_out, stride_h, stride_w, depthwise_output_gpu);

                /* Benchmark n_iterations runs of square_dgemm */
                auto start = std::chrono::steady_clock::now();
                for (int it = 0; it < n_iterations; ++it) {
                    dws_conv(X_gpu, F_DW_gpu, F_1D_gpu, O_gpu, B, H_in, W_in, C_in, H_f, W_f, N_dw, H_out, W_out, C_out, stride_h, stride_w, depthwise_output_gpu);
                }
                auto end = std::chrono::steady_clock::now();
                std::chrono::duration<double> diff = end - start;
                seconds = diff.count();

                /*  compute average time */
                avg_time = seconds / n_iterations;
            }

            fprintf(stdout, "Tensor Size: %d  \tKernel Size: %d\tTime: %f s\n", n , k, avg_time);
            idx += 1;
            
        }
    }

    cudaFree(X_gpu);
    cudaFree(F_DW_gpu);
    cudaFree(F_1D_gpu);
    cudaFree(O_gpu);
    cudaFree(depthwise_output_gpu);
}

void run(int argc, char** argv) {
    std::cout << std::fixed << std::setprecision(2);

    int debug = find_int_arg(argc, argv, "-debug", 0);
    int correctness = find_int_arg(argc, argv, "-correctness", 0);
    int seed = find_int_arg(argc, argv, "-s", 0);

    char* savename = find_string_option(argc, argv, "-o", nullptr);
    std::ofstream fsave(savename);

    int B = find_int_arg(argc, argv, "-B", 2);
    int C_in = find_int_arg(argc, argv, "-C_in", 3);
    int W_in = find_int_arg(argc, argv, "-W_in", 4);
    int H_in = find_int_arg(argc, argv, "-H_in", 4);
    
    int C_out = find_int_arg(argc, argv, "-C_out", 3);
    int W_out = find_int_arg(argc, argv, "-W_out", 2);
    int H_out = find_int_arg(argc, argv, "-H_out", 2);
    
    int N_dw = find_int_arg(argc, argv, "-N_dw", 1); // FOR TESTING.
    int H_f = find_int_arg(argc, argv, "-H_f", 2);
    int W_f = find_int_arg(argc, argv, "-W_f", 2);

    int N_1d = C_out;
    int stride_h = find_int_arg(argc, argv, "-stride_h", 2);
    int stride_w = find_int_arg(argc, argv, "-stride_w", 2);

    float* input = (float *) calloc(B * C_in * W_in * H_in, sizeof(float));
    float* F_DW = (float *) calloc(N_dw * C_in * H_f * W_f, sizeof(float));
    float* F_1D = (float *) calloc(N_1d * C_in * N_dw, sizeof(float));
    float* output = (float *) calloc(B * C_out * W_out * H_out, sizeof(float));
    float *depthwise_output = (float *) calloc(B * W_out * H_out * C_in * N_dw, sizeof(float));

    fill(input, B * C_in * W_in * H_in, seed);
    fill(F_DW, N_dw * C_in * H_f * W_f, seed);
    fill(F_1D, N_1d * C_in * N_dw, seed);

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

    cudaMemcpy(X_gpu, input, B * C_in * W_in * H_in * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(F_DW_gpu, F_DW, N_dw * C_in * H_f * W_f * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(F_1D_gpu, F_1D, C_out * C_in * N_dw * sizeof(float), cudaMemcpyHostToDevice);

    if (debug != 0) {
        fprintf(stderr, "Input \n");
        printTensor(input, B, W_in, H_in, C_in);
        fprintf(stderr, "-----  \n");
        fprintf(stderr, "Output before \n");
        printTensor(output, B, W_out, H_out, C_out);
        fprintf(stderr, "----- \n");
        // TODO: PRINT FILTERS
    }

    dws_conv(X_gpu, F_DW_gpu, F_1D_gpu, O_gpu, B, H_in, W_in, C_in, H_f, W_f, N_dw, H_out, W_out, C_out, stride_h, stride_w, depthwise_output_gpu);

    cudaMemcpy(output, O_gpu, B * C_out * W_out * H_out * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(X_gpu);
    cudaFree(F_DW_gpu);
    cudaFree(F_1D_gpu);
    cudaFree(O_gpu);
    cudaFree(depthwise_output_gpu);

    if (debug != 0) {
        fprintf(stderr, "Output after \n");
        printTensor(output, B, W_out, H_out, C_out);
        fprintf(stderr, "----- \n");
    }

    if (fsave) {
        fsave << B << " " << H_in  << " " << W_in  << " " << C_in  << " " << H_f << " " << W_f << " " << N_dw  << " " << H_out  << " " << W_out  << " " << C_out  << " " << stride_h  << " " << stride_w  << std::endl;
        save_tensor(fsave, input, B * C_in * W_in * H_in, "Input");
        save_tensor(fsave, F_DW, N_dw * C_in * H_f * W_f, "Filter-depthwise");
        save_tensor(fsave, F_1D, N_1d * C_in * N_dw, "Filter-1D");
        save_tensor(fsave, output, B * C_out * W_out * H_out, "Output");
        fsave.close();
    }
}

int main(int argc, char** argv) {
    init_conv(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
    int bench = find_int_arg(argc, argv, "-benchmark", 0);
    if (bench == 1) {
        bool all_sizes = find_int_arg(argc, argv, "-all", 0) == 1;
        benchmark(all_sizes);
    } else {
        run(argc, argv);
    }
    return 0;
}