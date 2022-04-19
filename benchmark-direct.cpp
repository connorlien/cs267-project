#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <cmath> // For: fabs

#include <cblas.h>
#include <stdio.h>

#define row_major(i, j, num_rows) ((i) * (num_rows) + (j))

extern "C" {
    extern void dws_conv(double*, double*, double*, double*, int, int, int, int, int, int, int, int, int, int, int, int);
}

void fill(double* p, int n) {
    static std::random_device rd;
    static std::default_random_engine gen(rd());
    static std::uniform_real_distribution<> dis(-1.0, 1.0);
    for (int i = 0; i < n; ++i)
        p[i] = 2 * dis(gen) - 1;
}

void fillDet(double* p, int n){
    for (int i = 0; i < n; ++i){
	    p[i] = i + 1.0;
    }
}

void fillConst(double* p, int n, double k){
    for (int i = 0; i < n; ++i) {
        p[i] = k;
	}
}

void fillZero(double* p, int n){
    fillConst(p, n, 0.0);
}

void printTensor(double* p, int B, int W, int H, int C) {
    // for (int i = 0; i < n * n; ++i) {
    //     if (i > 0 && i % n == 0) {
    //         fprintf(stderr, "\n");
    //     }
    //     fprintf(stderr, "%f ", p[i]);
	// }
    int mat_size = H * W;
    int img_size = mat_size * C;

    for (int b = 0; b < B; b += 1) {
        fprintf(stderr, "Image %d \n", b);
        for (int c = 0; c < C; c += 1) {
            fprintf(stderr, "Channel %d \n", c);
            for (int w = 0; w < W; w += 1) {
                for (int h = 0; h < H; h += 1) {
                    double val = *(p + b * img_size + c * mat_size + row_major(h, w, W));
                    fprintf(stderr, "%f ", val);
                }
                fprintf(stderr, "\n");
            }
        }
    }
}

/* The benchmarking program */
// double *X, double *F_DW, double *F_1D, double *O, int B, int H_in, int W_in, int C_in, int H_f, int W_f, int N_dw, int H_out, int W_out, int C_out, int stride_h, int stride_w)
int main(int argc, char** argv) {
    std::cout << std::fixed << std::setprecision(2);

    int B = 1;
    int C_in = 1;
    int W_in = 4;
    int H_in = 4;
    double* input = (double *) calloc(B * C_in * W_in * H_in, sizeof(double));
    fillConst(input, B * C_in * W_in * H_in, 1.0);
    fprintf(stderr, "Input \n");
    printTensor(input, B, W_in, H_in, C_in);
    fprintf(stderr, "-----  \n");

    int C_out = 1;
    int W_out = 3;
    int H_out = 3;
    double* output = (double *) calloc(B * C_out * W_out * H_out, sizeof(double));
    fillConst(input, B * C_in * W_in * H_in, 1.0);
    fprintf(stderr, "Output before \n");
    printTensor(output, B, W_out, H_out, C_out);
    fprintf(stderr, "----- \n");

    int N_dw = 1;
    int H_f = 2;
    int W_f = 2;
    double* F_DW = (double *) calloc(N_dw * C_in * H_f * W_f, sizeof(double));
    fillConst(F_DW, N_dw * C_in * H_f * W_f, 1.0);

    int N_1d = C_out;
    double* F_1D = (double *) calloc(N_1d * C_in * N_dw, sizeof(double));
    fillConst(F_1D, N_1d * C_in * N_dw, 1.0);

    int stride_h = 2;
    int stride_w = 2;

    dws_conv(input, F_DW, F_1D, output, B, H_in, W_in, C_in, H_f, W_f, N_dw, H_out, W_out, C_out, stride_h, stride_w);
    fprintf(stderr, "Output after \n");
    printTensor(output, B, W_out, H_out, C_out);
    fprintf(stderr, "----- \n");

    return 0;
}