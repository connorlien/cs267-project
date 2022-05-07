#include <stdio.h>
#include <stdlib.h>
#include "dws-gpu.cu"

void init_conv(int bbpw, int fbpw, int wbpw, int hbpw, int cbpw, int bbdw, int cbdw, int fdw, int hbdw, int wbdw, int hfdw, int wfbdw)
{
    init_conv_gpu(bbpw, fbpw, wbpw, hbpw, cbpw, bbdw, cbdw, fdw, hbdw, wbdw, hfdw, wfbdw);
}

void dws_conv(double *X, double *F_DW, double *F_1D, double *O, int B, int H_in, int W_in, int C_in, int H_f, int W_f, int N_dw, int H_out, int W_out, int C_out, int stride_h, int stride_w, double* depthwise_output)
{
    dws_conv_gpu(X, F_DW, F_1D, O, B, H_in, W_in, C_in, H_f, W_f, N_dw, H_out, W_out, C_out, stride_h, stride_w, depthwise_output);
}
