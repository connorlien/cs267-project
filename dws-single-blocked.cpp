#include <stdio.h>
#include <stdlib.h>

#define min(a, b) (((a) < (b)) ? (a) : (b))
#define row_major(i, j, num_rows) ((i) * (num_rows) + (j))

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

static void single_op_blocked(double *X, double *F_DW, double *F_1D, double *O, int B, int H_in, int W_in, int C_in, int H_f, int W_f, int N_dw, int H_out, int W_out, int C_out, int stride_h, int stride_w, int b_, int c_, int f_, int h_, int w_)
{
    int mat_size = W_in * H_in;
    int f_size = W_f * H_f;
    int img_size = mat_size * C_in;

    int mat_size_out = W_out * H_out;
    int out_size = mat_size_out * C_out;

    int B_b = min(BATCH_BLOCK_DW, B - b_);
    int C_b = min(CHANNEL_BLOCK_DW, C_in - c_);
    int F_b = min(FILTER_DW, N_dw - f_);
    int H_b = min(HEIGHT_BLOCK_DW, H_out - h_);
    int W_b = min(WIDTH_BLOCK_DW, W_out - w_);

    for (int b = 0; b < B_b; b += 1)
    {
        // PTRS TO IMG IN BATCH
        double *curr_img = X + (b + b_) * img_size;
        double *curr_out = O + (b + b_) * out_size;
        for (int c = 0; c < C_b; c += 1)
        {
            for (int f = 0; f < F_b; f += 1)
            {
                for (int h = 0; h < H_b; h += 1)
                {
                    for (int w = 0; w < W_b; w += 1)
                    {
                        // Do 2D Convolution channelwise
                        double *curr_channel = curr_img + mat_size * (c + c_);
                        // MOST LIKELY SET TO 1.
                        // MICROKERNEL - tile if needed.
                        
                        double temp = 0.0;
                        // Depthwise conv
                        for (int w_f = 0; w_f < W_f; w_f += 1)
                        {
                            for (int h_f = 0; h_f < H_f; h_f += 1)
                            {
                                // PTR TO CURRENT POSITION IN FILTER
                                double *f_curr = F_DW + f_size * ((c + c_) * N_dw + (f + f_)) + row_major(h_f, w_f, W_f);

                                // PTR TO INPUT POSITION
                                int h_curr = h_f + stride_h * (h + h_);
                                int w_curr = w_f + stride_w * (w + w_);
                                double *curr_inp = curr_channel + row_major(h_curr, w_curr, W_in);

                                // PTR TO INPUT POSITION
                                temp += *f_curr * *curr_inp;
                            }
                        }
                        // store temp in somewhere

                        // LOOP 
                        for (int f_1d = 0; f_1d < C_out; f_1d += 1) {
                            double *o_curr = curr_out + mat_size_out * f_1d + row_major((h + h_), (w + w_), W_out);
                            double *curr = F_1D + f_1d * (N_dw * C_in) + N_dw * (c + c_) + (f + f_);
                            *o_curr += temp * (*curr);
                        }
                    }
                }
            }
        }
    }
}

static void single_op(double *X, double *F_DW, double *F_1D, double *O, int B, int H_in, int W_in, int C_in, int H_f, int W_f, int N_dw, int H_out, int W_out, int C_out, int stride_h, int stride_w)
{
    for (int b = 0; b < B; b += BATCH_BLOCK_DW)
    {
        for (int c = 0; c < C_in; c += CHANNEL_BLOCK_DW)
        {
            for (int f = 0; f < N_dw; f += FILTER_DW)
            {
                for (int h = 0; h < H_out; h += HEIGHT_BLOCK_DW)
                {
                    for (int w = 0; w < W_out; w += WIDTH_BLOCK_DW)
                    {
                        single_op_blocked(X, F_DW, F_1D, O, B, H_in, W_in, C_in, H_f, W_f, N_dw, H_out, W_out, C_out, stride_h, stride_w, b, c, f, h, w);
                    }
                }
            }
        }
    }
}

void init_conv(int bbpw, int fbpw, int wbpw, int hbpw, int cbpw, int bbdw, int cbdw, int fdw, int hbdw, int wbdw, int hfdw, int wfbdw) {
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
}

void dws_conv(double *X, double *F_DW, double *F_1D, double *O, int B, int H_in, int W_in, int C_in, int H_f, int W_f, int N_dw, int H_out, int W_out, int C_out, int stride_h, int stride_w, double* depthwise_output)
{
    single_op(X, F_DW, F_1D, O, B, H_in, W_in, C_in, H_f, W_f, N_dw, H_out, W_out, C_out, stride_h, stride_w);
}