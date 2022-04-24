#include <stdio.h>
#include <stdlib.h>

#define min(a, b) (((a) < (b)) ? (a) : (b))
#define row_major(i, j, num_rows) ((i) * (num_rows) + (j))
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

static void dw_conv_blocked(double *X, double *F_DW, double *O, int B, int H_in, int W_in, int C_in, int H_f, int W_f, int N_dw, int H_out, int W_out, int stride_h, int stride_w, int b_, int c_, int f_, int w_, int h_, int w_f_, int h_f_)
{
    int mat_size = W_in * H_in;
    int f_size = W_f * H_f;
    int img_size = mat_size * C_in;

    int temp_out_img_size = W_out * H_out;
    int temp_out_size = temp_out_img_size * N_dw * C_in;

    int B_b = min(BATCH_BLOCK_DW, B - b_);
    int C_b = min(CHANNEL_BLOCK_DW, C_in - c_);
    int F_b = min(FILTER_DW, N_dw - f_);
    int W_b = min(WIDTH_BLOCK_DW, W_out - w_);
    int H_b = min(HEIGHT_BLOCK_DW, H_out - h_);
    int W_f_b = min(WIDTH_FILTER_BLOCK_DW, W_f - w_f_);
    int H_f_b = min(HEIGHT_FILTER_BLOCK_DW, H_f - h_f_);

    for (int b = 0; b < B_b; b += 1)
    {
        // PTRS TO IMG IN BATCH
        double *curr_img = X + (b + b_) * img_size;
        double *curr_out = O + (b + b_) * temp_out_size;
        for (int c = 0; c < C_b; c += 1)
        {
            // Do 2D Convolution channelwise
            double *curr_channel = curr_img + mat_size * (c + c_);
            // Filters are 2D
            for (int f = 0; f < F_b; f += 1)
            {
                for (int w = 0; w < W_b; w += 1)
                {
                    for (int h = 0; h < H_b; h += 1)
                    {
                        // MICROKERNEL - tile if needed.
                        for (int w_f = 0; w_f < W_f_b; w_f += 1)
                        {
                            for (int h_f = 0; h_f < H_f_b; h_f += 1)
                            {
                                // PTR TO CURRENT POSITION IN FILTER
                                double *f_curr = F_DW + f_size * ((c + c_) * N_dw + (f + f_)) + row_major((h_f + h_f_), (w_f + w_f_), W_f);

                                // PTR TO INPUT POSITION
                                int h_curr = (h_f + h_f_) + stride_h * (h + h_);
                                int w_curr = (w_f + w_f_) + stride_w * (w + w_);
                                double *curr_inp = curr_channel + row_major(h_curr, w_curr, W_in);

                                // PTR TO INPUT POSITION
                                double *curr_out_xy = curr_out + temp_out_img_size * ((c + c_) * N_dw + (f_ + f)) + row_major((h + h_), (w + w_), W_out);

                                // CONVOLVE
                                *curr_out_xy = *curr_out_xy + *f_curr * *curr_inp;
                            }
                        }
                    }
                }
            }
        }
    }
}


static void dw_conv(double *X, double *F_DW, double *O, int B, int H_in, int W_in, int C_in, int H_f, int W_f, int N_dw, int H_out, int W_out, int stride_h, int stride_w)
{
    for (int b = 0; b < B; b += BATCH_BLOCK_DW)
    {
        for (int c = 0; c < C_in; c += CHANNEL_BLOCK_DW)
        {
            for (int f = 0; f < N_dw; f += FILTER_DW)
            {
                for (int w = 0; w < W_out; w += WIDTH_BLOCK_DW)
                {
                    for (int h = 0; h < H_out; h += HEIGHT_BLOCK_DW)
                    {
                        for (int w_f = 0; w_f < W_f; w_f += WIDTH_FILTER_BLOCK_DW)
                        {
                            for (int h_f = 0; h_f < H_f; h_f += HEIGHT_FILTER_BLOCK_DW)
                            {
                                dw_conv_blocked(X, F_DW, O, B, H_in, W_in, C_in, H_f, W_f, N_dw, H_out, W_out, stride_h, stride_w, b, c, f, w, h, w_f, h_f);
                            }
                        }
                    }
                }
            }
        }
    }
}

void pw_microkernel_1x1x8(double *input, double *filter, double *output, int mat_size) {
    // Declare
    double A0, A1, A2, A3, A4, A5, A6, A7;
    double B0, B1, B2, B3, B4, B5, B6, B7;
    double C0, C1, C2, C3, C4, C5, C6, C7;

    // Load
    A0 = *(input + mat_size * (0));
    A1 = *(input + mat_size * (1));
    A2 = *(input + mat_size * (2));
    A3 = *(input + mat_size * (3));
    A4 = *(input + mat_size * (4));
    A5 = *(input + mat_size * (5));
    A6 = *(input + mat_size * (6));
    A7 = *(input + mat_size * (7));

    B0 = filter[0];
    B1 = filter[1];
    B2 = filter[2];
    B3 = filter[3];
    B4 = filter[4];
    B5 = filter[5];
    B6 = filter[6];
    B7 = filter[7];

    // Compute
    C0 = A0 * B0; 
    C1 = A1 * B1; 
    C2 = A2 * B2; 
    C3 = A3 * B3; 
    C4 = A4 * B4; 
    C5 = A5 * B5;
    C6 = A6 * B6; 
    C7 = A7 * B7; 

    *output += C0 + C1 + C2 + C3 + C4 + C5 + C6 + C7;
}

void pw_blocked(int B, int H_in, int W_in, int C_in, int C_out, int B_b, int F_b, int W_b, int H_b, int C_b, int b_, int f_, int w_, int h_, int c_, double* F_1D, double* O, double* X) {
    int mat_size = W_in * H_in;
    int img_size = mat_size * C_in;
    int out_size = mat_size * C_out;

    for (int b = 0; b < B_b; b += 1)
    {
        double *curr_img = X + (b_ + b) * img_size;
        double *curr_out = O + (b_ + b) * out_size;

        for (int f = 0; f < F_b; f += 1)
        {
            for (int w = 0; w < W_b; w += 1)
            {
                for (int h = 0; h < H_b; h += 1)
                {
                    double *o_curr = curr_out + mat_size * (f + f_) + row_major((h + h_), (w + w_), W_in);
                    int c = 0;
                    for (; c < C_b / 8 * 8; c += 8)
                    {
                        double *inp_curr = curr_img + mat_size * (c + c_) + row_major((h + h_), (w + w_), W_in);
                        double *f_curr = F_1D + (f + f_) * C_in + (c + c_);
                        pw_microkernel_1x1x8(inp_curr, f_curr, o_curr, mat_size);
                    }
                    for (; c < C_b; c += 1)
                    {
                        double *f_curr = F_1D + (f + f_) * C_in + (c + c_);
                        double *inp_curr = curr_img + mat_size * (c + c_) + row_major((h + h_), (w + w_), W_in);
                        *o_curr += (*f_curr) * (*inp_curr);
                    }
                }
            }
        }
    }
}

static void pw_conv(double *X, double *F_1D, double *O, int B, int H_in, int W_in, int C_in, int C_out)
{
    for (int b = 0; b < B; b += BATCH_BLOCK_PW)
    {
        int B_b = min(BATCH_BLOCK_PW, B - b);
        for (int f = 0; f < C_out; f += FILTER_BLOCK_PW)
        {
            int F_b = min(FILTER_BLOCK_PW, C_out - f);
            for (int w = 0; w < W_in; w += WIDTH_BLOCK_PW)
            {
                int W_b = min(WIDTH_BLOCK_PW, W_in - w);
                for (int h = 0; h < H_in; h += HEIGHT_BLOCK_PW)
                {
                    int H_b = min(HEIGHT_BLOCK_PW, H_in - h);
                    for (int c = 0; c < C_in; c += CHANNEL_BLOCK_PW)
                    {
                        int C_b = min(CHANNEL_BLOCK_PW, C_in - c);
                        pw_blocked(B, H_in, W_in, C_in, C_out, B_b, F_b, W_b, H_b, C_b, b, f, w, h, c, F_1D, O, X);
                    }
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

    VEC_SIZE = AVX_BITS / sizeof(double);
}

void dws_conv(double *X, double *F_DW, double *F_1D, double *O, int B, int H_in, int W_in, int C_in, int H_f, int W_f, int N_dw, int H_out, int W_out, int C_out, int stride_h, int stride_w, double* depthwise_output)
{
    dw_conv(X, F_DW, depthwise_output, B, H_in, W_in, C_in, H_f, W_f, N_dw, H_out, W_out, stride_h, stride_w);
    pw_conv(depthwise_output, F_1D, O, B, H_out, W_out, C_in * N_dw, C_out);
}