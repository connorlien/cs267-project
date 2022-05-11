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

static void dw_conv_blocked(float *X, float *F_DW, float *O, int B, int H_in, int W_in, int C_in, int H_f, 
    int W_f, int N_dw, int H_out, int W_out, int stride_h, int stride_w, int b_, int c_, int f_, int w_, int h_, int w_f_, int h_f_)
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
        float *curr_img = X + (b + b_) * img_size;
        float *curr_out = O + (b + b_) * temp_out_size;
        for (int c = 0; c < C_b; c += 1)
        {
            // Do 2D Convolution channelwise
            float *curr_channel = curr_img + mat_size * (c + c_);
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
                                float *f_curr = F_DW + f_size * ((c + c_) * N_dw + (f + f_)) + row_major((h_f + h_f_), (w_f + w_f_), W_f);

                                // PTR TO INPUT POSITION
                                int h_curr = (h_f + h_f_) + stride_h * (h + h_);
                                int w_curr = (w_f + w_f_) + stride_w * (w + w_);
                                float *curr_inp = curr_channel + row_major(h_curr, w_curr, W_in);

                                // PTR TO INPUT POSITION
                                float *curr_out_xy = curr_out + temp_out_img_size * ((c + c_) * N_dw + (f_ + f)) + row_major((h + h_), (w + w_), W_out);

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


static void dw_conv(float *X, float *F_DW, float *O, int B, int H_in, int W_in, int C_in, int H_f, int W_f, int N_dw, int H_out, int W_out, int stride_h, int stride_w)
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

void pw_blocked(int B, int H_in, int W_in, int C_in, int C_out, int B_b, int F_b, int W_b, int H_b, int C_b, int b_, int f_, int w_, int h_, int c_, float* F_1D, float* O, float* X) {
    int mat_size = W_in * H_in;
    int img_size = mat_size * C_in;
    int out_size = mat_size * C_out;

    for (int b = 0; b < B_b; b += 1)
    {
        float *curr_img = X + (b_ + b) * img_size;
        float *curr_out = O + (b_ + b) * out_size;

        for (int f = 0; f < F_b; f += 1)
        {
            for (int w = 0; w < W_b; w += 1)
            {
                for (int h = 0; h < H_b; h += 1)
                {
                    float *o_curr = curr_out + mat_size * (f + f_) + row_major((h + h_), (w + w_), W_in);
                    for (int c = 0; c < C_b; c += 1)
                    {
                        float *f_curr = F_1D + (f + f_) * C_in + (c + c_);
                        float *inp_curr = curr_img + mat_size * (c + c_) + row_major((h + h_), (w + w_), W_in);
                        *o_curr += (*f_curr) * (*inp_curr);
                    }
                }
            }
        }
    }
}

static void pw_conv(float *X, float *F_1D, float *O, int B, int H_in, int W_in, int C_in, int C_out)
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

    // b, ci, co, wo, ho, wf, hf [2, 2, 4, 19, 19, 19, 2, 19, 2]
    /* BATCH_BLOCK_PW = 2;
    FILTER_BLOCK_PW = 2;
    WIDTH_BLOCK_PW = 19;
    HEIGHT_BLOCK_PW = 19;
    CHANNEL_BLOCK_PW = 4;

    BATCH_BLOCK_DW = 2;
    CHANNEL_BLOCK_DW = 2;
    FILTER_DW = 4;
    HEIGHT_BLOCK_DW = 19;
    WIDTH_BLOCK_DW = 19;
    HEIGHT_FILTER_BLOCK_DW = 2;
    WIDTH_FILTER_BLOCK_DW = 2; */

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

void dws_conv(float *X, float *F_DW, float *F_1D, float *O, int B, int H_in, int W_in, int C_in, int H_f, int W_f, int N_dw, int H_out, int W_out, int C_out, int stride_h, int stride_w, float* depthwise_output)
{
    dw_conv(X, F_DW, depthwise_output, B, H_in, W_in, C_in, H_f, W_f, N_dw, H_out, W_out, stride_h, stride_w);
    pw_conv(depthwise_output, F_1D, O, B, H_out, W_out, C_in * N_dw, C_out);
}