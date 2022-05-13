#include <stdio.h>
#include <stdlib.h>

#define min(a, b) (((a) < (b)) ? (a) : (b))
#define row_major(i, j, num_rows) ((i) * (num_rows) + (j))

int BATCH_BLOCK_PW;
int WIDTH_BLOCK_PW;
int HEIGHT_BLOCK_PW;

int BATCH_BLOCK_DW;
int HEIGHT_BLOCK_DW;
int WIDTH_BLOCK_DW;

static void dw_conv(float *X, float *F_DW, float *O, int B, int H_in, int W_in, int C_in, int H_f, int W_f, int N_dw, int H_out, int W_out, int stride_h, int stride_w)
{
    int mat_size = W_in * H_in;
    int f_size = W_f * H_f;
    int img_size = mat_size * C_in;

    int temp_out_img_size = W_out * H_out;
    int temp_out_size = temp_out_img_size * N_dw * C_in;
    for (int b_ = 0; b_ < B; b_ += BATCH_BLOCK_DW)
    {
        int B_b = min(BATCH_BLOCK_DW, B - b_);
        for (int c_ = 0; c_ < C_in; c_ += 1)
        {
            for (int h_ = 0; h_ < H_out; h_ += HEIGHT_BLOCK_DW)
            {
                int H_b = min(HEIGHT_BLOCK_DW, H_out - h_);
                for (int w_ = 0; w_ < W_out; w_ += WIDTH_BLOCK_DW)
                {
                    int W_b = min(WIDTH_BLOCK_DW, W_out - w_);

                    // BLOCKING
                    for (int b = 0; b < B_b; b += 1)
                    {
                        // PTRS TO IMG IN BATCH
                        float *curr_img = X + (b + b_) * img_size;
                        float *curr_out = O + (b + b_) * temp_out_size;

                        // Do 2D Convolution channelwise
                        float *curr_channel = curr_img + mat_size * (c_);
                        // Filters are 2D

                        for (int h = 0; h < H_b; h += 1)
                        {
                            for (int w = 0; w < W_b; w += 1)
                            {
                                for (int h_f = 0; h_f < H_f; h_f += 1)
                                {
                                    // MICROKERNEL - tile if needed.
                                    for (int w_f = 0; w_f < W_f; w_f += 1)
                                    {

                                        // PTR TO CURRENT POSITION IN FILTER
                                        float *f_curr = F_DW + f_size * (c_) + row_major((h_f), (w_f), W_f);

                                        // PTR TO INPUT POSITION
                                        int h_curr = (h_f) + stride_h * (h + h_);
                                        int w_curr = (w_f) + stride_w * (w + w_);
                                        float *curr_inp = curr_channel + row_major(h_curr, w_curr, W_in);

                                        // PTR TO INPUT POSITION
                                        float *curr_out_xy = curr_out + temp_out_img_size * (c_) + row_major((h + h_), (w + w_), W_out);

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
    }
}

static void pw_conv(float *X, float *F_1D, float *O, int B, int H_in, int W_in, int C_in, int C_out)
{
    int mat_size = W_in * H_in;
    int img_size = mat_size * C_in;
    int out_size = mat_size * C_out;

    for (int b = 0; b < B; b += BATCH_BLOCK_PW)
    {
        int B_b = min(BATCH_BLOCK_PW, B - b);
        for (int c = 0; c < C_in; c += 1)
        {
            for (int f = 0; f < C_out; f += 1)
            {
                float *f_curr = F_1D + f * C_in + c;
                for (int h = 0; h < H_in; h += HEIGHT_BLOCK_PW)
                {
                    int H_b = min(HEIGHT_BLOCK_PW, H_in - h);
                    for (int w = 0; w < W_in; w += WIDTH_BLOCK_PW)
                    {
                        int W_b = min(WIDTH_BLOCK_PW, W_in - w);

                        // Blocking
                        for (int b_ = 0; b_ < B_b; b_ += 1)
                        {
                            float *curr_img = X + (b_ + b) * img_size;
                            float *curr_out = O + (b_ + b) * out_size;

                            float *inp_curr = curr_img + mat_size * c;
                            float *o_curr = curr_out + mat_size * f;
                            for (int h_ = 0; h_ < H_b; h_ += 1)
                            {
                                for (int w_ = 0; w_ < W_b; w_ += 1)
                                {
                                    *(o_curr + row_major((h + h_), (w + w_), W_in)) += (*f_curr) * *(inp_curr + row_major((h + h_), (w + w_), W_in));
                                }
                            }
                        }
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
    WIDTH_BLOCK_PW = wbpw;
    HEIGHT_BLOCK_PW = hbpw;

    BATCH_BLOCK_DW = bbdw;
    HEIGHT_BLOCK_DW = hbdw;
    WIDTH_BLOCK_DW = wbdw;

    if (bbpw < 0 || wbpw < 0 || hbpw < 0 || bbdw < 0 || bbdw < 0 || hbdw < 0 || wbdw < 0 ) {
        exit(-1);
    }
}

void dws_conv(float *X, float *F_DW, float *F_1D, float *O, int B, int H_in, int W_in, int C_in, int H_f, int W_f, int N_dw, int H_out, int W_out, int C_out, int stride_h, int stride_w, float *depthwise_output)
{
    dw_conv(X, F_DW, depthwise_output, B, H_in, W_in, C_in, H_f, W_f, N_dw, H_out, W_out, stride_h, stride_w);
    pw_conv(depthwise_output, F_1D, O, B, H_out, W_out, C_in * N_dw, C_out);
}