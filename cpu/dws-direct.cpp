#include <stdio.h>
#include <stdlib.h>

#define min(a, b) (((a) < (b)) ? (a) : (b))
#define row_major(i, j, num_rows) ((i) * (num_rows) + (j))

static void dw_conv(float *X, float *F_DW, float *O, int B, int H_in, int W_in, int C_in, int H_f, int W_f, int N_dw, int H_out, int W_out, int stride_h, int stride_w)
{
    int mat_size = W_in * H_in;
    int f_size = W_f * H_f;
    int img_size = mat_size * C_in;

    int temp_out_img_size = W_out * H_out;
    int temp_out_size = temp_out_img_size * N_dw * C_in;

    for (int f = 0; f < N_dw; f += 1)
    {
        for (int b = 0; b < B; b += 1)
        {
            // PTRS TO IMG IN BATCH
            float *curr_img = X + b * img_size;
            float *curr_out = O + b * temp_out_size;
            // Do 2D Convolution channelwise
            // Filters are 2D
            for (int c = 0; c < C_in; c += 1)
            {
                for (int h = 0; h < H_out; h += 1)
                {
                    for (int w = 0; w < W_out; w += 1)
                    {
                        // MICROKERNEL - tile if needed.
                        for (int h_f = 0; h_f < H_f; h_f += 1)
                        {
                            for (int w_f = 0; w_f < W_f; w_f += 1)
                            {
                                float *curr_channel = curr_img + mat_size * c;

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
    }
}

static void pw_conv(float *X, float *F_1D, float *O, int B, int H_in, int W_in, int C_in, int C_out)
{
    int mat_size = W_in * H_in;
    int img_size = mat_size * C_in;
    int out_size = mat_size * C_out;

    for (int b = 0; b < B; b += 1)
    {
        float *curr_img = X + b * img_size;
        float *curr_out = O + b * out_size;
        for (int c = 0; c < C_in; c += 1)
        {
            for (int f = 0; f < C_out; f += 1)
            {
                float *f_curr = F_1D + f * C_in + c;
                for (int h = 0; h < H_in; h += 1)
                {
                    for (int w = 0; w < W_in; w += 1)
                    {
                        float *o_curr = curr_out + mat_size * f + row_major(h, w, W_in);
                        float *inp_curr = curr_img + mat_size * c + row_major(h, w, W_in);
                        *o_curr += (*f_curr) * (*inp_curr);
                    }
                }
            }
        }
    }
}

void init_conv(int bbpw, int fbpw, int wbpw, int hbpw, int cbpw, int bbdw, int cbdw, int fdw, int hbdw, int wbdw, int hfdw, int wfbdw)
{
}

void dws_conv(float *X, float *F_DW, float *F_1D, float *O, int B, int H_in, int W_in, int C_in, int H_f, int W_f, int N_dw, int H_out, int W_out, int C_out, int stride_h, int stride_w, float *depthwise_output)
{
    dw_conv(X, F_DW, depthwise_output, B, H_in, W_in, C_in, H_f, W_f, N_dw, H_out, W_out, stride_h, stride_w);
    pw_conv(depthwise_output, F_1D, O, B, H_out, W_out, C_in * N_dw, C_out);
}