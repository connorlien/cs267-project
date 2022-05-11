#include <stdio.h>
#include <stdlib.h>

#define min(a, b) (((a) < (b)) ? (a) : (b))
#define row_major(i, j, num_rows) ((i) * (num_rows) + (j))

static void single_op(float *X, float *F_DW, float *F_1D, float *O, int B, int H_in, int W_in, int C_in, int H_f, int W_f, int N_dw, int H_out, int W_out, int C_out, int stride_h, int stride_w)
{
    int mat_size = W_in * H_in;
    int f_size = W_f * H_f;
    int img_size = mat_size * C_in;

    int mat_size_out = W_out * H_out;
    int out_size = mat_size_out * C_out;

    for (int b = 0; b < B; b += 1)
    {
        // PTRS TO IMG IN BATCH
        float *curr_img = X + b * img_size;
        float *curr_out = O + b * out_size;
        for (int c = 0; c < C_in; c += 1)
        {
            for (int f = 0; f < N_dw; f += 1)
            {
                for (int h = 0; h < H_out; h += 1)
                {
                    for (int w = 0; w < W_out; w += 1)
                    {
                        // Do 2D Convolution channelwise
                        float *curr_channel = curr_img + mat_size * c;
                        // MOST LIKELY SET TO 1.
                        // MICROKERNEL - tile if needed.
                        
                        float temp = 0.0;
                        // Depthwise conv
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
                                temp += *f_curr * *curr_inp;
                            }
                        }

                        // LOOP 
                        for (int f_1d = 0; f_1d < C_out; f_1d += 1) {
                            float *o_curr = curr_out + mat_size_out * f_1d + row_major(h, w, W_out);
                            float *curr = F_1D + f_1d * (N_dw * C_in) + N_dw * c + f;
                            *o_curr += temp * (*curr);
                        }
                    }
                }
            }
        }
    }
}

void init_conv(int bbpw, int fbpw, int wbpw, int hbpw, int cbpw, int bbdw, int cbdw, int fdw, int hbdw, int wbdw, int hfdw, int wfbdw) {
}

void dws_conv(float *X, float *F_DW, float *F_1D, float *O, int B, int H_in, int W_in, int C_in, int H_f, int W_f, int N_dw, int H_out, int W_out, int C_out, int stride_h, int stride_w, float* depthwise_output)
{
    single_op(X, F_DW, F_1D, O, B, H_in, W_in, C_in, H_f, W_f, N_dw, H_out, W_out, C_out, stride_h, stride_w);
}