#include <stdio.h>
#include <stdlib.h>

#define min(a, b) (((a) < (b)) ? (a) : (b))
#define row_major(i, j, num_rows) ((i) * (num_rows) + (j))

static void dw_conv(double *X, double *F_DW, double *O, int B, int H_in, int W_in, int C_in, int H_f, int W_f, int N_dw, int H_out, int W_out, int stride_h, int stride_w)
{
    int mat_size = W_in * H_in;
    int f_size = W_f * H_f;
    int img_size = mat_size * C_in;

    int temp_out_img_size = W_out * H_out;
    int temp_out_size = temp_out_img_size * N_dw * C_in;

    for (int b = 0; b < B; b += 1)
    {
        // PTRS TO IMG IN BATCH
        double *curr_img = X + b * img_size;
        double *curr_out = O + b * temp_out_size;
        for (int c = 0; c < C_in; c += 1)
        {
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
    }
}

static void pw_conv(double *X, double *F_1D, double *O, int B, int H_in, int W_in, int C_in, int C_out)
{
    int mat_size = W_in * H_in;
    int img_size = mat_size * C_in;
    int out_size = mat_size * C_out;

    for (int b = 0; b < B; b += 1)
    {
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
    dw_conv(X, F_DW, depthwise_output, B, H_in, W_in, C_in, H_f, W_f, N_dw, H_out, W_out, stride_h, stride_w);
    pw_conv(depthwise_output, F_1D, O, B, H_out, W_out, C_in * N_dw, C_out);
}