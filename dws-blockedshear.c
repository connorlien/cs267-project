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
int HEIGHT_FILTER_BLOCK_DW_P;
int WIDTH_FILTER_BLOCK_DW_P;

static void dw_conv_blocked(double *X, double *F_DW, double *O, int B, int H_in, int W_in, int C_in, int H_f,
                            int W_f, int N_dw, int H_out, int W_out, int stride_h, int stride_w, int b_, int c_, int f_, int w_, int h_, int q6_, int q7_, int r6_, int r7_)
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

    int W_f_b = min(WIDTH_FILTER_BLOCK_DW, stride_w - r7_);
    int W_f_b_p = min(WIDTH_FILTER_BLOCK_DW_P,  W_f / stride_w - q7_);

    int H_f_b = min(HEIGHT_FILTER_BLOCK_DW, stride_h - r6_);
    int H_f_b_p = min(HEIGHT_FILTER_BLOCK_DW_P,  H_f / stride_h - q6_);

    // for (int r6 = 0; r6 < stride_h; r6 += 1)
    //                     {
    //                         for (int q6 = 0; q6 < H_f / stride_h; q6 += 1)
    //                         {
    //                             for (int r7 = 0; r7 < stride_w; r7 += 1)
    //                             {
    //                                 for (int q7 = 0; q7 < W_f / stride_w; q7 += 1)
    //                                 {

    for (int f = 0; f < F_b; f += 1)
    {
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

                for (int h = 0; h < H_b; h += 1)
                {
                    for (int w = 0; w < W_b; w += 1)
                    {
                        for (int r6 = 0; r6 < H_f_b; r6 += 1)
                        {
                            for (int q6 = 0; q6 < H_f_b_p; q6 += 1)
                            {
                                for (int r7 = 0; r7 < W_f_b; r7 += 1)
                                {
                                    for (int q7 = 0; q7 < W_f_b_p; q7 += 1)
                                    {
                                        int h_f = stride_h * (q6 + q6_) + (r6 + r6_);
                                        int w_f = stride_w * (q7 + q7_) + (r7 + r7_);

                                        // PTR TO CURRENT POSITION IN FILTER
                                        double *f_curr = F_DW + f_size * ((c + c_) * N_dw + (f + f_)) + row_major(h_f, w_f, W_f);

                                        // PTR TO INPUT POSITION
                                        int h_curr = h_f + stride_h * (h + h_);
                                        int w_curr = w_f  + stride_w * (w + w_);
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
    }
}

static void dw_conv(double *X, double *F_DW, double *O, int B, int H_in, int W_in, int C_in, int H_f, int W_f, int N_dw, int H_out, int W_out, int stride_h, int stride_w)
{
    for (int f = 0; f < N_dw; f += FILTER_DW)
    {
        for (int b = 0; b < B; b += BATCH_BLOCK_DW)
        {
            for (int c = 0; c < C_in; c += CHANNEL_BLOCK_DW)
            {
                for (int h = 0; h < H_out; h += HEIGHT_BLOCK_DW)
                {
                    for (int w = 0; w < W_out; w += WIDTH_BLOCK_DW)
                    {
                        for (int r6 = 0; r6 < stride_h; r6 += HEIGHT_FILTER_BLOCK_DW)
                        {
                            for (int q6 = 0; q6 < H_f / stride_h; q6 += HEIGHT_FILTER_BLOCK_DW_P)
                            {
                                for (int r7 = 0; r7 < stride_w; r7 += WIDTH_FILTER_BLOCK_DW)
                                {
                                    for (int q7 = 0; q7 < W_f / stride_w; q7 += WIDTH_FILTER_BLOCK_DW_P)
                                    {
                                        dw_conv_blocked(X, F_DW, O, B, H_in, W_in, C_in, H_f, W_f, N_dw, H_out, W_out, stride_h, stride_w, b, c, f, w, h, q6, q7, r6, r7);
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

void pw_blocked(int B, int H_in, int W_in, int C_in, int C_out, int B_b, int F_b, int W_b, int H_b, int C_b, int b_, int f_, int w_, int h_, int c_, double *F_1D, double *O, double *X)
{
    int mat_size = W_in * H_in;
    int img_size = mat_size * C_in;
    int out_size = mat_size * C_out;

    for (int b = 0; b < B_b; b += 1)
    {
        double *curr_img = X + (b_ + b) * img_size;
        double *curr_out = O + (b_ + b) * out_size;

        for (int c = 0; c < C_b; c += 1)
        {
            for (int f = 0; f < F_b; f += 1)
            {
                double *f_curr = F_1D + (f + f_) * C_in + (c + c_);
                for (int h = 0; h < H_b; h += 1)
                {
                    for (int w = 0; w < W_b; w += 1)
                    {
                        double *o_curr = curr_out + mat_size * (f + f_) + row_major((h + h_), (w + w_), W_in);
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
        for (int c = 0; c < C_in; c += CHANNEL_BLOCK_PW)
        {
            int C_b = min(CHANNEL_BLOCK_PW, C_in - c);
            for (int f = 0; f < C_out; f += FILTER_BLOCK_PW)
            {
                int F_b = min(FILTER_BLOCK_PW, C_out - f);
                for (int h = 0; h < H_in; h += HEIGHT_BLOCK_PW)
                {
                    int H_b = min(HEIGHT_BLOCK_PW, H_in - h);
                    for (int w = 0; w < W_in; w += WIDTH_BLOCK_PW)
                    {
                        int W_b = min(WIDTH_BLOCK_PW, W_in - w);
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

void init_conv(int bbpw, int fbpw, int wbpw, int hbpw, int cbpw, int bbdw, int cbdw, int fdw, int hbdw, int wbdw, int hfdw, int wfbdw, int hfdwp, int wfdwp)
{
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
    HEIGHT_FILTER_BLOCK_DW_P = hfdwp;
    WIDTH_FILTER_BLOCK_DW_P = wfdwp;
}

void dws_conv(double *X, double *F_DW, double *F_1D, double *O, int B, int H_in, int W_in, int C_in, int H_f, int W_f, int N_dw, int H_out, int W_out, int C_out, int stride_h, int stride_w, double *depthwise_output)
{
    dw_conv(X, F_DW, depthwise_output, B, H_in, W_in, C_in, H_f, W_f, N_dw, H_out, W_out, stride_h, stride_w);
    pw_conv(depthwise_output, F_1D, O, B, H_out, W_out, C_in * N_dw, C_out);
}