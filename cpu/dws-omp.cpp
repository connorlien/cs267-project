#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <string.h>
#include <omp.h>

#define min(a, b) (((a) < (b)) ? (a) : (b))
#define row_major(i, j, num_rows) ((i) * (num_rows) + (j))
#define pad(B) (((B) % 8 != 0 ? (B) + 8 - ((B) % 8) : (B)))
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

void dws_microkernel(float *input, float *filter, float *output, int mat_size)
{
    /*
    int A0, A1, A2, A3, A4, A5, A6, A7;
    int B0, B1, B2, B3, B4, B5, B6, B7;
    int C0, C1, C2, C3, C4, C5, C6, C7;
    */
}

static void dw_conv_blocked(float *X, float *F_DW, float *O, int B, int H_in, int W_in, int C_in, int H_f, int W_f, int N_dw, int H_out, int W_out, int stride_h, int stride_w, int b_, int c_, int f_, int w_, int h_, int w_f_, int h_f_)
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

    for (int f = 0; f < F_b; f += 1)
    {
        for (int b = 0; b < B_b; b += 1)
        {
            // PTRS TO IMG IN BATCH
            float *curr_img = X + (b + b_) * img_size;
            float *curr_out = O + (b + b_) * temp_out_size;
            for (int c = 0; c < C_b; c += 1)
            {
                float *curr_channel = curr_img + mat_size * (c + c_);
            
                for (int h = 0; h < H_b; h += 1)
                {
                    for (int w = 0; w < W_b; w += 1)
                    {
                        // PTR TO OUTPUT POSITION
                        float *curr_out_xy = curr_out + temp_out_img_size * ((c + c_) * N_dw + (f_ + f)) + row_major((h + h_), (w + w_), W_out);

                        for (int h_f = 0; h_f < H_f_b; h_f += 1)
                        {
                            for (int w_f = 0; w_f < W_f_b; w_f += 1)
                            {
                                // PTR TO CURRENT POSITION IN FILTER
                                float *f_curr = F_DW + f_size * ((c + c_) * N_dw + (f + f_)) + row_major((h_f + h_f_), (w_f + w_f_), W_f);

                                // PTR TO INPUT POSITION
                                int h_curr = (h_f + h_f_) + stride_h * (h + h_);
                                int w_curr = (w_f + w_f_) + stride_w * (w + w_);
                                float *curr_inp = curr_channel + row_major(h_curr, w_curr, W_in);

                                // CONVOLVE
                                float result = *f_curr * *curr_inp;

                                *curr_out_xy += result;
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
    #pragma omp parallel for collapse(5)
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
                        for (int h_f = 0; h_f < H_f; h_f += HEIGHT_FILTER_BLOCK_DW)
                        {
                            for (int w_f = 0; w_f < W_f; w_f += WIDTH_FILTER_BLOCK_DW)
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

void pw_microkernel_1x1x16x1(float *input, float *filter, float *output, int mat_size)
{
    // Declare
    __m512 a, b, c;

    // Load
    a = _mm512_set_ps(*(input + mat_size * (15)),
                      *(input + mat_size * (14)),
                      *(input + mat_size * (13)),
                      *(input + mat_size * (12)),
                      *(input + mat_size * (11)),
                      *(input + mat_size * (10)),
                      *(input + mat_size * (9)),
                      *(input + mat_size * (8)),
                      *(input + mat_size * (7)),
                      *(input + mat_size * (6)),
                      *(input + mat_size * (5)),
                      *(input + mat_size * (4)),
                      *(input + mat_size * (3)),
                      *(input + mat_size * (2)),
                      *(input + mat_size * (1)),
                      *(input + mat_size * (0)));
                      
    b = _mm512_loadu_ps(filter);

    c = _mm512_mul_ps(a, b);

    // Store
    *output += _mm512_reduce_add_ps(c);
}

void pw_microkernel_1x1x16x16(float *input, float *filter, float *output, int mat_size, int C_in)
{
    // Declare
    __m512 a;
    __m512 b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15;
    __m512 c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15;

    // Load
    a = _mm512_set_ps(*(input + mat_size * (15)),
                      *(input + mat_size * (14)),
                      *(input + mat_size * (13)),
                      *(input + mat_size * (12)),
                      *(input + mat_size * (11)),
                      *(input + mat_size * (10)),
                      *(input + mat_size * (9)),
                      *(input + mat_size * (8)),
                      *(input + mat_size * (7)),
                      *(input + mat_size * (6)),
                      *(input + mat_size * (5)),
                      *(input + mat_size * (4)),
                      *(input + mat_size * (3)),
                      *(input + mat_size * (2)),
                      *(input + mat_size * (1)),
                      *(input + mat_size * (0)));

    b0 = _mm512_loadu_ps(filter);
    b1 = _mm512_loadu_ps(filter + C_in * 1);
    b2 = _mm512_loadu_ps(filter + C_in * 2);
    b3 = _mm512_loadu_ps(filter + C_in * 3);
    b4 = _mm512_loadu_ps(filter + C_in * 4);
    b5 = _mm512_loadu_ps(filter + C_in * 5);
    b6 = _mm512_loadu_ps(filter + C_in * 6);
    b7 = _mm512_loadu_ps(filter + C_in * 7);
    b8 = _mm512_loadu_ps(filter + C_in * 8);
    b9 = _mm512_loadu_ps(filter + C_in * 9);
    b10 = _mm512_loadu_ps(filter + C_in * 10);
    b11 = _mm512_loadu_ps(filter + C_in * 11);
    b12 = _mm512_loadu_ps(filter + C_in * 12);
    b13 = _mm512_loadu_ps(filter + C_in * 13);
    b14 = _mm512_loadu_ps(filter + C_in * 14);
    b15 = _mm512_loadu_ps(filter + C_in * 15);

    c0 = _mm512_mul_ps(a, b0);
    c1 = _mm512_mul_ps(a, b1);
    c2 = _mm512_mul_ps(a, b2);
    c3 = _mm512_mul_ps(a, b3);
    c4 = _mm512_mul_ps(a, b4);
    c5 = _mm512_mul_ps(a, b5);
    c6 = _mm512_mul_ps(a, b6);
    c7 = _mm512_mul_ps(a, b7);
    c8 = _mm512_mul_ps(a, b8);
    c9 = _mm512_mul_ps(a, b9);
    c10 = _mm512_mul_ps(a, b10);
    c11 = _mm512_mul_ps(a, b11);
    c12 = _mm512_mul_ps(a, b12);
    c13 = _mm512_mul_ps(a, b13);
    c14 = _mm512_mul_ps(a, b14);
    c15 = _mm512_mul_ps(a, b15);

    // Store
    *output += _mm512_reduce_add_ps(c0);
    *(output + C_in * 1) += _mm512_reduce_add_ps(c1);
    *(output + C_in * 2) += _mm512_reduce_add_ps(c2);
    *(output + C_in * 3) += _mm512_reduce_add_ps(c3);
    *(output + C_in * 4) += _mm512_reduce_add_ps(c4);
    *(output + C_in * 5) += _mm512_reduce_add_ps(c5);
    *(output + C_in * 6) += _mm512_reduce_add_ps(c6);
    *(output + C_in * 7) += _mm512_reduce_add_ps(c7);
    *(output + C_in * 8) += _mm512_reduce_add_ps(c8);
    *(output + C_in * 9) += _mm512_reduce_add_ps(c9);
    *(output + C_in * 10) += _mm512_reduce_add_ps(c10);
    *(output + C_in * 11) += _mm512_reduce_add_ps(c11);
    *(output + C_in * 12) += _mm512_reduce_add_ps(c12);
    *(output + C_in * 13) += _mm512_reduce_add_ps(c13);
    *(output + C_in * 14) += _mm512_reduce_add_ps(c14);
    *(output + C_in * 15) += _mm512_reduce_add_ps(c15);
}

void pw_blocked(int B, int H_in, int W_in, int C_in, int C_out, int B_b, int F_b, int W_b, int H_b, int C_b, int b_, int f_, int w_, int h_, int c_, float *F_1D, float *O, float *X)
{
    int mat_size = W_in * H_in;
    int img_size = mat_size * C_in;
    int out_size = mat_size * C_out;

    for (int b = 0; b < B_b; b += 1)
    {
        float *curr_img = X + (b_ + b) * img_size;
        float *curr_out = O + (b_ + b) * out_size;
        int f = 0;
        for (; f < F_b / 16 * 16; f += 16)
        {
            for (int h = 0; h < H_b; h += 1)
            {
                for (int w = 0; w < W_b; w += 1)
                {
                    float *o_curr = curr_out + mat_size * (f + f_) + row_major((h + h_), (w + w_), W_in);
                    int c = 0;
                    for (; c < C_b / 16 * 16; c += 16)
                    {
                        float *inp_curr = curr_img + mat_size * (c + c_) + row_major((h + h_), (w + w_), W_in);
                        float *f_curr = F_1D + (f + f_) * C_in + (c + c_);
                        pw_microkernel_1x1x16x16(inp_curr, f_curr, o_curr, mat_size, C_in);
                    }
                    for (; c < C_b; c += 1)
                    {
                        float *f_curr = F_1D + (f + f_) * C_in + (c + c_);
                        float *inp_curr = curr_img + mat_size * (c + c_) + row_major((h + h_), (w + w_), W_in);
                        *o_curr += (*f_curr) * (*inp_curr);
                    }
                }
            }
        }
        for (; f < F_b; f += 1)
        {
            for (int h = 0; h < H_b; h += 1)
            {
                for (int w = 0; w < W_b; w += 1)
                {
                    float *o_curr = curr_out + mat_size * (f + f_) + row_major((h + h_), (w + w_), W_in);
                    int c = 0;
                    for (; c < C_b / 16 * 16; c += 16)
                    {
                        float *inp_curr = curr_img + mat_size * (c + c_) + row_major((h + h_), (w + w_), W_in);
                        float *f_curr = F_1D + (f + f_) * C_in + (c + c_);
                        pw_microkernel_1x1x16x1(inp_curr, f_curr, o_curr, mat_size);
                    }
                    for (; c < C_b; c += 1)
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
    #pragma omp parallel for collapse(4)
    for (int b = 0; b < B; b += BATCH_BLOCK_PW)
    {
        for (int f = 0; f < C_out; f += FILTER_BLOCK_PW)
        {
            for (int h = 0; h < H_in; h += HEIGHT_BLOCK_PW)
            {
                for (int w = 0; w < W_in; w += WIDTH_BLOCK_PW)                
                {
                    for (int c = 0; c < C_in; c += CHANNEL_BLOCK_PW)
                    {
                        int H_b = min(HEIGHT_BLOCK_PW, H_in - h);
                        int F_b = min(FILTER_BLOCK_PW, C_out - f);
                        int W_b = min(WIDTH_BLOCK_PW, W_in - w);
                        int B_b = min(BATCH_BLOCK_PW, B - b);
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

void init_conv(int bbpw, int fbpw, int wbpw, int hbpw, int cbpw, int bbdw, int cbdw, int fdw, int hbdw, int wbdw, int hfdw, int wfbdw)
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

    VEC_SIZE = AVX_BITS / sizeof(float);
}

void dws_conv(float *X, float *F_DW, float *F_1D, float *O, int B, int H_in, int W_in, int C_in, int H_f, int W_f, int N_dw, int H_out, int W_out, int C_out, int stride_h, int stride_w, float *depthwise_output)
{
    dw_conv(X, F_DW, depthwise_output, B, H_in, W_in, C_in, H_f, W_f, N_dw, H_out, W_out, stride_h, stride_w);
    pw_conv(depthwise_output, F_1D, O, B, H_out, W_out, C_in * N_dw, C_out);
}