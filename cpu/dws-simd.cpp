#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <string.h>

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

int VEC_SIZE = 16;

// assume that C_in alwas <= 16
void dws_microkernel(float *X, __m512 filt[16], float *O, int B, int H_in, int W_in, int C_in, int H_f, int W_f, int N_dw, int H_out, int W_out, int curr_h, int curr_w)
{
    int mat_size = W_in * H_in;
    int temp_out_img_size = W_out * H_out;
    
    __m512 inp;
    __m512 out;

    for (int i = 0; i < H_f; i++)
    {
        for (int j = 0; j < W_f; j++)
        {
            float* curr_inp = X + row_major((curr_h + i), (curr_w + j), W_in);

            inp = _mm512_set_ps(15 < C_in ? *(curr_inp + mat_size * (15)) : 0.0,
                                14 < C_in ? *(curr_inp + mat_size * (14)) : 0.0,
                                13 < C_in ? *(curr_inp + mat_size * (13)) : 0.0,
                                12 < C_in ? *(curr_inp + mat_size * (12)) : 0.0,
                                11 < C_in ? *(curr_inp + mat_size * (11)) : 0.0,
                                10 < C_in ? *(curr_inp + mat_size * (10)) : 0.0,
                                9 < C_in ? *(curr_inp + mat_size * (9)) : 0.0,
                                8 < C_in ? *(curr_inp + mat_size * (8)) : 0.0,
                                7 < C_in ? *(curr_inp + mat_size * (7)) : 0.0,
                                6 < C_in ? *(curr_inp + mat_size * (6)) : 0.0,
                                5 < C_in ? *(curr_inp + mat_size * (5)) : 0.0,
                                4 < C_in ? *(curr_inp + mat_size * (4)) : 0.0,
                                3 < C_in ? *(curr_inp + mat_size * (3)) : 0.0,
                                2 < C_in ? *(curr_inp + mat_size * (2)) : 0.0,
                                1 < C_in ? *(curr_inp + mat_size * (1)) : 0.0,
                                0 < C_in ? *(curr_inp + mat_size * (0)) : 0.0);

            out = _mm512_set_ps(15 < C_in ? *(O + temp_out_img_size * (15)) : 0.0,
                                14 < C_in ? *(O + temp_out_img_size * (14)) : 0.0,
                                13 < C_in ? *(O + temp_out_img_size * (13)) : 0.0,
                                12 < C_in ? *(O + temp_out_img_size * (12)) : 0.0,
                                11 < C_in ? *(O + temp_out_img_size * (11)) : 0.0,
                                10 < C_in ? *(O + temp_out_img_size * (10)) : 0.0,
                                9 < C_in ? *(O + temp_out_img_size * (9)) : 0.0,
                                8 < C_in ? *(O + temp_out_img_size * (8)) : 0.0,
                                7 < C_in ? *(O + temp_out_img_size * (7)) : 0.0,
                                6 < C_in ? *(O + temp_out_img_size * (6)) : 0.0,
                                5 < C_in ? *(O + temp_out_img_size * (5)) : 0.0,
                                4 < C_in ? *(O + temp_out_img_size * (4)) : 0.0,
                                3 < C_in ? *(O + temp_out_img_size * (3)) : 0.0,
                                2 < C_in ? *(O + temp_out_img_size * (2)) : 0.0,
                                1 < C_in ? *(O + temp_out_img_size * (1)) : 0.0,
                                0 < C_in ? *(O + temp_out_img_size * (0)) : 0.0);


            out = _mm512_fmadd_ps(inp, filt[i * 3 + j] ,out);

            // Store back to out
            for (int c = 0; c < C_in; c++)
            {
                *(O + temp_out_img_size * c) = out[c];
            }
        }
    }
}

static void dw_conv(float *X, float *F_DW, float *O, int B, int H_in, int W_in, int C_in, int H_f, int W_f, int N_dw, int H_out, int W_out, int stride_h, int stride_w)
{
    int mat_size = W_in * H_in;
    int f_size = W_f * H_f;
    int img_size = mat_size * C_in;

    int temp_out_img_size = W_out * H_out;
    int temp_out_size = temp_out_img_size * N_dw * C_in;

    __m512 filters[9];
    for (int i = 0; i < H_f; i++)
    {
        for (int j = 0; j < W_f; j++)
        {
            float *curr_filter = F_DW + row_major(i, j, W_f);

            filters[i * 3 + j] = _mm512_set_ps (15 < C_in ? *(curr_filter + f_size * (15)) : 0.0,
                                14 < C_in ? *(curr_filter + f_size * (14)) : 0.0,
                                13 < C_in ? *(curr_filter + f_size * (13)) : 0.0,
                                12 < C_in ? *(curr_filter + f_size * (12)) : 0.0,
                                11 < C_in ? *(curr_filter + f_size * (11)) : 0.0,
                                10 < C_in ? *(curr_filter + f_size * (10)) : 0.0,
                                9 < C_in ? *(curr_filter + f_size * (9)) : 0.0,
                                8 < C_in ? *(curr_filter + f_size * (8)) : 0.0,
                                7 < C_in ? *(curr_filter + f_size * (7)) : 0.0,
                                6 < C_in ? *(curr_filter + f_size * (6)) : 0.0,
                                5 < C_in ? *(curr_filter + f_size * (5)) : 0.0,
                                4 < C_in ? *(curr_filter + f_size * (4)) : 0.0,
                                3 < C_in ? *(curr_filter + f_size * (3)) : 0.0,
                                2 < C_in ? *(curr_filter + f_size * (2)) : 0.0,
                                1 < C_in ? *(curr_filter + f_size * (1)) : 0.0,
                                0 < C_in ? *(curr_filter + f_size * (0)) : 0.0);
        }
    }

    for (int f = 0; f < N_dw; f += 1)
    {
        for (int b = 0; b < B; b += 1)
        {
            // PTRS TO IMG IN BATCH
            float *curr_img = X + b * img_size;
            float *curr_out = O + b * temp_out_size;
            // Do 2D Convolution channelwise
            // Filters are 2D
            for (int h = 0; h < H_out; h += 1)
            {
                for (int w = 0; w < W_out; w += 1)
                {

                    dws_microkernel(curr_img, filters, curr_out + row_major(h, w, W_out), B, H_in, W_in, C_in, H_f, W_f, N_dw, H_out, W_out, stride_h * h, stride_w * w);
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

static void pw_conv(float *X, float *F_1D, float *O, int B, int H_in, int W_in, int C_in, int C_out)
{
    int mat_size = W_in * H_in;
    int img_size = mat_size * C_in;
    int out_size = mat_size * C_out;

    for (int b = 0; b < B; b += BATCH_BLOCK_PW)
    {
        float *curr_img = X + (b)*img_size;
        float *curr_out = O + (b)*out_size;
        int f = 0;

        for (; f < C_out / 16 * 16; f += 16)
        {
            for (int h = 0; h < H_in; h += 1)
            {
                for (int w = 0; w < W_in; w += 1)
                {
                    float *o_curr = curr_out + mat_size * f + row_major(h, w, W_in);
                    int c = 0;
                    for (; c < C_in / 16 * 16; c += 16)
                    {
                        float *inp_curr = curr_img + mat_size * c + row_major(h, w, W_in);
                        float *f_curr = F_1D + f * C_in + c;
                        pw_microkernel_1x1x16x16(inp_curr, f_curr, o_curr, mat_size, C_in);
                    }
                    for (; c < C_in; c += 1)
                    {
                        float *f_curr = F_1D + f * C_in + c;
                        float *inp_curr = curr_img + mat_size * c + row_major(h, w, W_in);
                        *o_curr += (*f_curr) * (*inp_curr);
                    }
                }
            }
        }
        for (; f < C_out; f += 1)
        {
            for (int h = 0; h < H_in; h += 1)
            {
                for (int w = 0; w < W_in; w += 1)
                {
                    float *o_curr = curr_out + mat_size * f + row_major(h, w, W_in);
                    int c = 0;
                    for (; c < C_in / 16 * 16; c += 16)
                    {
                        float *inp_curr = curr_img + mat_size * c + row_major(h, w, W_in);
                        float *f_curr = F_1D + f * C_in + c;
                        pw_microkernel_1x1x16x1(inp_curr, f_curr, o_curr, mat_size);
                    }
                    for (; c < C_in; c += 1)
                    {
                        float *f_curr = F_1D + f * C_in + c;
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

    float *X_aligned = X;                               //(float *)_mm_malloc(inp_size * sizeof(float), 64);
    float *F_DW_aligned = F_DW;                         // (float *)_mm_malloc(f_dw_size * sizeof(float), 64);
    float *F_1D_aligned = F_1D;                         // (float *) _mm_malloc(f_1d_size * sizeof(float), 64);
    float *O_aligned = O;                               // (float *)_mm_malloc(out_size * sizeof(float), 64);
    float *depthwise_output_aligned = depthwise_output; //(float *)_mm_malloc(B * W_out * H_out * C_in * N_dw * sizeof(float), 64);

    // memcpy(X_aligned, X, inp_size * sizeof(float));
    // memcpy(F_DW_aligned, F_DW, f_dw_size * sizeof(float));
    // memcpy(F_1D_aligned, F_1D, f_1d_size * sizeof(float));
    // memcpy(O_aligned, O, out_size * sizeof(float));

    dw_conv(X_aligned, F_DW_aligned, depthwise_output_aligned, B, H_in, W_in, C_in, H_f, W_f, N_dw, H_out, W_out, stride_h, stride_w);
    pw_conv(depthwise_output_aligned, F_1D_aligned, O_aligned, B, H_out, W_out, C_in * N_dw, C_out);
    // memcpy(O, O_aligned, out_size * sizeof(float));

    // _mm_free(X_aligned);
    // _mm_free(F_DW_aligned);
    // _mm_free(F_1D_aligned);
    // _mm_free(O_aligned);
}