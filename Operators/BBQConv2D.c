#include "CBin-NN.h"

void BBQConv2D(float out[],
               int out_ch,
               int out_dim,
               int in_ch,
               int in_dim,
               int ker_size,
               int stride,
               int padding,
               int bias[],
               int weight[],
               float bn_wt[],
               int input[])
{
    int i, j, k, m, n, l, N, in_row, in_col, input_idx, weight_idx, out_idx, in_ch_pad, out_ch_pad;
    int sum = 0, pop_count = 0;
    float conv_out;

    /* Channel padding to a mulitple of 32 */
    if(out_ch % 32 != 0)
    {
        out_ch_pad = ceil((float)(out_ch)/32) * 32;
    }
    else
    {
        out_ch_pad = out_ch;
    }

    if (in_ch % 32 != 0)
    {
        in_ch_pad = ceil((float)(in_ch)/32) * 32;
        if (in_ch < 32) N = in_ch; else N = 32;
    }
    else
    {
        in_ch_pad = in_ch;
        N = 32;
    }

    for (i = 0; i < out_ch; i++)
    {
        for (j = 0; j < out_dim; j++)
        {
            for (k = 0; k < out_dim; k++)
            {
                for (m = 0; m < ker_size; m++)
                {
                    for (n = 0; n < ker_size; n++)
                    {
                        /* Check padding */
                        in_row = stride * j + m - padding;
                        in_col = stride * k + n - padding;
                        if (in_row >= 0 && in_col >= 0 && in_row < in_dim && in_col < in_dim)
                        {
                            for (l = 0; l < in_ch_pad>>5; l++)
                            {
                                input_idx = (in_row * in_dim + in_col) * in_ch_pad>>5 + l;
                                weight_idx = i * in_ch * ker_size * ker_size + (m * ker_size + n) * in_ch_pad>>5 + l;
                                sum = __builtin_popcount(weight[weight_idx]^input[input_idx]);
                                pop_count += N - (sum<<1);
                            }
                        }
                    }
                }
                /* Bias is usually not used in BNNs, however, in case it is used uncomment the next line of code */
                // pop_count += bias[i];
                /* Batch Normalization Fusion or direct output addition */
                conv_out = pop_count + bn_wt[i];
                
                /* For BBQ, we do not pack to bit array, we output float directly */
                out_idx = (i + (j * out_dim + k) * out_ch);
                out[out_idx] = conv_out;
                pop_count = 0;
            }
        }
    }
}
