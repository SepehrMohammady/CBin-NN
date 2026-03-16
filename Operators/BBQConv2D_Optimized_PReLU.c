#include "CBin-NN.h"

void BBQConv2D_Optimized_PReLU(float out[],
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
                               float shift[],
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

    for (i = 0; i < out_ch_pad>>5; i++)
    {
        for (j = 0; j < out_dim; j++)
        {
            for (k = 0; k < out_dim; k++)
            {
                /* Array to hold intermediate popcounts for unrolled loop */
                int pop_counts[32] = {0}; 
                
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
                                input_idx = (in_row * in_dim + in_col) * (in_ch_pad>>5) + l;
                                int in_val = input[input_idx];
                                
                                /* Unroll output channels calculation */
                                #pragma GCC unroll 32
                                for(int u = 0; u < 32; u++) {
                                    weight_idx = ((i * 32 + u) * in_ch * ker_size * ker_size) + (m * ker_size + n) * (in_ch_pad>>5) + l;
                                    sum = __builtin_popcount(weight[weight_idx] ^ in_val);
                                    pop_counts[u] += N - (sum<<1);
                                }
                            }
                        }
                    }
                }
                
                /* Process outputs and applying BN and PReLU for the unrolled block */
                for(int u = 0; u < 32; u++) {
                    int real_out_ch = i * 32 + u;
                    if(real_out_ch < out_ch) {
                        /* Batch Normalization Fusion */
                        conv_out = pop_counts[u] + bn_wt[real_out_ch];
                        
                        /* PReLU Activation */
                        if (conv_out < 0) {
                            conv_out = conv_out * shift[real_out_ch];
                        }

                        /* Output as float */
                        out_idx = real_out_ch + (j * out_dim + k) * out_ch;
                        out[out_idx] = conv_out;
                    }
                }
            }
        }
    }
}
