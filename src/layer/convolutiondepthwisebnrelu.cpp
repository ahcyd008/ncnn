// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "convolutiondepthwisebnrelu.h"

#include "layer_type.h"
#include <math.h>

namespace ncnn {

DEFINE_LAYER_CREATOR(ConvolutionDepthWiseBNRelu)

ConvolutionDepthWiseBNRelu::ConvolutionDepthWiseBNRelu()
{
    one_blob_only = true;
    support_inplace = false;
}

ConvolutionDepthWiseBNRelu::~ConvolutionDepthWiseBNRelu()
{

}

int ConvolutionDepthWiseBNRelu::load_param(const ParamDict& pd)
{
    num_output = pd.get(0, 0);
    kernel_w = pd.get(1, 0);
    kernel_h = pd.get(11, kernel_w);
    dilation_w = pd.get(2, 1);
    dilation_h = pd.get(12, dilation_w);
    stride_w = pd.get(3, 1);
    stride_h = pd.get(13, stride_w);
    pad_w = pd.get(4, 0);
    pad_h = pd.get(14, pad_w);
    bias_term = pd.get(5, 0);
    weight_data_size = pd.get(6, 0);
    group = pd.get(7, 1);

    if (num_output % group != 0)
    {
        // reject invalid group
        return -100;
    }

    bn_channels = num_output;
    bn_eps = 0.0f;

    relu_slope = 0.0f;

    return 0;
}

int ConvolutionDepthWiseBNRelu::load_model(const ModelBin& mb)
{
    weight_data = mb.load(weight_data_size, 0);
    if (weight_data.empty())
        return -100;

    if (bias_term)
    {
        bias_data = mb.load(num_output, 1);
        if (bias_data.empty())
            return -100;
    }

    //batch noraml
    bn_slope_data = mb.load(bn_channels, 1);
    if (bn_slope_data.empty())
        return -100;

    bn_mean_data = mb.load(bn_channels, 1);
    if (bn_mean_data.empty())
        return -100;

    bn_var_data = mb.load(bn_channels, 1);
    if (bn_var_data.empty())
        return -100;

    bn_bias_data = mb.load(bn_channels, 1);
    if (bn_bias_data.empty())
        return -100;

    bn_a_data.create(bn_channels);
    if (bn_a_data.empty())
        return -100;
    bn_b_data.create(bn_channels);
    if (bn_b_data.empty())
        return -100;

    for (int i=0; i<bn_channels; i++)
    {
        float sqrt_var = sqrt(bn_var_data[i] + bn_eps);
        bn_a_data[i] = bn_bias_data[i] - bn_slope_data[i] * bn_mean_data[i] / sqrt_var;
        bn_b_data[i] = bn_slope_data[i] / sqrt_var;
    }

    //relu
    //none

    return 0;
}

int ConvolutionDepthWiseBNRelu::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // convolv with NxN kernel
    // value = value + bias

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

    if (channels % group != 0 || num_output % group != 0)
    {
        // reject invalid group
        return -100;
    }

//     fprintf(stderr, "ConvolutionDepthWise input %d x %d  pad = %d %d  ksize=%d %d  stride=%d %d\n", w, h, pad_w, pad_h, kernel_w, kernel_h, stride_w, stride_h);

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    Mat bottom_blob_unbordered = bottom_blob;

    Mat bottom_blob_bordered = bottom_blob_unbordered;
    if (pad_w > 0 || pad_h > 0)
    {
        copy_make_border(bottom_blob_unbordered, bottom_blob_bordered, pad_h, pad_h, pad_w, pad_w, BORDER_CONSTANT, 0.f, opt.workspace_allocator, opt.num_threads);
        if (bottom_blob_bordered.empty())
            return -100;

        w = bottom_blob_bordered.w;
        h = bottom_blob_bordered.h;
    }
    else if (pad_w == -233 && pad_h == -233)
    {
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;
        if (wpad > 0 || hpad > 0)
        {
            copy_make_border(bottom_blob_unbordered, bottom_blob_bordered, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, BORDER_CONSTANT, 0.f, opt.workspace_allocator, opt.num_threads);
            if (bottom_blob_bordered.empty())
                return -100;
        }

        w = bottom_blob_bordered.w;
        h = bottom_blob_bordered.h;
    }

    int outw = (w - kernel_extent_w) / stride_w + 1;
    int outh = (h - kernel_extent_h) / stride_h + 1;

    top_blob.create(outw, outh, num_output, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * dilation_h - kernel_w * dilation_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }

    // depth-wise
    if (channels == group && group == num_output)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int g=0; g<group; g++)
        {
            float* outptr = top_blob.channel(g);
            const float* kptr = (const float*)weight_data + maxk * g;
            const Mat m = bottom_blob_bordered.channel(g);
            float a = bn_a_data[g];
            float b = bn_b_data[g];

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    float sum = 0.f;

                    if (bias_term)
                        sum = bias_data[g];

                    const float* sptr = m.row(i*stride_h) + j*stride_w;

                    for (int k = 0; k < maxk; k++)
                    {
                        float val = sptr[ space_ofs[k] ];
                        float w = kptr[k];
                        sum += val * w;
                    }

                    sum = b*sum+a;
                    if (sum < 0)
                        sum = 0;

                    outptr[j] = sum;
                }

                outptr += outw;
            }
        }

        return 0;
    }

    const int channels_g = channels / group;
    const int num_output_g = num_output / group;

#ifdef _WIN32
    #pragma omp parallel for num_threads(opt.num_threads)
#else // _WIN32
    #pragma omp parallel for collapse(2) num_threads(opt.num_threads)
#endif // _WIN32
    for (int g=0; g<group; g++)
    {
        for (int p=0; p<num_output_g; p++)
        {
            float* outptr = top_blob.channel(g * num_output_g + p);
            const float* weight_data_ptr = (const float*)weight_data + maxk * channels_g * num_output_g * g;
            float a = bn_a_data[g * num_output_g + p];
            float b = bn_b_data[g * num_output_g + p];

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    float sum = 0.f;

                    if (bias_term)
                        sum = bias_data[num_output_g * g + p];

                    const float* kptr = weight_data_ptr + maxk * channels_g * p;

                    // channels_g
                    for (int q=0; q<channels_g; q++)
                    {
                        const Mat m = bottom_blob_bordered.channel(channels_g * g + q);
                        const float* sptr = m.row(i*stride_h) + j*stride_w;

                        for (int k = 0; k < maxk; k++)
                        {
                            float val = sptr[ space_ofs[k] ];
                            float w = kptr[k];
                            sum += val * w;
                        }

                        kptr += maxk;
                    }

                    sum = b*sum+a;
                    if (sum < 0)
                        sum = 0;

                    outptr[j] = sum;
                }

                outptr += outw;
            }
        }
    }

    return 0;
}

} // namespace ncnn