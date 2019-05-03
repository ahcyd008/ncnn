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

#ifndef LAYER_CONVOLUTIONBNRELU_H
#define LAYER_CONVOLUTIONBNRELU_H

#include "layer.h"

namespace ncnn {

    //not support int8 & quantize, only float32

class ConvolutionBNRelu : public Layer
{
public:
    ConvolutionBNRelu();
    ~ConvolutionBNRelu();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

public:
    // param
    int num_output;
    int kernel_w;
    int kernel_h;
    int dilation_w;
    int dilation_h;
    int stride_w;
    int stride_h;
    int pad_w;
    int pad_h;
    int bias_term;

    int weight_data_size;

    // model
    Mat weight_data;
    Mat bias_data;


    //batch normal
    int bn_channels;
    float bn_eps;

    Mat bn_slope_data;
    Mat bn_mean_data;
    Mat bn_var_data;
    Mat bn_bias_data;

    Mat bn_a_data;
    Mat bn_b_data;

    //relu
    float relu_slope;

};

} // namespace ncnn

#endif // LAYER_CONVOLUTIONBNRELU_H
