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

#include <stdio.h>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <time.h>

#include "net.h"
using namespace std;
using namespace cv;

static int detect_hed(ncnn::Net& hed, const cv::Mat& rgb, string outPath)
{
    // for(int i=0; i<10; i++){
    //     for (int j=0; j<10; j++)
    //     {
    //         //out[i*out.w+j] *= 255.0f;
    //         printf("[%d %d %d]", rgb.data[3*(i*rgb.cols+j)], rgb.data[3*(i*rgb.cols+j)+1], rgb.data[3*(i*rgb.cols+j)+2]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");


    ncnn::Mat in = ncnn::Mat::from_pixels(rgb.data, ncnn::Mat::PIXEL_RGB, rgb.cols, rgb.rows);

    const float mean_vals[3] = {123.f, 117.f, 104.f};//rgb
    in.substract_mean_normalize(mean_vals, 0);

    // for(int i=0; i<10; i++){
    //     for (int j=0; j<10; j++)
    //     {
    //         //out[i*out.w+j] *= 255.0f;
    //         printf("[%.f %.f %.f]", in[(i*in.w+j)], in[rgb.cols*rgb.rows+(i*in.w+j)], in[2*rgb.cols*rgb.rows+(i*in.w+j)]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");

    clock_t st = clock();

    ncnn::Extractor ex = hed.create_extractor();
    ex.set_light_mode(false);
    ex.set_num_threads(1);
    ex.input("input", in);
    printf("input end\n");

    ncnn::Mat out;
    ex.extract("output", out);
    printf("output end\n");
    printf("%d  %d  %f %d\n", out.w, out.h, double(clock() - st)/CLOCKS_PER_SEC, hed.get_logs().size());

    for(int i=0; i<out.h; i++){
        for (int j=0; j<out.w; j++)
        {
            out[i*out.w+j] *= 255.0f;
        }
    }

    for(int i=0; i<hed.get_logs().size(); i++){
        printf("%s", hed.get_logs()[i].c_str());
    }

    // for(int i=0; i<10; i++){
    //     for (int j=0; j<10; j++)
    //     {
    //         //out[i*out.w+j] *= 255.0f;
    //         printf("%d ", (int)out[i*out.w+j]);
    //     }
    //     printf("\n");
    // }

    cv::Mat edge(out.h, out.w, CV_8UC1);
    out.to_pixels(edge.data, ncnn::Mat::PIXEL_GRAY);
    cv::imwrite(outPath, edge);

    return 0;
}

int main(int argc, char** argv)
{
    ncnn::Net hed;
    printf("hed inited\n");
    hed.load_param("hed-ncnn.param");
    printf("load_param end\n");
    hed.load_model("hed-ncnn.bin");
    printf("load_model end\n");

    cv::Mat m = cv::imread("IMG1.jpg", CV_LOAD_IMAGE_COLOR);
    cvtColor(m, m, CV_BGR2RGB);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread failed 1 \n");
        return -1;
    }

    detect_hed(hed, m, "out1.png");

    cv::Mat m2 = cv::imread("IMG2.jpg", CV_LOAD_IMAGE_COLOR);
    cvtColor(m2, m2, CV_BGR2RGB);
    if (m2.empty())
    {
        fprintf(stderr, "cv::imread failed 2 \n");
        return -1;
    }

    detect_hed(hed, m2, "out2.png");

    return 0;
}
