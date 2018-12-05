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
#include <limits.h>

#include <iostream>

#include <fstream>
#include <set>
#include <limits>
#include <algorithm>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/message.h>

using namespace std;

#include "graph.pb.h"

static bool read_proto_from_binary(const char* filepath, google::protobuf::Message* message)
{
    std::ifstream fs(filepath, std::ifstream::in | std::ifstream::binary);
    if (!fs.is_open())
    {
        fprintf(stderr, "open failed %s\n", filepath);
        return false;
    }

    google::protobuf::io::IstreamInputStream input(&fs);
    google::protobuf::io::CodedInputStream codedstr(&input);

    codedstr.SetTotalBytesLimit(INT_MAX, INT_MAX / 2);

    bool success = message->ParseFromCodedStream(&codedstr);

    fs.close();

    return success;
}

static bool find_tensor_proto(const std::map<std::string, tensorflow::TensorProto>& weights,
                              const tensorflow::NodeDef& node, tensorflow::TensorProto& tensor)
{
    for (int j=0; j<node.input_size(); j++)
    {
        const std::string& input_name = node.input(j);

        const std::map<std::string, tensorflow::TensorProto>::const_iterator it = weights.find(input_name);
        if (it != weights.end())
        {
            tensor = it->second;
            return true;
        }
    }

    return false;
}

static bool get_tensor_proto(const std::map<std::string, tensorflow::TensorProto>& consts,
                             const tensorflow::NodeDef& node, tensorflow::TensorProto& tensor)
{
    const std::string& output_name = node.name();

    const std::map<std::string, tensorflow::TensorProto>::const_iterator it = consts.find(output_name);
    if (it != consts.end())
    {
        tensor = it->second;
        return true;
    }

    return false;
}

static bool find_attr_value(const tensorflow::NodeDef& node, const char* key, tensorflow::AttrValue& value)
{
    const google::protobuf::Map<std::string, tensorflow::AttrValue>& attr = node.attr();

    const google::protobuf::Map<std::string, tensorflow::AttrValue>::const_iterator it = attr.find(key);
    if (it != attr.end())
    {
        value = it->second;
        return true;
    }

    return false;
}

static int parse_tensor_reduction_dim(const tensorflow::TensorProto& tensor)
{
    int dim = 0;

    // dim == 0 // w h c -> X X X
    // dim == 1 // w h c -> X X c
    // dim == 2 // w h c -> X h c
    // dim == -1 // w h c -> w X X
    // dim == -2 // w h c -> w h X

    if (!tensor.tensor_content().empty() && tensor.dtype() == 3)// int32
    {
        const int* data = reinterpret_cast<const int*>(tensor.tensor_content().c_str());
        int size = tensor.tensor_content().size() / sizeof(int);

        // n h w c
        // n h w
        // n w
        // TODO investigate two stage / three stage reduction
        if (size == 2)
        {
            if (data[0] == 1 && data[1] == 2)
            {
                dim = 1;
            }
        }
    }
    else
    {
        int axis = tensor.int_val(0);
        if (axis == 1)
            dim = 0;
        else if (axis == 3)
            dim = -2;
    }

    return dim;
}

//merge biasAdd to Convolution Deconvolution ConvolutionDepthWise
int findNextBiasAdd(int begin, tensorflow::GraphDef& graph, std::map<std::string, tensorflow::TensorProto>& binaryop_consts, std::map<std::string, tensorflow::TensorProto>& weights)
{
    int node_count = graph.node_size();
    for (int i=begin; i<node_count; i++)
    {
        const tensorflow::NodeDef &node = graph.node(i);
        if(node.op() == "Identity" || node.op() == "Const")
        {
            if (binaryop_consts.find(node.name()) != binaryop_consts.end())
            {
                continue;
            }
            if (weights.find(node.name()) != weights.end())
            {
                continue;
            }
        }
        if(node.op() == "BiasAdd")
        {
            return i;
        }
        break;
    }
    return -1;
}

//merge biasAdd to Convolution Deconvolution ConvolutionDepthWise
bool wouldBeMergedBiasAdd(std::string node_name, int begin, tensorflow::GraphDef& graph)
{
    int node_count = graph.node_size();
    for (int i=begin; i<node_count; i++)
    {
        const tensorflow::NodeDef &node = graph.node(i);
        if(node.op() == "Conv2DBackpropInput" || node.op() == "Conv2D" || node.op() == "DepthwiseConv2dNative")
        {
            continue;
        }
        if(node.op() == "BiasAdd")
        {
            for (int bi=0; bi<node.input_size(); bi++)
            {
                const std::string &tmp = node.input(bi);
                if(node_name == tmp){
                    return true;
                }
            }
        }
        break;
    }
    return false;
}

void write_biases(tensorflow::TensorProto tensor, FILE* bp){
    const tensorflow::TensorShapeProto& shape = tensor.tensor_shape();

    int w = 0;
    int h = 0;
    int c = 0;

    if (shape.dim_size() == 1)
    {
        w = shape.dim(0).size();
    }
    else if (shape.dim_size() == 2)
    {
        h = shape.dim(0).size();
        w = shape.dim(1).size();
    }
    else if (shape.dim_size() == 3)
    {
        c = shape.dim(2).size();
        h = shape.dim(0).size();
        w = shape.dim(1).size();
    }

    int weight_data_size = 0;

    if (!tensor.tensor_content().empty())
    {
        if (tensor.dtype() == 1)// float
        {
            const float* data = reinterpret_cast<const float*>(tensor.tensor_content().c_str());
            weight_data_size = tensor.tensor_content().size() / sizeof(float);

            if (c == 0){
                fwrite(data, sizeof(float), weight_data_size, bp);
            }
            else
            {
                float tmp;
                // h-w-c to c-h-w
                for (int p=0; p<c; p++)
                {
                    for (int i=0; i<h; i++)
                    {
                        for (int j=0; j<w; j++)
                        {
                            tmp = data[i*w*c + j*c + p];
                            fwrite(&tmp, sizeof(float), 1, bp);
                        }
                    }
                }
            }
        }
        else if (tensor.dtype() == 3)// int32
        {
            const int* data = reinterpret_cast<const int*>(tensor.tensor_content().c_str());
            weight_data_size = tensor.tensor_content().size() / sizeof(int);

            float tmp;
            if (c == 0)
            {
                for (int i=0; i<weight_data_size; i++)
                {
                    tmp = data[i];
                    fwrite(&tmp, sizeof(float), 1, bp);
                }
            }
            else
            {
                // h-w-c to c-h-w
                for (int p=0; p<c; p++)
                {
                    for (int i=0; i<h; i++)
                    {
                        for (int j=0; j<w; j++)
                        {
                            tmp = data[i*w*c + j*c + p];
                            fwrite(&tmp, sizeof(float), 1, bp);
                        }
                    }
                }
            }
        }
    }
    else
    {
        if (tensor.dtype() == 1)// float
        {
            float val = tensor.float_val(0);
            fwrite(&val, sizeof(float), 1, bp);
        }
    }
}

class ParamLine{
public:
    string op;
    string name;
    int input_size;
    int output_size;
    string tops;
    string bottoms;
    string params;
    std::set<std::string> input_names;
    std::set<std::string> output_names;

    int kw, kh, sw, sh;
};

int main(int argc, char** argv)
{
    const char* tensorflowpb = argv[1];
    const char* ncnn_prototxt = argc >= 4 ? argv[2] : "ncnn.proto";
    const char* ncnn_modelbin = argc >= 4 ? argv[3] : "ncnn.bin";

    tensorflow::GraphDef graph;

    // load
    bool s1 = read_proto_from_binary(tensorflowpb, &graph);
    if (!s1)
    {
        fprintf(stderr, "read_proto_from_binary failed\n");
        return -1;
    }

    FILE* pp = fopen(ncnn_prototxt, "wb");
    FILE* bp = fopen(ncnn_modelbin, "wb");

    // magic
    fprintf(pp, "7767517\n");

    int node_count = graph.node_size();

    // fprintf(stderr, "node_count = %d\n\n", node_count);

    // node reference
    std::map<std::string, int> node_reference;

    // mapping for Const and Const-Identity
    std::map<std::string, tensorflow::TensorProto> weights;

    // Dropout like Identity
    std::set<std::string> dropouts;

    // Const before BinaryOp
    std::map<std::string, tensorflow::TensorProto> binaryop_consts;

    // global definition line

    for (int i=0; i<node_count; i++)
    {
        const tensorflow::NodeDef& node = graph.node(i);

        const std::string& output_name = node.name();

        if (node.op() == "Const")
        {
            tensorflow::AttrValue value;
            if (find_attr_value(node, "value", value))
            {
                const tensorflow::TensorProto& tensor = value.tensor();
                weights[output_name] = tensor;
            }
            continue;
        }
        else if (node.op() == "Identity")
        {
            const std::string& input_name = node.input(0);
            if (weights.find(input_name) != weights.end())
            {
                weights[output_name] = weights[input_name];
                continue;
            }
            else
            {
                dropouts.insert(output_name);
            }
        }
        else if (node.op() == "NoOp")
        {
            weights[output_name] = tensorflow::TensorProto();
            continue;
        }
        else
        {
            bool isBinaryOp = false;
            if (node.op() == "Add" || node.op() == "BiasAdd" || node.op() == "Div"
                || node.op() == "Mul" || node.op() == "RealDiv" || node.op() == "Sub")
            {
                isBinaryOp = true;
            }
            if (node.op() == "Max" || node.op() == "Maximum" || node.op() == "Min" || node.op() == "Minimum")
            {
                // check weights
                tensorflow::TensorProto tensor;
                if (!find_tensor_proto(weights, node, tensor))
                {
                    isBinaryOp = true;
                }
            }

            if (isBinaryOp)
            {
                // check weights
                for (int j=0; j<node.input_size(); j++)
                {
                    const std::string& input_name = node.input(j);

                    std::map<std::string, tensorflow::TensorProto>::iterator it = weights.find(input_name);
                    if (it != weights.end())
                    {
                        // binary op with const, insert MemoryData layer and const blob
                        binaryop_consts[input_name] = it->second;
                        weights.erase(it);
                    }
                }
            }
        }

        // input
        for (int j=0; j<node.input_size(); j++)
        {
            const std::string& input_name = node.input(j);
            // printf(stderr, "input = %s\n", input_name.c_str());

            if (weights.find(input_name) != weights.end())
            {
                continue;
            }

            if (node_reference.find(input_name) == node_reference.end())
            {
                node_reference[input_name] = 1;
            }
            else
            {
                node_reference[input_name] = node_reference[input_name] + 1;
            }
        }

        // output
        // fprintf(stderr, "output = %s\n", output_name.c_str());
    }

    std::map<std::string, int>::iterator it = node_reference.begin();
    while (it != node_reference.end())
    {
        if (it->second == 1)
        {
            node_reference.erase(it++);
        }
        else
        {
            ++it;
        }
    }

    int internal_split = 0;
    bool ignoreBiasAdd = false;
    //out params
    vector<ParamLine> param_lines;

    for (int i=0; i<node_count; i++)
    {
        const tensorflow::NodeDef& node = graph.node(i);
        ParamLine p_line;

        // layer definition line, repeated
        // [type] [name] [bottom blob count] [top blob count] [bottom blobs] [top blobs] [layer specific params]
        // fprintf(pp, "%-16s %-16s %d %d", layer.type().c_str(), layer.name().c_str(), node.input_size(), layer.top_size());

        if (node.op() == "Add")
        {
            p_line.op = "BinaryOp";
        }
        else if(node.op() == "BiasAdd"){
            if(ignoreBiasAdd){
                ignoreBiasAdd = false;
                continue;
            }
            p_line.op = "BinaryOp";
        }
        else if (node.op() == "AvgPool")
        {
            p_line.op = "Pooling";
        }
        else if (node.op() == "Concat" || node.op() == "ConcatV2")
        {
            p_line.op = "Concat";
        }
        else if (node.op() == "Const")
        {
            // check before binaryop
            tensorflow::TensorProto tensor;
            if (get_tensor_proto(binaryop_consts, node, tensor))
            {
                if((strstr(node.name().c_str(), "biases/read") != NULL)
                        && (ignoreBiasAdd || wouldBeMergedBiasAdd(node.name(), i+1, graph))){
                    continue;
                }
                p_line.op = "MemoryData";
            }
            else
            {
                continue;
            }
        }
        else if (node.op() == "Conv2D")
        {
            p_line.op = "Convolution";
        }
        else if (node.op() == "Conv2DBackpropInput")
        {
            p_line.op = "Deconvolution";
        }
        else if(node.op() == "FusedBatchNorm")
        {
            p_line.op = "BatchNorm";
        }
        else if (node.op() == "DepthwiseConv2dNative")
        {
            p_line.op = "ConvolutionDepthWise";
        }
        else if (node.op() == "Div" || node.op() == "RealDiv")
        {
            p_line.op = "BinaryOp";
        }
        else if (node.op() == "Exp")
        {
            p_line.op = "UnaryOp";
        }
        else if (node.op() == "ExpandDims")
        {
            p_line.op = "ExpandDims";
        }
        else if (node.op() == "Floor")
        {
            p_line.op = "UnaryOp";
        }
        else if (node.op() == "Identity")
        {
            // check before binaryop
            tensorflow::TensorProto tensor;
            if (get_tensor_proto(binaryop_consts, node, tensor))
            {
                if((strstr(node.name().c_str(), "biases/read") != NULL)
                        && (ignoreBiasAdd || wouldBeMergedBiasAdd(node.name(), i+1, graph))){
                    continue;
                }
                p_line.op = "MemoryData";
            }
            else if (dropouts.find(node.name()) != dropouts.end())
            {
                p_line.op = "Dropout";
            }
            else
            {
                continue;
            }
        }
        else if (node.op() == "LRN")
        {
            p_line.op = "LRN";
        }
        else if (node.op() == "MatMul")
        {
            p_line.op = "MatMul";
        }
        else if (node.op() == "Max" || node.op() == "Maximum")
        {
            // check weights
            tensorflow::TensorProto tensor;
            if (find_tensor_proto(weights, node, tensor))
            {
                p_line.op = "Reduction";
            }
            else
            {
                p_line.op = "BinaryOp";
            }
        }
        else if (node.op() == "MaxPool")
        {
            p_line.op = "Pooling";
        }
        else if (node.op() == "Min" || node.op() == "Minimum")
        {
            // check weights
            tensorflow::TensorProto tensor;
            if (find_tensor_proto(weights, node, tensor))
            {
                p_line.op = "Reduction";
            }
            else
            {
                p_line.op = "BinaryOp";
            }
        }
        else if (node.op() == "Mul")
        {
            p_line.op = "BinaryOp";
        }
        else if (node.op() == "Neg")
        {
            p_line.op = "UnaryOp";
        }
        else if (node.op() == "NoOp")
        {
            continue;
        }
        else if (node.op() == "Pad")
        {
            p_line.op = "Padding";
        }
        else if (node.op() == "Placeholder")
        {
            p_line.op = "Input";
        }
        else if (node.op() == "Prod")
        {
            p_line.op = "Reduction";
        }
        else if (node.op() == "Reciprocal")
        {
            p_line.op = "UnaryOp";
        }
        else if (node.op() == "Relu")
        {
            p_line.op = "ReLU";
        }
        else if (node.op() == "Reshape")
        {
            p_line.op = "Reshape";
        }
        else if (node.op() == "Rsqrt")
        {
            p_line.op = "UnaryOp";
        }
        else if (node.op() == "Sigmoid")
        {
            p_line.op = "Sigmoid";
        }
        else if (node.op() == "Softmax")
        {
            p_line.op = "Softmax";
        }
        else if (node.op() == "Square")
        {
            p_line.op = "UnaryOp";
        }
        else if (node.op() == "Squeeze")
        {
            p_line.op = "Squeeze";
        }
        else if (node.op() == "Sub")
        {
            p_line.op = "BinaryOp";
        }
        else if (node.op() == "Sum")
        {
            p_line.op = "Reduction";
        }
        else
        {
            p_line.op = node.op().c_str();
            fprintf(stderr, "%s not supported yet !\nn", node.op().c_str());
        }

        int input_size = node.input_size();
        for (int j=0; j<node.input_size(); j++)
        {
            const std::string& input_name = node.input(j);
            if (weights.find(input_name) != weights.end())
            {
                input_size--;
            }
        }

        p_line.name = node.name();
        p_line.input_size = input_size;
        p_line.output_size = 1;

        for (int j=0; j<node.input_size(); j++)
        {
            std::string input_name = node.input(j);

            if (weights.find(input_name) != weights.end())
            {
                continue;
            }

            if (node_reference.find(input_name) != node_reference.end())
            {
                int refidx = node_reference[input_name] - 1;
                node_reference[input_name] = refidx;

                char splitsuffix[256];
                sprintf(splitsuffix, "_splitncnn_%d", refidx);
                input_name = input_name + splitsuffix;
            }
            if(p_line.tops.empty()){
                p_line.tops = input_name;
            } else {
                p_line.tops = p_line.tops + " " + input_name;
            }
            p_line.input_names.insert(input_name);
        }

        if (node.op() == "Add" || node.op() == "BiasAdd")
        {
            int op_type = 0;
            p_line.params = "0="+to_string(op_type);
        }
        else if (node.op() == "AvgPool")
        {
            int pooling_type = 1;

            int kernel_size_h = 1;
            int kernel_size_w = 1;
            int stride_h = 1;
            int stride_w = 1;
            int pad = 0;

            int global_pooling = 0;
            int pad_mode = 1;

            tensorflow::AttrValue value_ksize;
            if (find_attr_value(node, "ksize", value_ksize))
            {
                // batch, height, width, channels
                kernel_size_h = value_ksize.list().i(1);
                kernel_size_w = value_ksize.list().i(2);
            }

            tensorflow::AttrValue value_strides;
            if (find_attr_value(node, "strides", value_strides))
            {
                // batch, height, width, channels
                stride_h = value_strides.list().i(1);
                stride_w = value_strides.list().i(2);
            }

            tensorflow::AttrValue value_padding;
            if (find_attr_value(node, "padding", value_padding))
            {
                if (value_padding.s() == "VALID")
                {
                    pad_mode = 1;
                }
                else if (value_padding.s() == "SAME")
                {
                    pad_mode = 2;
                }
            }

            p_line.params  = "0="+to_string(pooling_type);
            p_line.params += " 1=" + to_string(kernel_size_w);
            p_line.params += " 11=" + to_string(kernel_size_h);
            p_line.params += " 2=" + to_string(stride_w);
            p_line.params += " 12=" + to_string(stride_h);
            p_line.params += " 3=" + to_string(pad);
            p_line.params += " 4=" + to_string(global_pooling);
            p_line.params += " 5=" + to_string(pad_mode);
        }
        else if (node.op() == "Concat" || node.op() == "ConcatV2")
        {
            tensorflow::TensorProto tensor;
            if (find_tensor_proto(weights, node, tensor))
            {
                // TODO
                //   int axis = tensor.int_val(0);
            }
        }
        else if (node.op() == "Const" || node.op() == "Identity")
        {
            // check before binaryop
            tensorflow::TensorProto tensor;
            if (get_tensor_proto(binaryop_consts, node, tensor))
            {
                const tensorflow::TensorShapeProto& shape = tensor.tensor_shape();

                int w = 0;
                int h = 0;
                int c = 0;

                if (shape.dim_size() == 1)
                {
                    w = shape.dim(0).size();
                }
                else if (shape.dim_size() == 2)
                {
                    h = shape.dim(0).size();
                    w = shape.dim(1).size();
                }
                else if (shape.dim_size() == 3)
                {
                    c = shape.dim(2).size();
                    h = shape.dim(0).size();
                    w = shape.dim(1).size();
                }

                int weight_data_size = 0;

                if (!tensor.tensor_content().empty())
                {
                    if (tensor.dtype() == 1)// float
                    {
                        const float* data = reinterpret_cast<const float*>(tensor.tensor_content().c_str());
                        weight_data_size = tensor.tensor_content().size() / sizeof(float);

                        if (c == 0)
                            fwrite(data, sizeof(float), weight_data_size, bp);
                        else
                        {
                            float tmp;
                            // h-w-c to c-h-w
                            for (int p=0; p<c; p++)
                            {
                                for (int i=0; i<h; i++)
                                {
                                    for (int j=0; j<w; j++)
                                    {
                                        tmp = data[i*w*c + j*c + p];
                                        fwrite(&tmp, sizeof(float), 1, bp);
                                    }
                                }
                            }
                        }
                    }
                    else if (tensor.dtype() == 3)// int32
                    {
                        const int* data = reinterpret_cast<const int*>(tensor.tensor_content().c_str());
                        weight_data_size = tensor.tensor_content().size() / sizeof(int);

                        float tmp;
                        if (c == 0)
                        {
                            for (int i=0; i<weight_data_size; i++)
                            {
                                tmp = data[i];
                                fwrite(&tmp, sizeof(float), 1, bp);
                            }
                        }
                        else
                        {
                            // h-w-c to c-h-w
                            for (int p=0; p<c; p++)
                            {
                                for (int i=0; i<h; i++)
                                {
                                    for (int j=0; j<w; j++)
                                    {
                                        tmp = data[i*w*c + j*c + p];
                                        fwrite(&tmp, sizeof(float), 1, bp);
                                    }
                                }
                            }
                        }
                    }
                }
                else
                {
                    if (tensor.dtype() == 1)// float
                    {
                        float val = tensor.float_val(0);
                        fwrite(&val, sizeof(float), 1, bp);
                    }
                    else if (tensor.dtype() == 3)// int32
                    {
                        float val = tensor.int_val(0);
                        fwrite(&val, sizeof(float), 1, bp);
                    }
                }

                p_line.params  = "0="+to_string(w);
                p_line.params += " 1=" + to_string(h);
                p_line.params += " 2=" + to_string(c);
            }
        }
        else if (node.op() == "Conv2D")
        {
            // weights
            tensorflow::TensorProto tensor;
            find_tensor_proto(weights, node, tensor);

            const tensorflow::TensorShapeProto& shape = tensor.tensor_shape();

            int kernel_size_h = shape.dim(0).size();
            int kernel_size_w = shape.dim(1).size();
            int num_input = shape.dim(2).size();
            int num_output = shape.dim(3).size();

            int stride_h = 1;
            int stride_w = 1;
            int dilation_h = 1;
            int dilation_w = 1;
            int pad = 0;

            tensorflow::AttrValue value_strides;
            if (find_attr_value(node, "strides", value_strides))
            {
                // batch, height, width, channels
                stride_h = value_strides.list().i(1);
                stride_w = value_strides.list().i(2);
            }

            tensorflow::AttrValue value_padding;
            if (find_attr_value(node, "padding", value_padding))
            {
                if (value_padding.s() == "VALID")
                {
                    pad = 0;
                }
                else if (value_padding.s() == "SAME")
                {
                    pad = -233;
                }
            }

            tensorflow::AttrValue value_rate;
            if (find_attr_value(node, "rate", value_rate))
            {
                // height, width
                dilation_h = value_rate.list().i(0);
                dilation_w = value_rate.list().i(1);
            }

            int bias_term = 0;
            int weight_data_size = 0;

            // reorder h-w-i-o to o-i-h-w
            if (!tensor.tensor_content().empty())
            {
                int quantize_tag = 0;
                fwrite(&quantize_tag, sizeof(int), 1, bp);

                if (tensor.dtype() == 1)// float
                {
                    const float* data = reinterpret_cast<const float*>(tensor.tensor_content().c_str());
                    weight_data_size = tensor.tensor_content().size() / sizeof(float);

                    float tmp;
                    for (int p=0; p<num_output; p++)
                    {
                        for (int q=0; q<num_input; q++)
                        {
                            for (int i=0; i<kernel_size_h; i++)
                            {
                                for (int j=0; j<kernel_size_w; j++)
                                {
                                    tmp = data[i*kernel_size_w*num_input*num_output + j*num_input*num_output + q*num_output + p];
                                    fwrite(&tmp, sizeof(float), 1, bp);
                                }
                            }
                        }
                    }
                }
                else if (tensor.dtype() == 3)// int32
                {
                    const int* data = reinterpret_cast<const int*>(tensor.tensor_content().c_str());
                    weight_data_size = tensor.tensor_content().size() / sizeof(int);

                    float tmp;
                    for (int p=0; p<num_output; p++)
                    {
                        for (int q=0; q<num_input; q++)
                        {
                            for (int i=0; i<kernel_size_h; i++)
                            {
                                for (int j=0; j<kernel_size_w; j++)
                                {
                                    tmp = data[i*kernel_size_w*num_input*num_output + j*num_input*num_output + q*num_output + p];//[h,w,out_channels, in_channels]
                                    fwrite(&tmp, sizeof(float), 1, bp);
                                }
                            }
                        }
                    }
                }
            }

            int bias_index = findNextBiasAdd(i+1, graph, binaryop_consts, weights);
            if(bias_index > 0){
                const tensorflow::NodeDef& node_bias = graph.node(bias_index);
                tensorflow::TensorProto tensor_bias;
                std::string bias_input_name;
                bool has_bias = false;
                for (int bi=0; bi<node_bias.input_size(); bi++)
                {
                    const std::string& tmp = node_bias.input(bi);
                    if(strstr(tmp.c_str(), "biases/read") != NULL){
                        bias_input_name = tmp;
                        const std::map<std::string, tensorflow::TensorProto>::const_iterator it = binaryop_consts.find(bias_input_name);
                        if (it != binaryop_consts.end())
                        {
                            tensor_bias = it->second;
                            has_bias = true;
                        }
                        break;
                    }
                }
                if(has_bias){
                    ignoreBiasAdd = true;
                    bias_term = 1;
                    p_line.name = node_bias.name();
                    write_biases(tensor_bias, bp);
                }
            }
            p_line.kw = kernel_size_w;
            p_line.kh = kernel_size_h;
            p_line.sw = stride_w;
            p_line.sh = stride_h;
            p_line.params  = "0="+to_string(num_output);
            p_line.params += " 1=" + to_string(kernel_size_w);
            p_line.params += " 11=" + to_string(kernel_size_h);
            p_line.params += " 2=" + to_string(dilation_w);
            p_line.params += " 12=" + to_string(dilation_h);
            p_line.params += " 3=" + to_string(stride_w);
            p_line.params += " 13=" + to_string(stride_h);
            p_line.params += " 4=" + to_string(pad);
            p_line.params += " 5=" + to_string(bias_term);
            p_line.params += " 6=" + to_string(weight_data_size);
        }
        else if (node.op() == "Conv2DBackpropInput")
        {
            // weights
            tensorflow::TensorProto tensor;
            for (int j=0; j<node.input_size(); j++)
            {
                const std::string& input_name = node.input(j);
                const std::map<std::string, tensorflow::TensorProto>::const_iterator it = weights.find(input_name);
                if (it != weights.end())
                {
                    if((strstr(input_name.c_str(), "weights") != NULL)){
                        tensor = it->second;
                    }
                }
            }

            const tensorflow::TensorShapeProto& shape = tensor.tensor_shape();

            int kernel_size_h = shape.dim(0).size();
            int kernel_size_w = shape.dim(1).size();
            int num_output = shape.dim(2).size();
            int num_input = shape.dim(3).size();

            int stride_h = 1;
            int stride_w = 1;
            int dilation_h = 1;
            int dilation_w = 1;
            int pad = 0;

            tensorflow::AttrValue value_strides;
            if (find_attr_value(node, "strides", value_strides))
            {
                stride_h = value_strides.list().i(1);
                stride_w = value_strides.list().i(2);
            }

            tensorflow::AttrValue value_padding;
            if (find_attr_value(node, "padding", value_padding))
            {
                if (value_padding.s() == "VALID")
                {
                    pad = 0;
                }
                else if (value_padding.s() == "SAME")
                {
                    pad = -233;
                }
            }

            tensorflow::AttrValue value_rate;
            if (find_attr_value(node, "rate", value_rate))
            {
                dilation_h = value_rate.list().i(0);
                dilation_w = value_rate.list().i(1);
            }

            int bias_term = 0;
            int weight_data_size = 0;

            // reorder h-w-i-o to o-i-h-w
            if (!tensor.tensor_content().empty())
            {
                int quantize_tag = 0;
                fwrite(&quantize_tag, sizeof(int), 1, bp);

                if (tensor.dtype() == 1)// float
                {
                    const float* data = reinterpret_cast<const float*>(tensor.tensor_content().c_str());
                    weight_data_size = tensor.tensor_content().size() / sizeof(float);

                    float tmp;
                    for (int p=0; p<num_output; p++)
                    {
                        for (int q=0; q<num_input; q++)
                        {
                            for (int i=0; i<kernel_size_h; i++)
                            {
                                for (int j=0; j<kernel_size_w; j++)
                                {
                                    tmp = data[i*kernel_size_w*num_input*num_output + j*num_input*num_output + p*num_input + q];//[h,w,in_channels, out_channels]
                                    fwrite(&tmp, sizeof(float), 1, bp);
                                }
                            }
                        }
                    }
                }
                else if (tensor.dtype() == 3)// int32
                {
                    const int* data = reinterpret_cast<const int*>(tensor.tensor_content().c_str());
                    weight_data_size = tensor.tensor_content().size() / sizeof(int);

                    float tmp;
                    for (int p=0; p<num_output; p++)
                    {
                        for (int q=0; q<num_input; q++)
                        {
                            for (int i=0; i<kernel_size_h; i++)
                            {
                                for (int j=0; j<kernel_size_w; j++)
                                {
                                    tmp = data[i*kernel_size_w*num_input*num_output + j*num_input*num_output + p*num_input + q];
                                    fwrite(&tmp, sizeof(float), 1, bp);
                                }
                            }
                        }
                    }
                }
            }

            int bias_index = findNextBiasAdd(i+1, graph, binaryop_consts, weights);
            if(bias_index > 0){
                const tensorflow::NodeDef& node_bias = graph.node(bias_index);
                tensorflow::TensorProto tensor_bias;
                std::string bias_input_name;
                bool has_bias = false;
                for (int bi=0; bi<node_bias.input_size(); bi++)
                {
                    const std::string& tmp = node_bias.input(bi);
                    if(strstr(tmp.c_str(), "biases/read") != NULL){
                        bias_input_name = tmp;
                        const std::map<std::string, tensorflow::TensorProto>::const_iterator it = binaryop_consts.find(bias_input_name);
                        if (it != binaryop_consts.end())
                        {
                            tensor_bias = it->second;
                            has_bias = true;
                        }
                        break;
                    }
                }
                if(has_bias){
                    ignoreBiasAdd = true;
                    bias_term = 1;
                    p_line.name = node_bias.name();
                    write_biases(tensor_bias, bp);
                }
            }

            p_line.kw = kernel_size_w;
            p_line.kh = kernel_size_h;
            p_line.sw = stride_w;
            p_line.sh = stride_h;
            p_line.params  = "0="+to_string(num_output);
            p_line.params += " 1=" + to_string(kernel_size_w);
            p_line.params += " 11=" + to_string(kernel_size_h);
            p_line.params += " 2=" + to_string(dilation_w);
            p_line.params += " 12=" + to_string(dilation_h);
            p_line.params += " 3=" + to_string(stride_w);
            p_line.params += " 13=" + to_string(stride_h);
            p_line.params += " 4=" + to_string(pad);
            p_line.params += " 5=" + to_string(bias_term);
            p_line.params += " 6=" + to_string(weight_data_size);
        }
        else if (node.op() == "FusedBatchNorm")
        {
            //epsilon
            float eps = 0.0f;
            tensorflow::AttrValue value_eps;
            if (find_attr_value(node, "epsilon", value_eps))
            {
                eps = value_eps.f();
            }
            //mean var beta
            tensorflow::TensorProto tensor_gamma;
            tensorflow::TensorProto tensor_mean;
            tensorflow::TensorProto tensor_var;
            tensorflow::TensorProto tensor_beta;
            bool has_gamma = false;
            bool has_beta = false;
            for (int j=0; j<node.input_size(); j++)
            {
                const std::string& input_name = node.input(j);
                const std::map<std::string, tensorflow::TensorProto>::const_iterator it = weights.find(input_name);
                if (it != weights.end())
                {
                    if((strstr(input_name.c_str(), "beta") != NULL)){
                        tensor_beta = it->second;
                        has_beta = true;
                    }
                    if((strstr(input_name.c_str(), "moving_mean") != NULL)){
                        tensor_mean = it->second;
                    }
                    if((strstr(input_name.c_str(), "moving_variance") != NULL)){
                        tensor_var = it->second;
                    }
                    if((strstr(input_name.c_str(), "gamma") != NULL))
                    {
                        tensor_gamma = it->second;
                        has_gamma = true;
                    }
                }
            }
            int mean_size = tensor_mean.tensor_shape().dim(0).size();

            if(has_gamma){
                if (tensor_gamma.dtype() == 1)// float
                {
                    const float* data = reinterpret_cast<const float*>(tensor_gamma.tensor_content().c_str());
                    int size = tensor_gamma.tensor_content().size() / sizeof(float);
                    float tmp;
                    for (int p=0; p<size; p++)
                    {
                        tmp = data[p];
                        fwrite(&tmp, sizeof(float), 1, bp);
                    }
                } else if(tensor_gamma.dtype() == 3){
                    const int* data = reinterpret_cast<const int*>(tensor_gamma.tensor_content().c_str());
                    int size = tensor_gamma.tensor_content().size() / sizeof(int);
                    float tmp;
                    for (int p=0; p<size; p++)
                    {
                        tmp = data[p];
                        fwrite(&tmp, sizeof(float), 1, bp);
                    }
                }
            } else{
                std::vector<float> ones(mean_size, 1.f);
                fwrite(ones.data(), sizeof(float), ones.size(), bp);// default gamma 1.0
            }
      
            if (tensor_mean.dtype() == 1)// float
            {
                const float* data = reinterpret_cast<const float*>(tensor_mean.tensor_content().c_str());
                int size = tensor_mean.tensor_content().size() / sizeof(float);
                float tmp;
                for (int p=0; p<size; p++)
                {
                    tmp = data[p];
                    fwrite(&tmp, sizeof(float), 1, bp);
                }
            } else if(tensor_mean.dtype() == 3){// int32
                const int* data = reinterpret_cast<const int*>(tensor_mean.tensor_content().c_str());
                int size = tensor_mean.tensor_content().size() / sizeof(int);
                float tmp;
                for (int p=0; p<size; p++)
                {
                    tmp = data[p];
                    fwrite(&tmp, sizeof(float), 1, bp);
                }
            }
            if (tensor_var.dtype() == 1)// float
            {
                const float* data = reinterpret_cast<const float*>(tensor_var.tensor_content().c_str());
                int size = tensor_var.tensor_content().size() / sizeof(float);
                float tmp;
                for (int p=0; p<size; p++)
                {
                    tmp = data[p]+eps;
                    fwrite(&tmp, sizeof(float), 1, bp);
                }
            } else if(tensor_var.dtype() == 3){// int32
                const int* data = reinterpret_cast<const int*>(tensor_var.tensor_content().c_str());
                int size = tensor_var.tensor_content().size() / sizeof(int);
                float tmp;
                for (int p=0; p<size; p++)
                {
                    tmp = data[p]+eps;
                    fwrite(&tmp, sizeof(float), 1, bp);
                }
            }
            if(has_beta){
                if (tensor_beta.dtype() == 1)// float
                {
                    const float* data = reinterpret_cast<const float*>(tensor_beta.tensor_content().c_str());
                    int size = tensor_beta.tensor_content().size() / sizeof(float);
                    float tmp;
                    for (int p=0; p<size; p++)
                    {
                        tmp = data[p];
                        fwrite(&tmp, sizeof(float), 1, bp);
                    }
                } else if(tensor_beta.dtype() == 3){// int32
                    const int* data = reinterpret_cast<const int*>(tensor_beta.tensor_content().c_str());
                    int size = tensor_beta.tensor_content().size() / sizeof(int);
                    float tmp;
                    for (int p=0; p<size; p++)
                    {
                        tmp = data[p];
                        fwrite(&tmp, sizeof(float), 1, bp);
                    }
                }
            } else {
                std::vector<float> zeros(mean_size, 0.f);
                fwrite(zeros.data(), sizeof(float), zeros.size(), bp);// // default beta 0.0
            }

            p_line.params  = "0="+to_string(mean_size);
        }
        else if (node.op() == "DepthwiseConv2dNative")
        {
            // weights
            tensorflow::TensorProto tensor;
            find_tensor_proto(weights, node, tensor);

            const tensorflow::TensorShapeProto& shape = tensor.tensor_shape();

            int kernel_size_h = shape.dim(0).size();
            int kernel_size_w = shape.dim(1).size();
            int num_input = shape.dim(2).size();
            int channel_multiplier = shape.dim(3).size();

            int num_output = num_input * channel_multiplier;
            int group = num_input;

            int stride_h = 1;
            int stride_w = 1;
            int dilation_h = 1;
            int dilation_w = 1;
            int pad = 0;

            tensorflow::AttrValue value_strides;
            if (find_attr_value(node, "strides", value_strides))
            {
                // batch, height, width, channels
                stride_h = value_strides.list().i(1);
                stride_w = value_strides.list().i(2);
            }

            tensorflow::AttrValue value_padding;
            if (find_attr_value(node, "padding", value_padding))
            {
                if (value_padding.s() == "VALID")
                {
                    pad = 0;
                }
                else if (value_padding.s() == "SAME")
                {
                    pad = -233;
                }
            }

            tensorflow::AttrValue value_rate;
            if (find_attr_value(node, "rate", value_rate))
            {
                // height, width
                dilation_h = value_rate.list().i(0);
                dilation_w = value_rate.list().i(1);
            }

            int bias_term = 0;
            int weight_data_size = 0;

            // reorder h-w-i-cm to i-cm-h-w
            if (!tensor.tensor_content().empty())
            {
                int quantize_tag = 0;
                fwrite(&quantize_tag, sizeof(int), 1, bp);

                if (tensor.dtype() == 1)// float
                {
                    const float* data = reinterpret_cast<const float*>(tensor.tensor_content().c_str());
                    weight_data_size = tensor.tensor_content().size() / sizeof(float);

                    float tmp;
                    for (int p=0; p<num_input; p++)
                    {
                        for (int q=0; q<channel_multiplier; q++)
                        {
                            for (int i=0; i<kernel_size_h; i++)
                            {
                                for (int j=0; j<kernel_size_w; j++)
                                {
                                    tmp = data[i*kernel_size_w*channel_multiplier*num_input + j*channel_multiplier*num_input + p*channel_multiplier + q];
                                    fwrite(&tmp, sizeof(float), 1, bp);
                                }
                            }
                        }
                    }
                }
                else if (tensor.dtype() == 3)// int32
                {
                    const int* data = reinterpret_cast<const int*>(tensor.tensor_content().c_str());
                    weight_data_size = tensor.tensor_content().size() / sizeof(int);

                    float tmp;
                    for (int p=0; p<num_input; p++)
                    {
                        for (int q=0; q<channel_multiplier; q++)
                        {
                            for (int i=0; i<kernel_size_h; i++)
                            {
                                for (int j=0; j<kernel_size_w; j++)
                                {
                                    tmp = data[i*kernel_size_w*channel_multiplier*num_input + j*channel_multiplier*num_input + p*channel_multiplier + q];
                                    fwrite(&tmp, sizeof(float), 1, bp);
                                }
                            }
                        }
                    }
                }
            }

            int bias_index = findNextBiasAdd(i+1, graph, binaryop_consts, weights);
            if(bias_index > 0){
                const tensorflow::NodeDef& node_bias = graph.node(bias_index);
                tensorflow::TensorProto tensor_bias;
                std::string bias_input_name;
                bool has_bias = false;
                for (int bi=0; bi<node_bias.input_size(); bi++)
                {
                    const std::string& tmp = node_bias.input(bi);
                    if(strstr(tmp.c_str(), "biases/read") != NULL){
                        bias_input_name = tmp;
                        const std::map<std::string, tensorflow::TensorProto>::const_iterator it = binaryop_consts.find(bias_input_name);
                        if (it != binaryop_consts.end())
                        {
                            tensor_bias = it->second;
                            has_bias = true;
                        }
                        break;
                    }
                }
                if(has_bias){
                    ignoreBiasAdd = true;
                    bias_term = 1;
                    p_line.name = node_bias.name();
                    write_biases(tensor_bias, bp);
                }
            }

            p_line.kw = kernel_size_w;
            p_line.kh = kernel_size_h;
            p_line.sw = stride_w;
            p_line.sh = stride_h;
            p_line.params  = "0="+to_string(num_output);
            p_line.params += " 1=" + to_string(kernel_size_w);
            p_line.params += " 11=" + to_string(kernel_size_h);
            p_line.params += " 2=" + to_string(dilation_w);
            p_line.params += " 12=" + to_string(dilation_h);
            p_line.params += " 3=" + to_string(stride_w);
            p_line.params += " 13=" + to_string(stride_h);
            p_line.params += " 4=" + to_string(pad);
            p_line.params += " 5=" + to_string(bias_term);
            p_line.params += " 6=" + to_string(weight_data_size);
            p_line.params += " 7=" + to_string(group);
        }
        else if (node.op() == "Div" || node.op() == "RealDiv")
        {
            int op_type = 3;
            p_line.params += "0=" + to_string(op_type);
        }
        else if (node.op() == "Exp")
        {
            int op_type = 7;
            p_line.params += "0=" + to_string(op_type);
        }
        else if (node.op() == "ExpandDims")
        {
            int expand_w = 0;
            int expand_h = 0;
            int expand_c = 0;

            tensorflow::AttrValue value_dim;
            if (find_attr_value(node, "Tdim", value_dim))
            {
                int dim = value_dim.i();
                if (dim == 0)
                    expand_w = 1;
                if (dim == 1)
                    expand_h = 1;
                if (dim == 2)
                    expand_c = 1;
            }

            p_line.params  = "0="+to_string(expand_w);
            p_line.params += " 1=" + to_string(expand_h);
            p_line.params += " 2=" + to_string(expand_c);
        }
        else if (node.op() == "Floor")
        {
            int op_type = 2;
            p_line.params += "0=" + to_string(op_type);
        }
        else if (node.op() == "LRN")
        {
            int norm_region = 0;
            int local_size = 1;
            float alpha = 1.f;
            float beta = 0.5f;

            tensorflow::AttrValue value_depth_radius;
            if (find_attr_value(node, "depth_radius", value_depth_radius))
            {
                local_size = value_depth_radius.i() * 2 + 1;
            }

            tensorflow::AttrValue value_alpha;
            if (find_attr_value(node, "alpha", value_alpha))
            {
                alpha = value_alpha.f();
            }

            tensorflow::AttrValue value_beta;
            if (find_attr_value(node, "beta", value_beta))
            {
                beta = value_beta.f();
            }

            // TODO
            float bias = 1.f;
            tensorflow::AttrValue value_bias;
            if (find_attr_value(node, "bias", value_bias))
            {
                bias = value_bias.f();
            }

            p_line.params  = "0="+to_string(norm_region);
            p_line.params += " 1=" + to_string(local_size);
            p_line.params += " 2=" + to_string(alpha);
            p_line.params += " 3=" + to_string(beta);
        }
        else if (node.op() == "MatMul")
        {
            // weights
            tensorflow::TensorProto tensor;
            find_tensor_proto(weights, node, tensor);

            const tensorflow::TensorShapeProto& shape = tensor.tensor_shape();

            int num_input = shape.dim(0).size();
            int num_output = shape.dim(1).size();

            int bias_term = 0;
            int weight_data_size = 0;

            // reorder i-o to o-i
            if (!tensor.tensor_content().empty())
            {
                int quantize_tag = 0;
                fwrite(&quantize_tag, sizeof(int), 1, bp);

                if (tensor.dtype() == 1)// float
                {
                    const float* data = reinterpret_cast<const float*>(tensor.tensor_content().c_str());
                    weight_data_size = tensor.tensor_content().size() / sizeof(float);

                    float tmp;
                    for (int p=0; p<num_output; p++)
                    {
                        for (int q=0; q<num_input; q++)
                        {
                            tmp = data[q*num_output + p];
                            fwrite(&tmp, sizeof(float), 1, bp);
                        }
                    }
                }
                else if (tensor.dtype() == 3)// int32
                {
                    const int* data = reinterpret_cast<const int*>(tensor.tensor_content().c_str());
                    weight_data_size = tensor.tensor_content().size() / sizeof(int);

                    float tmp;
                    for (int p=0; p<num_output; p++)
                    {
                        for (int q=0; q<num_input; q++)
                        {
                            tmp = data[q*num_output + p];
                            fwrite(&tmp, sizeof(float), 1, bp);
                        }
                    }
                }
            }

            p_line.params  = "0="+to_string(num_output);
            p_line.params += " 1=" + to_string(bias_term);
            p_line.params += " 2=" + to_string(weight_data_size);
        }
        else if (node.op() == "Max" || node.op() == "Maximum")
        {
            // check weights
            tensorflow::TensorProto tensor;
            if (find_tensor_proto(weights, node, tensor))
            {
                int operation = 4;
                int dim = 0;
                float coeff = 1.f;

                dim = parse_tensor_reduction_dim(tensor);

                p_line.params  = "0="+to_string(operation);
                p_line.params += " 1=" + to_string(dim);
                p_line.params += " 2=" + to_string(coeff);
            }
            else
            {
                int op_type = 4;
                p_line.params  = "0="+to_string(op_type);
            }
        }
        else if (node.op() == "MaxPool")
        {
            int pooling_type = 0;

            int kernel_size_h = 1;
            int kernel_size_w = 1;
            int stride_h = 1;
            int stride_w = 1;
            int pad = 0;

            int global_pooling = 0;
            int pad_mode = 1;

            tensorflow::AttrValue value_ksize;
            if (find_attr_value(node, "ksize", value_ksize))
            {
                // batch, height, width, channels
                kernel_size_h = value_ksize.list().i(1);
                kernel_size_w = value_ksize.list().i(2);
            }

            tensorflow::AttrValue value_strides;
            if (find_attr_value(node, "strides", value_strides))
            {
                // batch, height, width, channels
                stride_h = value_strides.list().i(1);
                stride_w = value_strides.list().i(2);
            }

            tensorflow::AttrValue value_padding;
            if (find_attr_value(node, "padding", value_padding))
            {
                if (value_padding.s() == "VALID")
                {
                    pad_mode = 1;
                }
                else if (value_padding.s() == "SAME")
                {
                    pad_mode = 2;
                }
            }

            p_line.params  = "0="+to_string(pooling_type);
            p_line.params += " 1=" + to_string(kernel_size_w);
            p_line.params += " 11=" + to_string(kernel_size_h);
            p_line.params += " 2=" + to_string(stride_w);
            p_line.params += " 12=" + to_string(stride_h);
            p_line.params += " 3=" + to_string(pad);
            p_line.params += " 4=" + to_string(global_pooling);
            p_line.params += " 5=" + to_string(pad_mode);
        }
        else if (node.op() == "Min" || node.op() == "Minimum")
        {
            // check weights
            tensorflow::TensorProto tensor;
            if (find_tensor_proto(weights, node, tensor))
            {
                int operation = 5;
                int dim = 0;
                float coeff = 1.f;

                dim = parse_tensor_reduction_dim(tensor);

                p_line.params  = "0="+to_string(operation);
                p_line.params += " 1=" + to_string(dim);
                p_line.params += " 2=" + to_string(coeff);
            }
            else
            {
                int op_type = 5;
                p_line.params  = "0="+to_string(op_type);
            }
        }
        else if (node.op() == "Mul")
        {
            int op_type = 2;
            p_line.params  = "0="+to_string(op_type);
        }
        else if (node.op() == "Neg")
        {
            int op_type = 1;
            p_line.params  = "0="+to_string(op_type);
        }
        else if (node.op() == "NoOp")
        {
        }
        else if (node.op() == "Pad")
        {
            int top = 0;
            int bottom = 0;
            int left = 0;
            int right = 0;
            int type = 0;
            float value = 0.f;

            // check weights
            tensorflow::TensorProto tensor;
            if (find_tensor_proto(weights, node, tensor))
            {
                if (!tensor.tensor_content().empty() && tensor.dtype() == 3)// int32
                {
                    const int *data = reinterpret_cast<const int*>(tensor.tensor_content().c_str());
                    int size = tensor.tensor_content().size() / sizeof(int);

                    if (size == 8)
                    {
                        // n h w c
                        top = data[2];
                        bottom = data[3];
                        left = data[4];
                        right = data[5];
                    }
                }
            }

            tensorflow::AttrValue value_Tpaddings;
            if (find_attr_value(node, "Tpaddings", value_Tpaddings))
            {
                type = value_Tpaddings.i();
            }

            tensorflow::AttrValue value_T;
            if (find_attr_value(node, "T", value_T))
            {
                value = value_T.f();
            }

            p_line.params  = "0="+to_string(top);
            p_line.params += " 1=" + to_string(bottom);
            p_line.params += " 2=" + to_string(left);
            p_line.params += " 3=" + to_string(right);
            p_line.params += " 4=" + to_string(type);
            p_line.params += " 5=" + to_string(value);
        }
        else if (node.op() == "Placeholder")
        {
            int w = 0;
            int h = 0;
            int c = 0;
            tensorflow::AttrValue value;
            if (find_attr_value(node, "shape", value))
            {
                h = value.shape().dim(1).size();
                w = value.shape().dim(2).size();
                c = value.shape().dim(3).size();
            }

            p_line.params  = "0="+to_string(w);
            p_line.params += " 1=" + to_string(h);
            p_line.params += " 2=" + to_string(c);
        }
        else if (node.op() == "Prod")
        {
            int operation = 6;
            int dim = 0;
            float coeff = 1.f;

            // check weights
            tensorflow::TensorProto tensor;
            if (find_tensor_proto(weights, node, tensor))
            {
                dim = parse_tensor_reduction_dim(tensor);
            }

            p_line.params  = "0="+to_string(operation);
            p_line.params += " 1=" + to_string(dim);
            p_line.params += " 2=" + to_string(coeff);
        }
        else if (node.op() == "Reciprocal")
        {
            int op_type = 15;
            p_line.params  = "0="+to_string(op_type);
        }
        else if (node.op() == "Relu")
        {
            float slope = 0.f;
            p_line.params  = "0="+to_string(slope);
        }
        else if (node.op() == "Reshape")
        {
            tensorflow::TensorProto tensor;
            if (find_tensor_proto(weights, node, tensor))
            {
                if (!tensor.tensor_content().empty() && tensor.dtype() == 3)// int32
                {
                    const int* data = reinterpret_cast<const int*>(tensor.tensor_content().c_str());
                    int size = tensor.tensor_content().size() / sizeof(int);

                    // n h w c
                    // n h w
                    // n w
                    if (size == 4)
                    {
                        p_line.params  = "0="+to_string(data[2]);
                        p_line.params  = " 1="+to_string(data[1]);
                        p_line.params  = " 2="+to_string(data[3]);
                        p_line.params  = " 3=0";
                    }
                    if (size == 3)
                    {
                        p_line.params  = "0="+to_string(data[2]);
                        p_line.params  = " 1="+to_string(data[1]);
                        p_line.params  = " 2=-233";
                        p_line.params  = " 3=1";
                    }
                    if (size == 2)
                    {
                        p_line.params  = "0="+to_string(data[1]);
                        p_line.params  = " 1=-233";
                        p_line.params  = " 2=-233";
                        p_line.params  = " 3=1";
                    }
                }
            }
            else
            {
                // pass through
                p_line.params  = "0=0 1=0 2=0 3=0";
            }
        }
        else if (node.op() == "Rsqrt")
        {
            int op_type = 6;
            p_line.params  = "0="+to_string(op_type);
        }
        else if (node.op() == "Sigmoid")
        {
        }
        else if (node.op() == "Softmax")
        {
        }
        else if (node.op() == "Square")
        {
            int op_type = 4;
            p_line.params  = "0="+to_string(op_type);
        }
        else if (node.op() == "Squeeze")
        {
            int squeeze_w = 0;
            int squeeze_h = 0;
            int squeeze_c = 0;

            tensorflow::AttrValue value_squeeze_dims;
            if (find_attr_value(node, "squeeze_dims", value_squeeze_dims))
            {
                for (int i = 0; i<value_squeeze_dims.list().i_size(); i++)
                {
                    int dim = value_squeeze_dims.list().i(i);
                    if (dim == 0)
                        squeeze_w = 1;
                    if (dim == 1)
                        squeeze_h = 1;
                    if (dim == 2)
                        squeeze_c = 1;
                }
            }

            p_line.params  = "0="+to_string(squeeze_w);
            p_line.params += " 1=" + to_string(squeeze_h);
            p_line.params += " 2=" + to_string(squeeze_c);
        }
        else if (node.op() == "Sub")
        {
            int op_type = 1;
            p_line.params  = "0="+to_string(op_type);
        }
        else if (node.op() == "Sum")
        {
            int operation = 0;
            int dim = 0;
            float coeff = 1.f;

            // check weights
            tensorflow::TensorProto tensor;
            if (find_tensor_proto(weights, node, tensor))
            {
                dim = parse_tensor_reduction_dim(tensor);
            }

            p_line.params  = "0="+to_string(operation);
            p_line.params += " 1=" + to_string(dim);
            p_line.params += " 2=" + to_string(coeff);
        }
        else
        {
            const google::protobuf::Map<std::string, tensorflow::AttrValue>& attr = node.attr();

            google::protobuf::Map<std::string, tensorflow::AttrValue>::const_iterator it = attr.begin();
            for (; it != attr.end(); it++)
            {
                std::cerr << it->first << " #" << it->second.type() << std::endl;
            }
        }

        p_line.bottoms = p_line.name;
        p_line.output_names.insert(p_line.name);
        param_lines.push_back(p_line);

        std::string output_name = p_line.name;
        if (node_reference.find(output_name) != node_reference.end())
        {
            int refcount = node_reference[output_name];
            if (refcount > 1)
            {
                ParamLine p_line_split;
                p_line_split.op = "Split";
                p_line_split.name = "splitncnn_"+to_string(internal_split);
                p_line_split.input_size = 1;
                p_line_split.output_size = refcount;
                p_line_split.tops = output_name;
                p_line_split.input_names.insert(output_name);
                for (int j=0; j<refcount; j++)
                {
                    if(j == 0){
                        p_line_split.bottoms += output_name+"_splitncnn_"+to_string(j);
                    } else {
                        p_line_split.bottoms += " " + output_name+"_splitncnn_"+to_string(j);
                    }
                    p_line_split.output_names.insert(output_name+"_splitncnn_"+to_string(j));
                }
                param_lines.push_back(p_line_split);
                internal_split++;
            }
        }
    }

    vector<ParamLine> param_lines_merged;//op merge
    for(size_t i=0; i<param_lines.size(); i++){
        ParamLine p = param_lines[i];
        //only support conv 1x1s1 1x1s2 3x3s1 3x3s2
        string flag = p.op+"_"+to_string(p.kw)+"_"+to_string(p.kh)+"_"+to_string(p.sw)+"_"+to_string(p.sh);
        if(flag == "Convolution_1_1_1_1"
            || flag == "Convolution_1_1_2_2"
            || flag == "Convolution_3_3_1_1"
            || flag == "Convolution_3_3_2_2"
            || flag == "ConvolutionDepthWise_1_1_1_1"
            || flag == "ConvolutionDepthWise_1_1_2_2"
            || flag == "ConvolutionDepthWise_3_3_1_1"
            || flag == "ConvolutionDepthWise_3_3_2_2"){
            if(i+1<param_lines.size()){
                ParamLine next = param_lines[i+1];
                if(next.op == "BatchNorm"){
                    bool hasRelu = false;
                    if(i+2<param_lines.size()) {
                        ParamLine next_next = param_lines[i + 2];
                        if(next_next.op == "ReLU"){
                            p.op = p.op+"BNRelu";
                            p.name = next_next.name;
                            p.bottoms = next_next.bottoms;
                            p.output_names = next_next.output_names;
                            i = i+2;
                            hasRelu = true;
                            //printf("merge %s   %s\n", p.name.c_str(), p.op.c_str());
                        }
                    }
//                    if(!hasRelu){
//                        p.op = p.op+"BN";
//                        p.name = next.name;
//                        p.bottoms = next.bottoms;
//                        p.output_names = next.output_names;
//                        i = i+1;
//                    }
                } else if(next.op == "ReLU"){
//                    p.op = p.op+"Relu";
//                    p.name = next.name;
//                    p.bottoms = next.bottoms;
//                    p.output_names = next.output_names;
//                    i = i+1;
                }
            }
        }
        param_lines_merged.push_back(p);
    }

    std::set<std::string> blob_names;
    for(size_t i=0; i<param_lines_merged.size(); i++) {
        ParamLine p = param_lines_merged[i];
        for (set<string>::iterator it = p.input_names.begin(); it != p.input_names.end(); ++it)
        {
            blob_names.insert(*it);
        }
        for (set<string>::iterator it = p.output_names.begin(); it != p.output_names.end(); ++it)
        {
            blob_names.insert(*it);
        }
    }
    fprintf(pp, "%lu %lu\n", param_lines_merged.size(), blob_names.size());

    for(size_t i=0; i<param_lines_merged.size(); i++){
        ParamLine p = param_lines_merged[i];
        fprintf(pp, "%-28s", p.op.c_str());
        fprintf(pp, " %-32s", p.name.c_str());
        fprintf(pp, " %d", p.input_size);
        fprintf(pp, " %d", p.output_size);
        if(!p.tops.empty()){
            fprintf(pp, " %s", p.tops.c_str());
        }
        if(!p.bottoms.empty()) {
            fprintf(pp, " %s", p.bottoms.c_str());
        }
        if(!p.params.empty()) {
            fprintf(pp, " %s", p.params.c_str());
        }
        fprintf(pp, "\n");
    }

    fclose(pp);
    fclose(bp);

    printf("done\n");

    return 0;
}
