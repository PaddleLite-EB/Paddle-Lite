/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#ifdef FUSION_FCRELU_OP
#include "operators/kernel/fc_relu_kernel.h"
#include "fpga/api/fpga_api.h"

namespace paddle_mobile {
    namespace operators {

        template <>
        bool FusionFcReluKernel<FPGA, float>::Init(FusionFcReluParam *param) {
            bool relu_enabled = true;
            bool bn_enabled = false;
            const Tensor *input_x = param->InputX();
            auto input_x_ptr = input_x->data<float>();
            const Tensor *input_y = param->InputY();
            auto input_y_ptr = input_y->data<float>();
            const Tensor *input_z = param->InputZ();
            auto input_z_ptr = input_z->data<float>();
            Tensor *out = param->Out();
            auto out_ptr = out->mutable_data<float>();


            fpga::ConvArgs convArgs;
            convArgs.relu_enabled = relu_enabled;
            convArgs.bias_address = (void*)input_z_ptr;
            convArgs.filter_address = (void*)input_y_ptr;
            convArgs.filter_num = out->dims()[1];
            convArgs.group_num = 1;
            convArgs.bn.enabled = bn_enabled;
            convArgs.kernel.stride_w = 1;
            convArgs.kernel.stride_h = 1;
            convArgs.kernel.height = input_x->dims()[2];
            convArgs.kernel.width = input_x->dims()[3];
            convArgs.image.address = (void*)input_x_ptr;
            convArgs.image.channels = input_x->dims()[1];
            convArgs.image.height = input_x->dims()[2];
            convArgs.image.width = input_x->dims()[3];
            convArgs.image.pad_height = 1;
            convArgs.image.pad_width = 1;
            convArgs.image.scale_address = nullptr;//fc input has scale attribute??
            convArgs.output.address = (void*)out_ptr;
            convArgs.output.scale_address = nullptr;//fc output has scale attribute??
            param->SetFpgaArgs(convArgs);

            return true;
        }
        template <>
        void FusionFcReluKernel<FPGA, float>::Compute(const FusionFcReluParam &param) const {
            fpga::ComputeFpgaConv(param.FpgaArgs());
        };

    }  // namespace operators
}  // namespace paddle_mobile
#endif