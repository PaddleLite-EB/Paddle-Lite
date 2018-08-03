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
#ifdef FUSION_ELEMENTWISEADDRELU_OP

#include "operators/kernel/elementwise_add_relu_kernel.h"
#include "fpga/api/fpga_api.h"

namespace paddle_mobile {
    namespace operators {

        template <>
        bool ElementwiseAddReluKernel<FPGA, float>::Init(ElementwiseAddReluParam *param) {
            bool relu_enabled = true;
            const Tensor *input_x = param->InputX();
            const Tensor *input_y = param->InputY();
            Tensor *out = param->Out();
            auto input_x_ptr = input_x->data<float>();
            auto input_y_ptr = input_y->data<float>();
            auto out_ptr = out->data<float>();
            fpga::EWAddArgs ewaddArgs;
            ewaddArgs.relu_enabled = relu_enabled;
            ewaddArgs.const0 = 1;
            ewaddArgs.const1 = 1;
            ewaddArgs.image0.address = (void*)input_x_ptr;
            ewaddArgs.image0.channels = input_x->dims()[1];
            ewaddArgs.image0.scale_address = nullptr;//ew has scale attribute??
            ewaddArgs.image0.height = input_x->dims()[2];
            ewaddArgs.image0.width = input_x->dims()[3];
            ewaddArgs.image0.pad_height = 1;
            ewaddArgs.image0.pad_width = 1;
            ewaddArgs.image1.address = (void*)input_y_ptr;
            ewaddArgs.image1.channels = input_y->dims()[1];
            ewaddArgs.image1.scale_address = nullptr;//ew has scale attribute??
            ewaddArgs.image1.height = input_y->dims()[2];
            ewaddArgs.image1.width = input_y->dims()[3];
            ewaddArgs.image1.pad_height = 1;
            ewaddArgs.image1.pad_width = 1;
            ewaddArgs.output.scale_address = nullptr;
            ewaddArgs.output.address = (void*)out_ptr;
            param->SetFpgaArgs(ewaddArgs);
            return true;
        }

        template <>
        void ElementwiseAddReluKernel<FPGA, float>::Compute(const ElementwiseAddReluParam &param) const {
            fpga::ComputeFpgaEWAdd(param.FpgaArgs());
        }
    }  // namespace operators
}  // namespace paddle_mobile

#endif