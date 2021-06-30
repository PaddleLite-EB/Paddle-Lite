// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "lite/core/mir/fusion/conv_affine_channel_fuser.h"
#include <memory>
#include <vector>

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

void ConvAffineChannelFuser::BuildPattern() {
auto* conv_input =
      VarNode("conv_input")->assert_is_op_input(conv_type_, "Input")->AsInput();
  auto* conv_weight = VarNode("conv_weight")
                          ->assert_is_op_input(conv_type_, "Filter")
                          ->AsInput();
  auto* conv = OpNode("conv2d", conv_type_)->assert_is_op(conv_type_);

  auto bn_type_ = "affine_channel";
  auto* conv_out = VarNode("conv_out")
                       ->assert_is_op_output(conv_type_, "Output")
                       ->assert_is_op_input(bn_type_, "X")
                       ->AsIntermediate();

  auto* ac_scale = VarNode("ac_scale")
                       ->assert_is_op_input(bn_type_, "Scale")
                       ->AsIntermediate();
  auto* ac_bias =
      VarNode("ac_bias")->assert_is_op_input(bn_type_, "Bias")->AsInput();
  
  auto* ac = OpNode("ac", bn_type_)->assert_is_op(bn_type_)->AsIntermediate();

  auto* ac_out =
      VarNode("ac_out")->assert_is_op_output(bn_type_, "Out")->AsOutput();

  conv->LinksFrom({conv_input, conv_weight}).LinksTo({conv_out});
  ac->LinksFrom({conv_out, ac_scale, ac_bias}).LinksTo({ac_out});

}

void ConvAffineChannelFuser::InsertNewNode(SSAGraph* graph,
                                        const key2nodes_t& matched) {
  VLOG(4) << "========================== InsertNewNode ==========================";
  auto conv_instruct = matched.at("conv2d")->stmt();
  auto conv_op_desc = conv_instruct->mutable_op_info();
  auto conv = conv_instruct->op();
  auto* scope = conv->scope();

  auto ac_scale = scope->FindVar(matched.at("ac_scale")->arg()->name)
                        ->GetMutable<lite::Tensor>();
  auto ac_scale_data = ac_scale->mutable_data<float>();

  // conv
  std::string conv_weight_name = matched.at("conv_weight")->arg()->name;
  auto conv_weight_t =
      scope->FindVar(conv_weight_name)->GetMutable<lite::Tensor>();
  auto groups = conv_op_desc->GetAttr<int>("groups");
  if (conv_type_ == "conv2d_transpose") {
    CHECK_EQ(static_cast<size_t>(ac_scale->data_size()),
             static_cast<size_t>(conv_weight_t->dims()[1] * groups))
        << "The BN bias's size should be equal to the size of the first "
        << "dim size of the conv weights";
  } else {
    CHECK_EQ(static_cast<size_t>(ac_scale->data_size()),
             static_cast<size_t>(conv_weight_t->dims()[0]))
        << "The BN bias's size should be equal to the size of the first "
        << "dim size of the conv weights";
  }
  size_t weight_num = conv_weight_t->data_size();
  bool enable_int8 = conv_op_desc->HasAttr("enable_int8") ? true : false;
  bool is_weight_quantization = conv_op_desc->HasAttr("quantize_weight_bits");

  conv_op_desc->SetType(conv_type_);
  conv_op_desc->SetInput("Input", {matched.at("conv_input")->arg()->name});
  conv_op_desc->SetInput("Filter", {matched.at("conv_weight")->arg()->name});
  conv_op_desc->SetOutput("Output", {matched.at("ac_out")->arg()->name});
  conv_op_desc->SetInput("Bias",
                         {matched.at("ac_bias")->arg()->name});  // conv_bias
  IR_NODE_LINK_TO(matched.at("ac_bias"), matched.at("conv2d"));
  
#ifdef LITE_WITH_FPGA
  conv_op_desc->SetInput("Scale",
                         {matched.at("ac_scale")->arg()->name});  // conv_sias
  IR_NODE_LINK_TO(matched.at("ac_scale"), matched.at("conv2d"));
#endif

  auto update_conv_desc = *conv_instruct->mutable_op_info();
  conv_instruct->ResetOp(update_conv_desc, graph->valid_places());

  IR_OP_VAR_LINK(matched.at("conv2d"), matched.at("ac_out"));

}

// cpp::OpDesc ConvAffineChannelFuser::GenOpDesc(const key2nodes_t& matched) {
//   cpp::OpDesc op_desc = *matched.at("conv2d")->stmt()->op_info();
//   op_desc.SetOutput("Output", {matched.at("output")->arg()->name});
//   cpp::OpDesc act_op_desc = *matched.at("act")->stmt()->op_info();


//   return op_desc;
// }

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
