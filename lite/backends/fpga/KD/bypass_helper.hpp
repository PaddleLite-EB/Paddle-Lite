#pragma once
#include "lite/backends/fpga/KD/pes/cpu_pe.hpp"
#include "lite/backends/fpga/KD/pes/bypass_pe.hpp"
#include "lite/backends/fpga/KD/pe_params.hpp"


class BypassHelper {
public:
	BypassParam& out_param() {
		return bypass_out_pe_.param();
	}

	BypassParam& in_param() {
		return bypass_in_pe_.param();
	}


	void PrepareForRun(IoCopyParam& param) {
		BypassParam& out_param = bypass_out_pe_->param();

	    if (param.x->ZynqTensor()->aligned() &&
	        param.x->ZynqTensor()->shape().shouldAlign()) {
		    aligned_input_.mutableData<float16>(zynqmp::FP16,
	                                       param.x->ZynqTensor()->shape());
			bypass_align_.reset(new BypassPE());

			BypassParam& input_param = bypass_align_->param();
			input_param.input = param.x->ZynqTensor();
			input_param.output = &aligned_input_;
			bypass_align_->init();
			bypass_align_->apply();

			cpu_pe_.init();
			cpu_pe_.apply();

			out_param.input = &aligned_input_;
	    } else {
	      out_param.input = param.x->ZynqTensor();
	    }
	    out_param.output = param.y->ZynqTensor();
	    bypass_out_pe_->init();
	    bypass_out_pe_->apply();
	}

	void Run() {
		if (bypass_align_) {
			bypass_align_->dispatch();
			cpu_pe_.dispatch();
			aligned_input_.setAligned(true);
			aligned_input_.unalignImage();
		}

	    bypass_out_pe_.dispatch();
	}


private:
  zynqmp::Tensor aligned_input_;

  CPUPE cpu_pe_;
  std::unique_ptr<BypassPE> bypass_align_;
  std::unique_ptr<BypassPE> bypass_pe_;

 };