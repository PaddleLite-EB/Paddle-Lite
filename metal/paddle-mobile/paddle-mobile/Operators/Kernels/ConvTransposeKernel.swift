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

import Foundation

struct MetalConvTransposeParam {
  let kernelW: UInt16;
  let kernelH: UInt16;
  
  let strideX: UInt16;
  let strideY: UInt16;
  
  let paddingX: UInt16;
  let paddingY: UInt16;
  
  let dilationX: UInt16;
  let dilationY: UInt16;
}

class ConvTransposeKernel<P: PrecisionType>: Kernel, Computable{
  var metalParam: MetalConvTransposeParam!
  required init(device: MTLDevice, param: ConvTransposeParam<P>) {
    super.init(device: device, inFunctionName: "conv_transpose")
    let kernelWidth = UInt16(param.filter.width)
    let kernelHeight = UInt16(param.filter.height)
    
    let strideX = UInt16(param.stride[0])
    let strideY = UInt16(param.stride[1])
    let paddingX = UInt16(param.paddings[0])
    let paddingY = UInt16(param.paddings[1])
    let dilationX = UInt16(param.dilations[0])
    let dilationY = UInt16(param.dilations[1])
    
    metalParam = MetalConvTransposeParam.init(kernelW: kernelWidth, kernelH: kernelHeight, strideX: strideX, strideY: strideY, paddingX: paddingX, paddingY: paddingY, dilationX: dilationX, dilationY: dilationY)
    
    param.output.initTexture(device: device, inTranspose: param.input.transpose)
  }
  
  func compute(commandBuffer: MTLCommandBuffer, param: ConvTransposeParam<P>) throws {
    guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
      throw PaddleMobileError.predictError(message: " encoder is nil")
    }
    
    encoder.setTexture(param.input.metalTexture, index: 0)
    encoder.setTexture(param.output.metalTexture, index: 1)
    encoder.setBytes(&metalParam, length: MemoryLayout<MetalConvTransposeParam>.size, index: 0)
    encoder.setBuffer(param.filter.buffer, offset: 0, index: 1)
    encoder.dispatch(computePipline: pipline, outTexture: param.output.metalTexture)
    encoder.endEncoding()
  }
}


