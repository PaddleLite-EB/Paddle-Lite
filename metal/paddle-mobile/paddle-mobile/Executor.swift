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

public class Executor<P: PrecisionType> {
    var ops: [Runable] = []
    public init(program: Program) throws {
        for block in program.programDesc.blocks {
            for varDesc in block.vars {
                if !varDesc.persistable {
                    program.scope.vars[varDesc.name] = Texture.init()
                }
            }
            for op in block.ops {
                do {
                    let op = try OpCreator<P>.shared.creat(opDesc: op, scope: program.scope)
                    ops.append(op)
                } catch let error {
                    throw error
                }
            }
        }
    }
    
    public func predict() {
        for op in ops {
            op.run()
        }
    }
}

//public let paddle_executor: Executor = Executor.init()
