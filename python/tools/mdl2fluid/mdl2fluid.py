import json
import framework_pb2 as framework_pb2
import op_types as types


def load_mdl(mdl_json_path):
    # print('mdl json path : ' + mdl_json_path)
    with open(mdl_json_path, 'r') as f:
        return json.load(f)


class Converter:
    'convert mdlmodel to fluidmodel'

    def __init__(self, mdl_json_path):
        self.mdl_json_path = mdl_json_path
        self.mdl_json = load_mdl(self.mdl_json_path)
        self.program_desc = framework_pb2.ProgramDesc()

        # print(json_dick)
        # layers = (json_dick['layer'])
        # for layer in layers:
        #     print(layer)

    def convert(self):
        print 'convert begin.....'
        # add block_desc
        block_desc = self.program_desc.blocks.add()
        block_desc.idx = 0
        block_desc.parent_idx = -1
        self.package_ops(block_desc)
        print 'blocks: '
        print self.program_desc.blocks

    def package_ops(self, block_desc):
        # add ops with layer
        if 'layer' in self.mdl_json:

            layers_ = self.mdl_json['layer']
            for layer in layers_:
                desc_ops = block_desc.ops.add()

                # print layer
                # for i in layer:
                #     print i
                if 'name' in layer:
                    l_name = layer['name']

                if 'weight' in layer:
                    l_weights = layer['weight']

                if 'param' in layer:
                    l_params = layer['param']

                if 'output' in layer:
                    l_outputs = layer['output']

                if 'input' in layer:
                    l_inputs = layer['input']
                    inputs_add = desc_ops.inputs.add()
                    for i in l_inputs:
                        # print i
                        inputs_add.parameter = ''
                        inputs_add.arguments.append(i)

                if 'type' in layer:
                    l_type = layer['type']
                    # print l_type
                    # print mdl2fluid_op_layer_dict.get(l_type)
                    desc_ops.type = types.mdl2fluid_op_layer_dict.get(l_type)


mdl_path = "multiobjects/YOLO_Universal.json"
# print mdl_path
# # model
# mdl_model = load_mdl(mdl_path)
# for key in mdl_model:
#     print key
#
# # layer
# layers = mdl_model['layer']
# print layers
#
# for layer in layers:
#     print layer
#     for i in layer:
#         print i
#     if 'name' in layer:
#         l_name = layer['name']
#
#     if 'weight' in layer:
#         l_weights = layer['weight']
#
#     if 'param' in layer:
#         l_params = layer['param']
#
#     if 'output' in layer:
#         l_outputs = layer['output']
#
#     if 'input' in layer:
#         l_inputs = layer['input']
#
#     if 'type' in layer:
#         l_type = layer['type']
#
# print mdl_model['matrix']
#
# package()


converter = Converter(mdl_path)
converter.convert()
