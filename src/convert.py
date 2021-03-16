import torch
import argparse
import os
import sys
from types import MethodType
from utils import _init_paths

from models.pose_dla_dcn import get_pose_net
from fair.model import load_model

from torch.onnx import OperatorExportTypes
import onnx
from onnx_tf.backend import prepare

from dcn.functions.deform_conv import ModulatedDeformConvFunction

def pose_dla_forward(self, x):
    x = self.base(x)
    x = self.dla_up(x)
    y = []
    for i in range(self.last_level - self.first_level):
        y.append(x[i].clone())
    self.ida_up(y, 0, len(y))
    ret = []  ## change dict to list
    for head in self.heads:
        ret.append(self.__getattr__(head)(y[-1]))
    return ret

def convert(pth, _onnx, pb):
    
    # from torch.onnx import register_custom_op_symbolic
    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!register_custom_op_symbolic called")
    # register_custom_op_symbolic('custom_namespace::DCNv2', ModulatedDeformConvFunction.symbolic)
    
    # print(torch.__version__)
    name='mot'

    heads = {'hm': 1,
                'wh': 4,
                'reg': 2,
                'id': 128,}
    model = get_pose_net(34, heads, 256)
    
    model.forward = MethodType(pose_dla_forward, model)
    
    load_model(model, pth)
    
    model.cuda()
    dummy_input = torch.randn(1, 3, 608, 1088, device='cuda')

    output_names = [ 'hm' , 'wh', 'reg', 'id']
    torch.onnx.export(model, dummy_input, _onnx, verbose=True, output_names=output_names, operator_export_type=OperatorExportTypes.ONNX, opset_version=9,)
    # print(model)
    
    onnx_model = onnx.load(_onnx)  # load onnx model
    
    tf_rep = prepare(onnx_model)  # prepare tf representation
    tf_rep.export_graph(pb)  # export the model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth', type=str, default='', help="pth path to be converted")
    parser.add_argument('--onnx', type=str, default='', help="onnx path")
    parser.add_argument('--pb', type=str, default='', help="pb path")
    arg = parser.parse_args()
    assert arg.pth != '', 'pth path not defined'
    assert arg.onnx != '', 'onnx path not defined'
    assert arg.pb != '', 'pb path not defined'
    
    convert(arg.pth, arg.onnx, arg.pb)