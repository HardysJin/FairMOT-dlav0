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

def convert(pth, _onnx, pb):
    # print(torch.__version__)
    name='mot'

    heads = {'hm': 1,
                'wh': 4,
                'reg': 2,
                'id': 128,}
    model = get_pose_net(34, heads, 256)
    
    load_model(model, pth)
    
    model.cuda()
    dummy_input = torch.randn(1, 3, 608, 1088, device='cuda')

    output_names = [ 'hm' , 'wh', 'reg', 'id']
    torch.onnx.export(model, dummy_input, _onnx, verbose=True, output_names=output_names, operator_export_type=OperatorExportTypes.ONNX, opset_version=9,)
    # print(model)
    
    onnx.checker.check_model(_onnx)

    onnx_model = onnx.load(_onnx)  # load onnx model
    
    tf_rep = prepare(onnx_model)  # prepare tf representation
    tf_rep.export_graph(pb)  # export the model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth', type=str, default='../pretrained/dlav0_30.pth', help="pth path to be converted")
    parser.add_argument('--onnx', type=str, default='../pretrained/dlav0.onnx', help="onnx path")
    parser.add_argument('--pb', type=str, default='../pretrained/dlav0_30', help="pb path")
    arg = parser.parse_args()
    assert arg.pth != '', 'pth path not defined'
    assert arg.onnx != '', 'onnx path not defined'
    assert arg.pb != '', 'pb path not defined'
    
    convert(arg.pth, arg.onnx, arg.pb)