import torch
import argparse
import os
import sys
from types import MethodType
from utils import _init_paths

# from models.dlav0 import get_pose_net
from fair.model import create_model, load_model

from torch.onnx import OperatorExportTypes
import onnx

from onnxToCaffe.convertCaffe import onnxToCaffe

def convert(pth, _onnx, cp, cm):
    # print(torch.__version__)
    name='mot'

    heads = {'hm': 1,
                'wh': 4,
                'reg': 2,
                'id': 128,}
    # model = get_pose_net(34, heads, 256)
    model = create_model('dlav0_34', heads, 256)

    load_model(model, pth)
    
    model.cuda()
    dummy_input = torch.randn(1, 3, 608, 1088, device='cuda')

    output_names = [ 'hm' , 'wh', 'reg', 'id']
    torch.onnx.export(model, dummy_input, _onnx, verbose=True, output_names=output_names, operator_export_type=OperatorExportTypes.ONNX, opset_version=9,)
    
    onnx.checker.check_model(_onnx)
    
    onnxToCaffe(_onnx, cp, cm)

    # onnx_model = onnx.load(_onnx)  # load onnx model
    
    # tf_rep = prepare(onnx_model)  # prepare tf representation
    # tf_rep.export_graph(pb)  # export the model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth', type=str, default='../pretrained/dlav0.pth', help="pth path to be converted")
    parser.add_argument('--onnx', type=str, default='../pretrained/dlav0.onnx', help="onnx path")
    parser.add_argument('-cp', '--caffe_prototxt', type=str, default='../pretrained/dlav0.prototxt', help="caffe_prototxt path")
    parser.add_argument('-cm', '--caffe_caffemodel', type=str, default='../pretrained/dlav0.caffemodel', help="caffe_caffemodel path")
    arg = parser.parse_args()
    print(arg)
    assert arg.pth != '', 'pth path not defined'
    assert arg.onnx != '', 'onnx path not defined'
    assert arg.caffe_prototxt != '', 'caffe_prototxt path not defined'
    assert arg.caffe_caffemodel != '', 'caffe_caffemodel path not defined'
    
    convert(arg.pth, arg.onnx, arg.caffe_prototxt, arg.caffe_caffemodel)