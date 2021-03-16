import torch

from utils import _init_paths
from fair.model import load_model
from types import MethodType
import onnx
from torch.onnx import OperatorExportTypes
from collections import OrderedDict
from models.dlav0 import get_pose_net

from dcn.functions.deform_conv import ModulatedDeformConvFunction

from onnx_tf.backend import prepare


heads = {'hm': 1,
            'wh': 4,
            'reg': 2,
            'id': 128,}
model = get_pose_net(34, heads, 256)

load_model(model, '../pretrained/dlav0_30.pth')
model.eval()
model.cuda()

# print("================================================================================")
# print(torch.ops)
# print("================================================================================")

input = torch.zeros([1, 3, 608, 1088]).cuda()
torch.onnx.export(model, input, "../pretrained/dlav0.onnx", verbose=True, opset_version=9,)

onnx.checker.check_model("../pretrained/dlav0.onnx")

onnx_model = onnx.load("../pretrained/dlav0.onnx")  # load onnx model

tf_rep = prepare(onnx_model)  # prepare tf representation
tf_rep.export_graph("../pretrained/dlav0")  # export the model