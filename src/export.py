import torch

from utils import _init_paths
from fair.model import load_model
from types import MethodType
import onnx
from torch.onnx import OperatorExportTypes
from collections import OrderedDict
# from models.pose_dla_dcn import get_pose_net

from dcn.functions.deform_conv import ModulatedDeformConvFunction

# from onnx_tf.backend import prepare

torch.cuda.empty_cache()
heads = {'hm': 1,
            'wh': 4,
            'reg': 2,
            'id': 128,}
model = get_pose_net(34, heads, 256)

load_model(model, '../pretrained/fairmot_dla34.pth')
model.eval()
model.cuda()

print("================================================================================")
print(model)
print("================================================================================")

input = torch.zeros([1, 3, 608, 1088]).cuda()
# torch.onnx.export(model, input, "../pretrained/dla34.onnx", verbose=True, opset_version=9,)

# onnx.checker.check_model("../pretrained/dla34.onnx")

# onnx_model = onnx.load("../pretrained/dlav0.onnx")  # load onnx model

# tf_rep = prepare(onnx_model)  # prepare tf representation
# tf_rep.export_graph("../pretrained/dlav0")  # export the model