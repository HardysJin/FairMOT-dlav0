import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms
import matplotlib.pyplot as plt

from utils import _init_paths

from models.pose_dla_dcn import get_pose_net
from opts import opts
if __name__ == '__main__':
    torch.cuda.empty_cache()
    
    args = opts().parse()
    print(args)
    heads = {'hm': 1,
             'wh': 4,
             'reg': 2,
             'id': 128,}
    model = get_pose_net(34, heads, 256).cuda()
    
    torch.save(model.cpu().state_dict(), "../pretrained/final_mot.pth")
    
    dummy_input = torch.randn(1, 3, 608, 1088, device='cuda')
    torch.onnx.export(model, dummy_input, "../pretrained/final_mot.onnx", verbose=True)

    print(model)