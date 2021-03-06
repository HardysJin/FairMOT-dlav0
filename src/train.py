import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms
import matplotlib.pyplot as plt

from models.model import get_pose_net
from opts import opts
    
if __name__ == '__main__':
    torch.cuda.empty_cache()
    
    args = opts().parse()
    print(args)
    heads = {'hm': 1,
             'wh': 4,
             'reg': 2,
             'id': 128,}
    model = get_pose_net(34, heads, 256)
    print(model)