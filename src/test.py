import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms
import matplotlib.pyplot as plt

from utils import _init_paths

# from models.dlav0 import get_pose_net
from opts import opts

from fair.dataset import JointDataset, LoadImages
from models.mot_trainer import MotTrainer

from fair.decode import mot_decode

from fair.model import *

if __name__ == '__main__':
    torch.cuda.empty_cache()
    
    opt = opts().parse()
    
    # model = get_pose_net(34, opt.heads, 256)
    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    model = load_model(model, opt.load_model)
    model = model.to(opt.device)
    model.eval()

    with open("./datacfg.json", 'r') as f:
        import json
        data_config = json.load(f)
        trainset_paths = data_config['train']
        dataset_root = data_config['root']

    dataset = LoadImages("/home/hardys/Desktop/000058.jpg")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )

    for iter_id, (img_path, img, img0) in enumerate(dataloader):
        plt.subplot(1,2,1)
        plt.imshow(img0[0])
        
        
        output = model(img.cuda())[-1]
        print(output.keys())
        print(output['hm'].shape)
        
        pred_hm = output['hm'].cpu().detach().numpy()[0]
        pred_hm = np.moveaxis(pred_hm, 0, -1)
        print(np.min(pred_hm), np.max(pred_hm))

        plt.subplot(1,2,2)
        plt.imshow(pred_hm)
        plt.show()
