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

from jde.dataset import JointDataset

if __name__ == '__main__':
    torch.cuda.empty_cache()
    
    args = opts().parse()
    print(args)
    heads = {'hm': 1,
             'wh': 4,
             'reg': 2,
             'id': 128,}
    model = get_pose_net(34, heads, 256).cuda()

    with open("./datacfg.json", 'r') as f:
        import json
        data_config = json.load(f)
        trainset_paths = data_config['train']
        dataset_root = data_config['root']

    T = transforms.Compose([transforms.ToTensor()])
    dataset = JointDataset(args, dataset_root, trainset_paths, img_size=(1088, 608), augment=False, transforms=T)
    
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    
    start_epoch = 1
    # for epoch in range(start_epoch, opt.num_epochs + 1):
        
    for iter_id, batch in enumerate(train_loader):
        # print()
        img = np.squeeze(batch['input'].numpy()[0])
        img = np.moveaxis(img, 0, -1)
        
        plt.imshow(img)
        plt.show()
        break


    # torch.save(model.cpu().state_dict(), "../pretrained/final_mot.pth")
    # dummy_input = torch.randn(1, 3, 608, 1088, device='cuda')
    # torch.onnx.export(model, dummy_input, "../pretrained/final_mot.onnx", verbose=True)