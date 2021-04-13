import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms
import matplotlib.pyplot as plt

from utils import _init_paths

from opts import opts
from fair.dataset import JointDataset
from fair.decode import mot_decode
from fair.model import create_model, load_model, save_model

from models.mot_trainer import MotTrainer


if __name__ == '__main__':
    torch.cuda.empty_cache()
    
    opt = opts().parse()
    
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

    with open("./datacfg.json", 'r') as f:
        import json
        data_config = json.load(f)
        trainset_paths = data_config['train']
        dataset_root = data_config['root']

    T = transforms.Compose([transforms.ToTensor()])
    dataset = JointDataset(opt, dataset_root, trainset_paths, img_size=(1088, 608), augment=True, transforms=T)
    opt = opts().update_dataset_info_and_set_heads(opt, dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    
    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    
    trainer = MotTrainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, [opt.batch_size], opt.device)
    start_epoch = 1
    # model.load_state_dict(torch.load("../pretrained/fairmot_dla34.pth"))
    if opt.resume:
        model, optimizer, start_epoch = load_model(model, opt.load_model, trainer.optimizer, opt.resume, opt.lr, opt.lr_step)
    
    
    for epoch in range(start_epoch, opt.num_epochs + 1):
        log_dict_train, _ = trainer.train(epoch, dataloader)
        
        if epoch in opt.lr_step:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if epoch % 5 == 0 or epoch >= 25:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)
        
        
    save_model(os.path.join(opt.save_dir, '{}_{}.pth'.format(opt.exp_id, epoch)), epoch, model, optimizer)