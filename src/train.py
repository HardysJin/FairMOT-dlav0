import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms
import matplotlib.pyplot as plt

from utils import _init_paths

from models.dlav0 import get_pose_net
from opts import opts

from fair.dataset import JointDataset
from models.mot_trainer import MotTrainer

from fair.decode import mot_decode

from fair.model import load_model, save_model


if __name__ == '__main__':
    torch.cuda.empty_cache()
    
    opt = opts().init()

    model = get_pose_net(34, opt.heads, 256)

    # print(model)
    with open("./datacfg.json", 'r') as f:
        import json
        data_config = json.load(f)
        trainset_paths = data_config['train']
        dataset_root = data_config['root']

    T = transforms.Compose([transforms.ToTensor()])
    dataset = JointDataset(opt, dataset_root, trainset_paths, img_size=(1088, 608), augment=False, transforms=T)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
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
        
        
    save_model(os.path.join(opt.save_dir, 'mot_{}.pth'.format(epoch)), epoch, model, optimizer)
        
        
    # for iter_id, batch in enumerate(dataloader):
    #     for i, (k, v) in enumerate(batch.items()):
    #         print(k, v.shape)
    #     plt.subplot(2,2,1)
    #     img = np.squeeze(batch['input'].numpy()[0])
    #     img = np.moveaxis(img, 0, -1)
    #     plt.imshow(img)
    #     plt.subplot(2,2,2)
    #     img2 = batch['hm'].numpy()[0]
    #     img2 = np.moveaxis(img2, 0, -1)
    #     plt.imshow(img2)
        
        
    #     input = batch['input'].cuda()
    #     model.cuda()
    #     output = model(input)[-1]
    #     print(output.keys())
        
    #     dets, inds = mot_decode(output['hm'].sigmoid_(), output['wh'], reg=None, ltrb=opt.ltrb, K=opt.K)
        
    #     print(dets.shape, inds.shape)
        
    #     pred_hm = output['hm'].cpu().detach().numpy()[0]
    #     pred_hm = np.moveaxis(pred_hm, 0, -1)
    #     plt.subplot(2,2,3)
    #     plt.imshow(pred_hm)
        
    #     plt.show()
    #     break


    # torch.save(model.cpu().state_dict(), "../pretrained/final_mot.pth")
    # dummy_input = torch.randn(1, 3, 608, 1088, device='cuda')
    # torch.onnx.export(model, dummy_input, "../pretrained/final_mot.onnx", verbose=True)
