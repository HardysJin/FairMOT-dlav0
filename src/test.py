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

from fair.dataset import JointDataset, LoadImages
from models.mot_trainer import MotTrainer

from fair.decode import mot_decode

from fair.model import load_model, save_model

if __name__ == '__main__':
    torch.cuda.empty_cache()
    
    opt = opts().init()
    heads = {'hm': 1,
             'wh': 4,
             'reg': 2,
             'id': opt.reid_dim,
             }
    
    model = get_pose_net(34, heads, 256)
    model = load_model(model, opt.load_model)
    model = model.to(opt.device)
    model.eval()

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
        plt.subplot(1,2,2)
        plt.imshow(pred_hm)
        plt.show()
    
    # for epoch in range(start_epoch, opt.num_epochs + 1):
    #     log_dict_train, _ = trainer.train(epoch, dataloader)
    #     print(log_dict_train)
    #     if epoch % 5 == 0 or epoch >= 25:
    #         save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
    #                    epoch, model, optimizer)
        
        
    # save_model(os.path.join(opt.save_dir, 'mot_{}.pth'.format(epoch)), epoch, model, optimizer)
        
        
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
    #     with torch.no_grad():
            
    #         # output = model(input)[-1]
    #         # print(output.keys())
            
    #         # # dets, inds = mot_decode(output['hm'].sigmoid_(), output['wh'], reg=None, ltrb=opt.ltrb, K=opt.K)
            
    #         # # print(dets.shape, inds.shape)
    #         # print(output['hm'].shape)
    #         # pred_hm = output['hm'].cpu().detach().numpy()[0]
    #         # pred_hm = np.moveaxis(pred_hm, 0, -1)
    #         # plt.subplot(2,2,3)
    #         # plt.imshow(pred_hm)
            
            
    #         output = model(input)[-1]
            
            
    #         pred_hm = output['hm'].cpu().detach().numpy()[0]
    #         pred_hm = np.moveaxis(pred_hm, 0, -1)
    #         plt.subplot(2,2,3)
    #         plt.imshow(pred_hm)
            
    #         plt.show()
        
    #     plt.show()
    #     break


    # torch.save(model.cpu().state_dict(), "../pretrained/final_mot.pth")
    # dummy_input = torch.randn(1, 3, 608, 1088, device='cuda')
    # torch.onnx.export(model, dummy_input, "../pretrained/final_mot.onnx", verbose=True)
