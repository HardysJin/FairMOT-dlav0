import argparse
import os
import sys
import torch

class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # basic experiment setting
        # self.parser.add_argument('task', default='mot', help='mot')
        self.parser.add_argument('--exp_id', default='default')
        self.parser.add_argument('--test', action='store_true')
        self.parser.add_argument('--load_model', default='',
                                help='path to pretrained model')
        self.parser.add_argument('--resume', action='store_true',
                                help='resume an experiment. '
                                        'Reloaded the optimizer parameter and '
                                        'set load_model to model_last.pth '
                                        'in the exp dir if load_model is empty.') 
        self.parser.add_argument('--num_workers', type=int, default=8,
                                help='dataloader threads. 0 for single-thread.')
        # train
        self.parser.add_argument('--lr', type=float, default=1e-4,
                                help='learning rate for batch size 12.')
        self.parser.add_argument('--lr_step', type=str, default='20,25',
                                help='drop learning rate by 10.')
        self.parser.add_argument('--num_epochs', type=int, default=30,
                                help='total training epochs.')
        self.parser.add_argument('--batch_size', type=int, default=3,
                                help='batch size')
        self.parser.add_argument('--num_iters', type=int, default=-1,
                                help='default: #samples / batch_size.')

        self.parser.add_argument('--val_intervals', type=int, default=5,
                                help='number of epochs to run validation.')
        self.parser.add_argument('--trainval', action='store_true',
                                help='include validation in training and '
                                        'test on test set')
        
        self.parser.add_argument('--K', type=int, default=500,
                                help='max number of output objects.') 
        
        self.parser.add_argument('--down_ratio', type=int, default=4,
                            help='output stride. Currently only supports 4.')
        
        # loss
        self.parser.add_argument('--mse_loss',  action='store_true',
                                help='define heatmap loss function, True: MSELoss; False: FocalLoss')
        self.parser.add_argument('--hm_weight', type=float, default=1,
                                help='loss weight for keypoint heatmaps.')
        
        # self.parser.add_argument('--reg_loss', default='l1',
        #                         help='regression loss: sl1 | l1 | l2')
        self.parser.add_argument('--off_weight', type=float, default=1,
                                help='loss weight for keypoint local offsets.')
        
        self.parser.add_argument('--wh_weight', type=float, default=0.1,
                                help='loss weight for bounding box size.')
        
        # self.parser.add_argument('--id_loss', default='ce',
        #                         help='reid loss: ce | triplet')
        
        self.parser.add_argument('--id_weight', type=float, default=1,
                                help='loss weight for id')
        
        self.parser.add_argument('--ltrb', default=True,
                                help='regress left, top, right, bottom of bbox')
        
        self.parser.add_argument('--reid_dim', type=int, default=128,
                                 help="Output channels of Re-ID head")
        
        self.parser.add_argument('--print_iter', type=int, default=20, 
                                help='disable progress bar and print to screen.')
        
        self.parser.add_argument('--hide_data_time', action='store_true',
                                help='not display time during training.')

        self.parser.add_argument('--norm_wh', action='store_true',
                                help='L1(\hat(y) / y, 1) or L1(\hat(y), y)')
        self.parser.add_argument('--dense_wh', action='store_true',
                                help='apply weighted regression near center or '
                                    'just apply regression on center point.')
        self.parser.add_argument('--cat_spec_wh', action='store_true',
                                help='category specific bounding box size.')
        self.parser.add_argument('--not_reg_offset', action='store_true',
                                help='not regress local offset.')
        
        # tracking
        self.parser.add_argument('--test_mot16', default=False, help='test mot16')
        self.parser.add_argument('--val_mot15', default=False, help='val mot15')
        self.parser.add_argument('--test_mot15', default=False, help='test mot15')
        self.parser.add_argument('--val_mot16', default=False, help='val mot16 or mot15')
        self.parser.add_argument('--test_mot17', default=False, help='test mot17')
        self.parser.add_argument('--val_mot17', default=False, help='val mot17')
        self.parser.add_argument('--val_mot20', default=False, help='val mot20')
        self.parser.add_argument('--test_mot20', default=False, help='test mot20')
        self.parser.add_argument('--val_hie', default=False, help='val hie')
        self.parser.add_argument('--test_hie', default=False, help='test hie')
        self.parser.add_argument('--conf_thres', type=float, default=0.4, help='confidence thresh for tracking')
        self.parser.add_argument('--det_thres', type=float, default=0.3, help='confidence thresh for detection')
        self.parser.add_argument('--nms_thres', type=float, default=0.4, help='iou thresh for nms')
        self.parser.add_argument('--track_buffer', type=int, default=30, help='tracking buffer')
        self.parser.add_argument('--min-box-area', type=float, default=100, help='filter out tiny boxes')
        self.parser.add_argument('--input-video', type=str,
                                default='../videos/MOT16-03.mp4',
                                help='path to the input video')
        self.parser.add_argument('--output-format', type=str, default='video', help='video or text')
        self.parser.add_argument('--output-root', type=str, default='../outputs', help='expected output root path')

    def init(self):
        opt = self.parser.parse_args()
        
        opt.heads = {'hm': 1,
             'wh': 4,
             'reg': 2,
             'id': opt.reid_dim,
        }
        
        opt.reg_offset = not opt.not_reg_offset
        opt.num_stacks = 1
        
        opt.save_dir = "../pretrained"
        opt.lr_step = [int(i) for i in opt.lr_step.split(',')]
        if opt.resume:
            assert opt.load_model != "", "Trying to resume but weights path NOT defined"
        opt.gpus = [0] # add more gpu here
        opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
        dataset_cfg = {'default_resolution': [608, 1088], 'num_classes': 1,
                    'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278], 'nID': 14455}

        opt.default_resolution = dataset_cfg['default_resolution']
        opt.num_classes = dataset_cfg['num_classes']
        opt.mean = dataset_cfg['mean']
        opt.std = dataset_cfg['std']
        opt.nID = dataset_cfg['nID']
        return opt