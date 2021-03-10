import argparse
import os
import sys

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
        self.parser.add_argument('--lr_step', type=str, default='20',
                                help='drop learning rate by 10.')
        self.parser.add_argument('--num_epochs', type=int, default=30,
                                help='total training epochs.')
        self.parser.add_argument('--batch_size', type=int, default=12,
                                help='batch size')

        self.parser.add_argument('--val_intervals', type=int, default=5,
                                help='number of epochs to run validation.')
        self.parser.add_argument('--trainval', action='store_true',
                                help='include validation in training and '
                                        'test on test set')
        
        # test
        self.parser.add_argument('--K', type=int, default=500,
                                help='max number of output objects.') 
        
        
        self.parser.add_argument('--down_ratio', type=int, default=4,
                            help='output stride. Currently only supports 4.')
        
        # loss
        self.parser.add_argument('--mse_loss', action='store_true',
                                help='use mse loss or focal loss to train '
                                    'keypoint heatmaps.')

        self.parser.add_argument('--reg_loss', default='l1',
                                help='regression loss: sl1 | l1 | l2')
        self.parser.add_argument('--hm_weight', type=float, default=1,
                                help='loss weight for keypoint heatmaps.')
        self.parser.add_argument('--off_weight', type=float, default=1,
                                help='loss weight for keypoint local offsets.')
        self.parser.add_argument('--wh_weight', type=float, default=0.1,
                                help='loss weight for bounding box size.')
        self.parser.add_argument('--id_loss', default='ce',
                                help='reid loss: ce | triplet')
        self.parser.add_argument('--id_weight', type=float, default=1,
                                help='loss weight for id')
        
        # self.parser.add_argument('--reid_dim', type=int, default=128,
        #                         help='feature dim for reid')
        
        self.parser.add_argument('--ltrb', default=True,
                                help='regress left, top, right, bottom of bbox')

        # self.parser.add_argument('--norm_wh', action='store_true',
        #                         help='L1(\hat(y) / y, 1) or L1(\hat(y), y)')
        # self.parser.add_argument('--dense_wh', action='store_true',
        #                         help='apply weighted regression near center or '
        #                             'just apply regression on center point.')
        # self.parser.add_argument('--cat_spec_wh', action='store_true',
        #                         help='category specific bounding box size.')
        # self.parser.add_argument('--not_reg_offset', action='store_true',
        #                         help='not regress local offset.')

    def parse(self):
        return self.parser.parse_args()