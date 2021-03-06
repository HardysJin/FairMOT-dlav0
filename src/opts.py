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
    def parse(self):
        return self.parser.parse_args()