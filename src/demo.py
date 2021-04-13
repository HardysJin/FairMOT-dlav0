from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils import _init_paths

import logging
import os
import os.path as osp
from opts import opts
from fair.tracking_utils.utils import mkdir_if_missing
from fair.tracking_utils.log import logger

import fair.dataset as datasets

from fair.track import eval_seq


logger.setLevel(logging.INFO)


def demo(opt):
    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdir_if_missing(result_root)
    
    new_name = os.path.basename(opt.input_video).replace(' ', '_').split('.')[0] + '_' +opt.arch
    
    logger.info('Starting tracking...')
    dataloader = datasets.LoadVideo(opt.input_video, opt.img_size)
    result_filename = os.path.join(result_root, 'results.txt')
    frame_rate = dataloader.frame_rate

    frame_dir = None if opt.output_format == 'text' else osp.join(result_root, new_name)
    eval_seq(opt, dataloader, 'mot', result_filename,
             save_dir=frame_dir, show_image=False, frame_rate=frame_rate,
             use_cuda=opt.gpus!=[-1])

    if opt.output_format == 'video':
        output_video_path = osp.join(result_root, new_name + '.mp4')
        cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}'.format(osp.join(result_root, new_name), output_video_path)
        os.system(cmd_str)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()
    print(opt)
    # result_root = opt.output_root if opt.output_root != '' else '.'
    # new_name = os.path.basename(opt.input_video).replace(' ', '_').split('.')[0]
    
    # if opt.output_format == 'video':
    #     output_video_path = osp.join(result_root, new_name + '.mp4')
    #     cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}'.format(osp.join(result_root, new_name), output_video_path)
    #     os.system(cmd_str)
    # raise Exception
    demo(opt)
