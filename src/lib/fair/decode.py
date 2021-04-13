from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .utils.utils import _gather_feat, _tranpose_and_gather_feat

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _topk_channel(scores, K=40):
      batch, cat, height, width = scores.size()
      
      topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

      topk_inds = topk_inds % (height * width)
      topk_ys   = (topk_inds / width).int().float()
      topk_xs   = (topk_inds % width).int().float()

      return topk_scores, topk_inds, topk_ys, topk_xs

def _topk(scores, K=40, conf_thres=None):
    batch, cat, height, width = scores.size()
    # conf_thres = None
    if conf_thres:
        tmp = scores.view(-1)

        mask = tmp > conf_thres
        topk_inds = mask.nonzero()[...,-1]
        
        # topk_inds = mask.nonzero()[...,-1].unsqueeze(0)
        # print(topk_inds)
        
        topk_scores = tmp[topk_inds]
        # print(topk_scores)
        
        topk_inds = topk_inds % (height * width)
        topk_ys   = (topk_inds / width).int().float()
        topk_xs   = (topk_inds % width).int().float()
        # topk_clses = (topk_inds / topk_inds.size(1)).int()
        # print(topk_clses)
        topk_clses = torch.zeros((batch, topk_inds.size(0))).cuda()
        return topk_scores.unsqueeze(0), topk_inds.unsqueeze(0), topk_clses, topk_ys, topk_xs
        
    else:
        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)
    
    # print(topk_scores.shape, topk_inds.shape)
    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
    
    # print(topk_scores, topk_inds)
    # raise Exception
      
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()

    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def mot_decode(heat, wh, reg=None, ltrb=False, K=100, conf_thres=None):
    batch, cat, height, width = heat.size()

    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)
    # import matplotlib.pyplot as plt
    # plt.imshow(heat.cpu().numpy().squeeze())
    # plt.show()
    # raise Exception

    # conf_thres = None
    
    scores, inds, clses, ys, xs = _topk(heat, K=K, conf_thres=conf_thres)
    
    # print(clses.shape)
    K = inds.size(1)

    if reg is not None:
        reg = _tranpose_and_gather_feat(reg, inds)
        # print("reg", reg.shape)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    wh = _tranpose_and_gather_feat(wh, inds)
    if ltrb:
        wh = wh.view(batch, K, 4)
    else:
        wh = wh.view(batch, K, 2)
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    if ltrb:
        bboxes = torch.cat([xs - wh[..., 0:1],
                            ys - wh[..., 1:2],
                            xs + wh[..., 2:3],
                            ys + wh[..., 3:4]], dim=2)
    else:
        bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                            ys - wh[..., 1:2] / 2,
                            xs + wh[..., 0:1] / 2,
                            ys + wh[..., 1:2] / 2], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)

    return detections, inds
