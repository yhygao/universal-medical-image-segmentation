import torch
import torch.nn as nn
import torch.nn.functional as F
from . import metrics
import numpy as np
import pdb

def calculate_distance(label_pred, label_true, spacing, C, percentage=95):
    # the input args are torch tensors
    if label_pred.is_cuda:
        label_pred = label_pred.cpu()
        label_true = label_true.cpu()
    
    label_pred = label_pred.numpy().astype(np.bool)
    label_true = label_true.numpy().astype(np.bool)
    spacing = spacing.numpy()

    ASD_list = np.zeros(C)
    HD_list = np.zeros(C)

    for i in range(C):
        tmp_surface = metrics.compute_surface_distances(label_true[i], label_pred[i], spacing)
        dis_gt_to_pred, dis_pred_to_gt = metrics.compute_average_surface_distance(tmp_surface)
        ASD_list[i] = (dis_gt_to_pred + dis_pred_to_gt) / 2 

        HD = metrics.compute_robust_hausdorff(tmp_surface, percentage)
        HD_list[i] = HD

    return ASD_list, HD_list



def calculate_dice_split(pred, target, C, block_size=64*64*64):
    # evaluate every 128*128*128 block

    N, num = pred.shape
    assert C == N

    split_num = num // block_size     
    total_sum = torch.zeros(C).to(pred.device)
    total_intersection = torch.zeros(C).to(pred.device)
    
    for i in range(split_num):
        dice, intersection, summ = calculate_dice(pred[:, i*block_size:(i+1)*block_size], target[:, i*block_size:(i+1)*block_size], C)
        total_intersection += intersection
        total_sum += summ
    if num % block_size != 0:
        dice, intersection, summ = calculate_dice(pred[:, (i+1)*block_size:], target[:, (i+1)*block_size:], C)
        total_intersection += intersection
        total_sum += summ

    dice = 2 * total_intersection / (total_sum + 1e-5)

    return dice, total_intersection, total_sum

def calculate_dice(pred, target, C): 
    
    # pred and target are torch tensor
    N = pred.shape[0]
    assert C == N
    intersection= pred * target
    summ = pred + target
    
    intersection = intersection.sum(1)
    summ = summ.sum(1)

    intersection = intersection.float()
    summ = summ.float()
    
    eps = 1e-5
    summ += eps
    dice = 2 * intersection / summ

    return dice, intersection, summ
