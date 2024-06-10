import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from inference.utils import get_inference
from metric.utils import calculate_distance, calculate_dice, calculate_dice_split
import numpy as np
from .utils import concat_all_gather, remove_wrap_arounds
import logging
import pdb
from utils import is_master
from tqdm import tqdm
import SimpleITK as sitk
import math

def validation(net, dataloader, args):
    
    net.eval()

    dice_list = []
    unique_labels_list = []

    inference = get_inference(args)
    
    logging.info("Evaluating")
    
    with torch.no_grad():
        iterator = tqdm(dataloader)
        for (images, labels, tgt_idx, mod_idx, spacing) in iterator:
            if labels.max() != 0:
                # spacing here is used for distance metrics calculation
                images, labels = images.cuda().float(), labels.to(torch.int8).cuda()
                tgt_idx = tgt_idx.cuda().long()
                mod_idx = mod_idx.cuda().long().unsqueeze(1)
                C = torch.nonzero(tgt_idx.squeeze(0)+1).shape[0] # the number of classes, +1 to eliminate null tokens (tgt_idx with -1)
                
                if args.dimension == '2d':
                    images = images.permute(1, 0, 2, 3)
                
                torch.cuda.empty_cache()
                pred = inference(net, images, tgt_idx, mod_idx, args)

                del images

                pred[pred >= 0.5] = 1
                pred[pred < 0.5] = 0

                pred = pred.to(torch.int8)
                torch.cuda.empty_cache()
               
                if args.dimension == '2d':
                    labels = labels.squeeze(0)
                else:
                    label_pred = pred.squeeze(0)
                    labels = labels.squeeze(0)
                    
                label_pred = label_pred[:C, :, :, :]
                labels = labels[:C, :, :, :]

                torch.cuda.empty_cache()
                tmp_dice_list, _, _ = calculate_dice_split(label_pred.view(C, -1), labels.view(C, -1), C)

                unique_labels, _ = torch.max(labels.view(C, -1), dim=1)
                unique_labels = torch.nonzero(unique_labels).squeeze(1).cpu().numpy()

                del label_pred, labels

                dice_list.append(tmp_dice_list.cpu().numpy())
                unique_labels_list.append(unique_labels)

    out_dice = []
    for cls in range(0, C):
        out_dice.append([])

    for idx in range(len(dice_list)):
        for cls in range(0, C):
            if cls in unique_labels_list[idx]:
                out_dice[cls].append(dice_list[idx][cls])

    out_dice_mean = []
    for cls in range(0, C):
        out_dice_mean.append(np.array(out_dice[cls]).mean())

    return np.array(out_dice_mean)




def validation_ddp(net, dataloader, args):
    
    net.eval()

    dice_list = []
    unique_labels_list = []

    inference = get_inference(args)

    logging.info(f"Evaluating")

    with torch.no_grad():
        iterator = tqdm(dataloader) if is_master(args) else dataloader
        for (images, labels, tgt_idx, mod_idx, spacing) in iterator:
            # spacing here is used for distance metrics calculation
            
            images, labels = images.cuda().float(), labels.to(torch.int8).cuda()
            tgt_idx = tgt_idx.cuda().long()
            mod_idx = mod_idx.cuda().long().unsqueeze(1)
            C = torch.nonzero(tgt_idx.squeeze(0)+1).shape[0]
            
            if args.dimension == '2d':
                images = images.permute(1, 0, 2, 3)
            
            torch.cuda.empty_cache()
            pred = inference(net, images, tgt_idx, mod_idx, args)

            del images
            torch.cuda.empty_cache()

            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            pred = pred.to(torch.int8)
            torch.cuda.empty_cache()
            
            if args.dimension == '2d':
                labels = labels.squeeze(0)
            else:
                pred = pred.squeeze(0)
                labels = labels.squeeze(0)
 
            pred = pred[:C, :, :, :]
            labels = labels[:C, :, :, :]

            torch.cuda.empty_cache()
            tmp_dice_list, _, _ = calculate_dice_split(pred.view(C, -1), labels.view(C, -1), C)

            unique_labels, _ = torch.max(labels.view(C, -1), dim=1)
            unique_labels = torch.nonzero(unique_labels).squeeze(1).cpu().numpy() # non-zero means have that target object

            del pred, labels
            torch.cuda.empty_cache()

            unique_labels =  np.pad(unique_labels, (0, 100-len(unique_labels)), 'constant', constant_values=-1)
            # the length of padding is just a randomly picked number (most medical tasks don't have over 100 classes)
            # The padding here is because the all_gather in DDP requires the tensors in gpus have the same shape

            tmp_dice_list = tmp_dice_list.unsqueeze(0)
            unique_labels = np.expand_dims(unique_labels, axis=0)

            if args.distributed:
                # gather results from all gpus
                tmp_dice_list = concat_all_gather(tmp_dice_list)
                
                unique_labels = torch.from_numpy(unique_labels).cuda()
                unique_labels = concat_all_gather(unique_labels)
                unique_labels = unique_labels.cpu().numpy()
                

            tmp_dice_list = tmp_dice_list.cpu().numpy()
            for idx in range(len(tmp_dice_list)):  # get the result for each sample
                dice_list.append(tmp_dice_list[idx])
                unique_labels_list.append(unique_labels[idx])
    
    # Due to the DistributedSampler pad samples to make data evenly distributed to all gpus,
    # we need to remove the padded samples for correct evaluation.
    if args.distributed:
        world_size = dist.get_world_size()
        dataset_len = len(dataloader.dataset)

        padding_size = 0 if (dataset_len % world_size) == 0 else world_size - (dataset_len % world_size)
        
        for _ in range(padding_size):
            dice_list.pop()
            unique_labels_list.pop()
    
    out_dice = []
    for cls in range(0, C):
        out_dice.append([])
    
    for idx in range(len(dice_list)):
        for cls in range(0, C):
            if cls in unique_labels_list[idx]:
                out_dice[cls].append(dice_list[idx][cls])
    
    out_dice_mean = []
    for cls in range(0, C):
        out_dice_mean.append(np.array(out_dice[cls]).mean())

    return np.array(out_dice_mean)



def validation_ddp_with_large_images(net, dataloader, args):
    
    net.eval()

    dice_list = []
    unique_labels_list = []

    inference = get_inference(args)

    logging.info(f"Evaluating")

    with torch.no_grad():
        iterator = tqdm(dataloader) if is_master(args) else dataloader
        for (images, labels, tgt_idx, mod_idx, spacing) in iterator:
            # spacing here is used for distance metrics calculation
            
            images, labels = images.cuda().float(), labels.to(torch.int8).cuda()
            tgt_idx = tgt_idx.cuda().long()
            mod_idx = mod_idx.cuda().long().unsqueeze(1)
            C = torch.nonzero(tgt_idx.squeeze(0)+1).shape[0]
            tgt_idx = tgt_idx[:, :C+1]

            if args.dimension == '2d':
                images = images.permute(1, 0, 2, 3)
            
            torch.cuda.empty_cache()
            
            B, _, D, H, W = images.shape

            z_len = 600.

            if D > z_len:
                num_z_chunks = math.ceil(D / z_len)
                z_chunk_len = math.ceil(D / num_z_chunks)

                image_chunk_list = []
                for i in range(num_z_chunks):
                    image_chunk_list.append(images[:, :, i*z_chunk_len:(i+1)*z_chunk_len, :, :])
                label_pred_list = []
                for image_chunk in image_chunk_list:
                    pred = inference(net, image_chunk, tgt_idx, mod_idx, args)
                    label_pred_list.append(pred)
                pred = torch.cat(label_pred_list, dim=2)

            else:
                pred = inference(net, images, tgt_idx, mod_idx, args)

            del images
            torch.cuda.empty_cache()

            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            pred = pred.to(torch.int8)
            torch.cuda.empty_cache()
            
            if args.dimension == '2d':
                labels = labels.squeeze(0)
            else:
                pred = pred.squeeze(0)
                labels = labels.squeeze(0)
 
            pred = pred[:C, :, :, :]
            labels = labels[:C, :, :, :]


            torch.cuda.empty_cache()
            tmp_dice_list, _, _ = calculate_dice_split(pred.view(C, -1), labels.view(C, -1), C)

            unique_labels, _ = torch.max(labels.view(C, -1), dim=1)
            unique_labels = torch.nonzero(unique_labels).squeeze(1).cpu().numpy() # non-zero means have that target object

            del pred, labels
            torch.cuda.empty_cache()

            unique_labels =  np.pad(unique_labels, (0, 100-len(unique_labels)), 'constant', constant_values=-1)
            # the length of padding is just a randomly picked number (most medical tasks don't have over 100 classes)
            # The padding here is because the all_gather in DDP requires the tensors in gpus have the same shape

            tmp_dice_list = tmp_dice_list.unsqueeze(0)
            unique_labels = np.expand_dims(unique_labels, axis=0)

            if args.distributed:
                # gather results from all gpus
                tmp_dice_list = concat_all_gather(tmp_dice_list)
                
                unique_labels = torch.from_numpy(unique_labels).cuda()
                unique_labels = concat_all_gather(unique_labels)
                unique_labels = unique_labels.cpu().numpy()
                
            tmp_dice_list = tmp_dice_list.cpu().numpy()
            for idx in range(len(tmp_dice_list)):  # get the result for each sample
                dice_list.append(tmp_dice_list[idx])
                unique_labels_list.append(unique_labels[idx])
    
    # Due to the DistributedSampler pad samples to make data evenly distributed to all gpus,
    # we need to remove the padded samples for correct evaluation.
    if args.distributed:
        world_size = dist.get_world_size()
        dataset_len = len(dataloader.dataset)

        padding_size = 0 if (dataset_len % world_size) == 0 else world_size - (dataset_len % world_size)
        
        for _ in range(padding_size):
            dice_list.pop()
            unique_labels_list.pop()
    
    out_dice = []
    for cls in range(0, C):
        out_dice.append([])
    
    for idx in range(len(dice_list)):
        for cls in range(0, C):
            if cls in unique_labels_list[idx]:
                out_dice[cls].append(dice_list[idx][cls])
    
    out_dice_mean = []
    for cls in range(0, C):
        out_dice_mean.append(np.array(out_dice[cls]).mean())

    return np.array(out_dice_mean)



