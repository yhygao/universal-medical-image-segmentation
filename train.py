import builtins
import logging
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from model.utils import get_model
from training.dataset.utils import get_dataset
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler


from training.utils import update_ema_variables
from training.losses import BinaryDiceLoss, BinaryCrossEntropyLoss
from training.validation import validation_ddp_with_large_images as validation
from training.utils import (
    exp_lr_scheduler_with_warmup, 
    log_evaluation_result, 
    log_overall_result, 
    get_optimizer,
    unwrap_model_checkpoint,
)
import yaml
import argparse
import time
import math
import sys
import pdb
import warnings
import copy

from utils import (
    configure_logger,
    save_configure,
    is_master,
    AverageMeter,
    ProgressMeter,
    resume_load_optimizer_checkpoint,
    resume_load_model_checkpoint,
)
warnings.filterwarnings("ignore", category=UserWarning)



def train_net(net, trainset, valset_list, testset_list, args, ema_net=None):
    
    ########################################################################################
    # Dataset Creation
    
    samples_weight = torch.from_numpy(np.array(trainset.weight_list))
    train_sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight)*10)

    trainLoader = data.DataLoader(
        trainset, 
        batch_size=args.batch_size, 
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        pin_memory=(args.aug_device != 'gpu'),
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers>0),
    )
    
    valLoader_list = []
    for i in range(len(args.dataset_name_list)):
        dataset_name = args.dataset_name_list[i]
        valset = valset_list[i]
        val_sampler = DistributedSampler(valset) if args.distributed else None
        valLoader = data.DataLoader(
            valset,
            batch_size=1,  # has to be 1 sample per gpu, as the input size of 3D input is different
            shuffle=False, 
            sampler=val_sampler,
            pin_memory=True,
            num_workers=0
        )
        valLoader_list.append(valLoader)
    
    testLoader_list = []
    for i in range(len(args.dataset_name_list)):
        dataset_name = args.dataset_name_list[i]
        testset = testset_list[i]
        test_sampler = DistributedSampler(testset) if args.distributed else None
        testLoader = data.DataLoader(
            testset,
            batch_size=1,  # has to be 1 sample per gpu, as the input size of 3D input is different
            shuffle=False, 
            sampler=test_sampler,
            pin_memory=True,
            num_workers=0
        )
        testLoader_list.append(testLoader)
    
    logging.info(f"Created Dataset and DataLoader")

    ########################################################################################
    # Initialize tensorboard, optimizer, amp scaler and etc.
    writer = SummaryWriter(os.path.join(f"{args.log_path}", f"{args.unique_name}")) if is_master(args) else None

    optimizer = get_optimizer(args, net)
    
    if args.resume:
        resume_load_optimizer_checkpoint(optimizer, args)
    
    criterion_ce = BinaryCrossEntropyLoss(class_num=args.tn).cuda(args.proc_idx)
    criterion_dl = BinaryDiceLoss(class_num=args.tn).cuda(args.proc_idx)
    criterion_mod = nn.CrossEntropyLoss(ignore_index=-1).cuda(args.proc_idx)
    
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    ########################################################################################
    # Start training
    best_Dice = np.zeros(np.array(args.dataset_classes_list).sum())
    
    for epoch in range(args.start_epoch, args.epochs):
        logging.info(f"Starting epoch {epoch+1}/{args.epochs}")
        exp_scheduler = exp_lr_scheduler_with_warmup(optimizer, init_lr=args.base_lr, epoch=epoch, warmup_epoch=args.warmup_epoch, max_epoch=args.epochs)
        logging.info(f"Current lr: {exp_scheduler:.4e}")
       
        train_epoch(trainLoader, net, ema_net, optimizer, epoch, writer, criterion_ce, criterion_dl, criterion_mod, scaler, args)
        
        ##################################################################################
        # Evaluation, save checkpoint and log training info
        net_for_eval = ema_net if args.ema else net

        if is_master(args):
            # save the latest checkpoint, including net, ema_net, and optimizer
            net_state_dict, ema_net_state_dict = unwrap_model_checkpoint(net, ema_net, args)

            torch.save({
                'epoch': epoch+1,
                'model_state_dict': net_state_dict,
                'ema_model_state_dict': ema_net_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.cp_path, args.dataset, args.unique_name, 'latest.pth'))


        if (epoch+1) % args.val_freq == 0 or (epoch+1>(args.epochs-20)):
            best_Dice = validate(valLoader_list, net_for_eval, net, ema_net, epoch, writer, optimizer, args, best_Dice, prefix='val')
            _ = validate(testLoader_list, net_for_eval, net, ema_net, epoch, writer, optimizer, args, best_Dice, prefix='test')

    return best_Dice

def validate(loader_list, net_for_eval, net, ema_net, epoch, writer, optimizer, args, best_Dice, prefix='test'):
    all_dice = []
    for idx in range(len(loader_list)):
        Loader = loader_list[idx]
        dataset_name = args.dataset_name_list[idx]

        dice_list_test = validation(net_for_eval, Loader, args)

        if is_master(args):
            logging.info(f"{dataset_name}: {dice_list_test}")
            logging.info(f"{dataset_name} mean: {dice_list_test.mean()}")
            log_evaluation_result(writer, dice_list_test, dataset_name, epoch, args, prefix)
        all_dice += list(dice_list_test)
    
    if is_master(args):
        all_dice = np.array(all_dice)
        log_overall_result(writer, all_dice, epoch, args, prefix)

        if all_dice.mean() >= best_Dice.mean() and prefix == 'val':
            best_Dice = all_dice

            # Save the checkpoint with best performance
            net_state_dict, ema_net_state_dict = unwrap_model_checkpoint(net, ema_net, args)

            torch.save({
                'epoch': epoch+1,
                'model_state_dict': net_state_dict,
                'ema_model_state_dict': ema_net_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.cp_path, args.dataset, args.unique_name, 'best.pth'))

        logging.info("Evaluation Done")
        logging.info(f"Dice: {all_dice.mean():.4f}/Best Dice: {best_Dice.mean():.4f}")
    
    return best_Dice



def train_epoch(trainLoader, net, ema_net, optimizer, epoch, writer, criterion_ce, criterion_dl, criterion_mod, scaler, args):
    batch_time = AverageMeter("Time", ":6.2f")
    epoch_loss = AverageMeter("Loss", ":.2f")
    epoch_loss_seg = AverageMeter("Loss_seg", ":.2f")
    epoch_loss_mod = AverageMeter("Loss_mod", ":.2f")
    epoch_mod_acc = AverageMeter("Acc_mod", ":.2f")

    progress = ProgressMeter(
        #len(trainLoader), 
        args.iter_per_epoch,
        [batch_time, epoch_loss_seg, epoch_loss_mod, epoch_mod_acc], 
        prefix="Epoch: [{}]".format(epoch+1),
    )
    
    net.train()

    tic = time.time()
    iter_num_per_epoch = 0
    for i, (img, label, tgt_idx, mod_idx) in enumerate(trainLoader):
        
        img = img.cuda(args.proc_idx, non_blocking=True).float()
        label = label.cuda(args.proc_idx, non_blocking=True).long()
        tgt_idx = tgt_idx.cuda(args.proc_idx, non_blocking=True).long()
        mod_idx = mod_idx.cuda(args.proc_idx, non_blocking=True).long()
        step = i + epoch * len(trainLoader) # global steps
            
        # remove extra padded for eficiency
        max_cls_len = torch.max(torch.nonzero(tgt_idx!=-1), dim=0)[0][1] + 1 
        tgt_idx = tgt_idx[:, :max_cls_len+1]
        label = label[:, :max_cls_len+1, :, :, :]

        optimizer.zero_grad()

        loss_seg = 0
        if args.amp:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                result, mod_result = net(img, tgt_idx, mod_idx.unsqueeze(1))
                
                if isinstance(result, tuple) or isinstance(result, list):
                    # If use deep supervision, add all loss together 
                    for j in range(len(result)):
                        loss_seg += args.aux_weight[j] * (criterion_ce(result[j], label, tgt_idx) + criterion_dl(result[j], label, tgt_idx))
                else:
                
                    loss_seg = criterion_ce(result, label, tgt_idx) + criterion_dl(result, label, tgt_idx)
                if (mod_idx == -1).all():
                    loss_mod = torch.tensor(0.0, requires_grad=True).to(mod_result.device)
                else:
                    loss_mod = criterion_mod(mod_result, mod_idx)
               
                loss = loss_seg + args.loss_mod_weight * loss_mod

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        else:
            result, mod_result = net(img, tgt_idx, mod_idx.unsqueeze(1))
            if isinstance(result, tuple) or isinstance(result, list):
                # If use deep supervision, add all loss together 
                for j in range(len(result)):
                    loss_seg += args.aux_weight[j] * (criterion_ce(result[j], label, tgt_idx) + criterion_dl(result[j], label, tgt_idx))
            else:
                loss_seg = criterion_ce(result, label, tgt_idx) + criterion_dl(result, label, tgt_idx)

            if (mod_idx == -1).all():
                loss_mod = torch.tensor(0.0, requires_grad=True).to(mod_result.device)
            else:
                loss_mod = criterion_mod(mod_result, mod_idx)
            loss = loss_seg + args.loss_mod_weight * loss_mod

            loss.backward()
            optimizer.step()
        if args.ema:
            update_ema_variables(net, ema_net, args.ema_alpha, step)

        _, mod_pred = torch.max(mod_result.data, 1)
        correct = (mod_pred == mod_idx).sum().item()
        total = (mod_idx != -1).sum().item()
        mod_acc = correct / total if total != 0 else 0


        epoch_loss.update(loss.item(), img.shape[0])
        epoch_loss_seg.update(loss_seg.item(), img.shape[0])
        epoch_loss_mod.update(loss_mod.item(), img.shape[0])
        epoch_mod_acc.update(mod_acc, total)
        batch_time.update(time.time() - tic)
        tic = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
        
        if args.dimension == '3d':
            iter_num_per_epoch += 1
            if iter_num_per_epoch > args.iter_per_epoch:
                break

        if is_master(args):
            writer.add_scalar('Train/Loss', epoch_loss.avg, epoch+1)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch+1)
            writer.add_scalar('Train/Loss_seg', epoch_loss_seg.avg, epoch+1)
            writer.add_scalar('Train/Loss_mod', epoch_loss_mod.avg, epoch+1)
            writer.add_scalar('Train/Acc_mod', epoch_mod_acc.avg, epoch+1)


    


def get_parser():
    parser = argparse.ArgumentParser(description='Hermes, universal medical image segmentation')
    parser.add_argument('--dataset', type=str, default='universal', help='dataset name')
    parser.add_argument('--model', type=str, default='hermes_resunet', help='model name')
    parser.add_argument('--dimension', type=str, default='3d', help='2d model or 3d model')
    parser.add_argument('--pretrain', action='store_true', help='if use pretrained weight for init')
    parser.add_argument('--amp', action='store_true', help='if use the automatic mixed precision for faster training')
    parser.add_argument('--torch_compile', action='store_true', help='use torch.compile to accelerate training, only supported by PyTorch2.0')

    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--resume', action='store_true', help='if resume training from checkpoint')
    parser.add_argument('--load', type=str, default=False, help='load pretrained model')
    parser.add_argument('--cp_path', type=str, default='./exp/', help='the path to save checkpoint and logging info')
    parser.add_argument('--log_path', type=str, default='./log/', help='the path to save tensorboard log')
    parser.add_argument('--unique_name', type=str, default='test', help='unique experiment name')
    
    parser.add_argument('--gpu', type=str, default='0,1,2,3,4,5,6,7')

    args = parser.parse_args()

    config_path = 'config/%s/%s_%s.yaml'%(args.dataset, args.model, args.dimension)
    if not os.path.exists(config_path):
        raise ValueError("The specified configuration doesn't exist: %s"%config_path)

    logging.info('Loading configurations from %s'%config_path)

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    for key, value in config.items():
        setattr(args, key, value)

    return args
    


def init_network(args):
    net = get_model(args, pretrain=args.pretrain)

    if args.ema:
        ema_net = get_model(args, pretrain=args.pretrain)
        logging.info("Use EMA model for evaluation")
    else:
        ema_net = None
    
    if args.resume:
        resume_load_model_checkpoint(net, ema_net, args)
    
    if args.torch_compile:
        net = torch.compile(net)


    return net, ema_net 





def main_worker(proc_idx, ngpus_per_node, args, result_dict=None, trainset=None, valset=None, testset=None):
    # seed each process
    if args.reproduce_seed is not None:
        random.seed(args.reproduce_seed)
        np.random.seed(args.reproduce_seed)
        torch.manual_seed(args.reproduce_seed)

        if hasattr(torch, "set_deterministic"):
            torch.set_deterministic(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # set process specific info
    args.proc_idx = proc_idx
    args.ngpus_per_node = ngpus_per_node

    # suppress printing if not master
    if args.multiprocessing_distributed and args.proc_idx != 0:
        def print_pass(*args, **kwargs):
            pass

        builtins.print = print_pass
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + proc_idx
        
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=f"{args.dist_url}",
            world_size=args.world_size,
            rank=args.rank,
        )
        torch.cuda.set_device(args.proc_idx)

        # adjust data settings according to multi-processing
        args.batch_size = int(args.batch_size / args.world_size)


    args.cp_dir = f"{args.cp_path}/{args.dataset}/{args.unique_name}"
    os.makedirs(args.cp_dir, exist_ok=True)
    configure_logger(args.rank, args.cp_dir+f"/log.txt")
    save_configure(args)

    logging.info(
        f"\nDataset: {args.dataset},\n"
        + f"Model: {args.model},\n"
        + f"Dimension: {args.dimension}"
    )
    

    net, ema_net = init_network(args)

    net.to(f"cuda")
    if args.ema:
        ema_net.to(f"cuda")
    

    if args.distributed:
        net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = DistributedDataParallel(net, device_ids=[args.proc_idx], find_unused_parameters=True)
        # set find_unused_parameters to True if some of the parameters is not used in forward
        
        if args.ema:
            ema_net = nn.SyncBatchNorm.convert_sync_batchnorm(ema_net)
            ema_net = DistributedDataParallel(ema_net, device_ids=[args.proc_idx], find_unused_parameters=True)
            
            for p in ema_net.parameters():
                p.requires_grad_(False)


    logging.info(f"Created Model")
    best_Dice = train_net(net, trainset, valset, testset, args, ema_net)
    
    logging.info(f"Training and evaluation are done")
    
    if args.distributed:
        if is_master(args):
            # collect results from the master process
            result_dict['best_Dice'] = best_Dice
    else:
        return best_Dice
        

        



if __name__ == '__main__':
    # parse the arguments
    args = get_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.log_path = os.path.join(args.log_path,  args.dataset)

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size
    
    if args.world_size > 1:
        args.multiprocessing_distributed = True
    else:
        args.multiprocessing_distributed = False
    
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed


    if args.multiprocessing_distributed:
        with mp.Manager() as manager:
        # use the Manager to gather results from the processes
            result_dict = manager.dict()
                
            # Since we have ngpus_per_node processes per node, the total world_size
            # needs to be adjusted accordingly
            trainset = get_dataset(args, dataset_name_list=args.dataset_name_list, mode='train') 
            valset_list = []
            for dataset_name in args.dataset_name_list:
                valset = get_dataset(args, dataset_name_list=[dataset_name], mode='val')
                valset_list.append(valset)
            testset_list = []
            for dataset_name in args.dataset_name_list:
                testset = get_dataset(args, dataset_name_list=[dataset_name], mode='test')
                testset_list.append(testset)
            # Use torch.multiprocessing.spawn to launch distributed processes:
            # the main_worker process function
            mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, result_dict, trainset, valset_list, testset_list))
            best_Dice = result_dict['best_Dice']
    else:
        trainset = get_dataset(args, dataset_name_list=args.dataset_name_list, mode='train')
        valset_list = []
        for dataset_name in args.dataset_name_list:
            valset = get_dataset(args, dataset_name_list=[dataset_name], mode='val')
            valset_list.append(valset)
        testset_list = []
        for dataset_name in args.dataset_name_list:
            testset = get_dataset(args, dataset_name_list=[dataset_name], mode='test')
            testset_list.append(testset)
        # Simply call main_worker function
        best_Dice = main_worker(0, ngpus_per_node, args, trainset=trainset, valset=valset_list, testset=testset_list)

    
    #############################################################################################

    with open(f"{args.cp_path}/{args.dataset}/{args.unique_name}/results.txt",  'w') as f:
        np.set_printoptions(precision=4, suppress=True) 
        f.write('Dice\n')
        f.write(f"Each Class Dice: {best_Dice}\n")
        f.write(f"All classes Dice Avg: {best_Dice.mean()}\n")

        f.write("\n")

    logging.info('Training done.')

    sys.exit(0)

