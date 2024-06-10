import numpy as np
import torch
import torch.nn as nn
import pdb

def get_model(args, pretrain=False):
    if args.dimension == '3d':
        if args.model == 'hermes_medformer':
            from .dim3 import Hermes_MedFormer

            return Hermes_MedFormer(args.in_chan, args.base_chan, map_size=args.map_size, conv_block=args.conv_block, conv_num=args.conv_num, trans_num=args.trans_num, num_heads=args.num_heads, fusion_depth=args.fusion_depth, fusion_dim=args.fusion_dim, fusion_heads=args.fusion_heads, expansion=args.expansion, attn_drop=args.attn_drop, proj_drop=args.proj_drop, proj_type=args.proj_type, norm=args.norm, act=args.act, kernel_size=args.kernel_size, scale=args.down_scale, aux_loss=args.aux_loss, tn=args.tn, mn=args.mn)
        
        elif args.model == 'hermes_resunet':
            from .dim3 import Hermes_UNet
            
            return Hermes_UNet(args.in_chan, args.base_chan, scale=args.down_scale, norm=args.norm, kernel_size=args.kernel_size, block=args.block, num_block=args.num_block, tn=args.tn, mn=args.mn)


        else:
            raise ValueError('Invalid model name')
    
    else:
        raise ValueError('Invalid dimension, should be \'3d\'')

