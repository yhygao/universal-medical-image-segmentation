import torch
import torch.nn as nn
import torch.nn.functional as F
from .hermes_unet_utils import inconv, down_block, up_block, PriorInitFusionLayer
from .hermes_utils import HierarchyPriorClassifier, ModalityClassifier
from .utils import get_block, get_norm
import pdb


class Hermes_UNet(nn.Module):
    def __init__(self, 
        in_ch, 
        base_ch, 
        scale=[2,2,2,2], 
        kernel_size=[3,3,3,3], 
        block='BasicBlock', 
        num_block=[2,2,2,2],
        pool=True, 
        norm='in', 
        tn=72,  
        mn=6
        ):
        super().__init__()
        '''
        Args:
            in_ch: the num of input channel
            base_ch: the num of channels in the entry level
            scale: should be a list to indicate the downsample scale along each axis 
                in each level, e.g. [1, 1, 2, 2] such that all axis use the same scale
                or [[1,2,2], [2,2,2], [2,2,2], [2,2,2]] for difference scale on each axis
            kernel_size: the 3D kernel size of each level
                e.g. [3,3,3,3] or [[1,3,3], [1,3,3], [3,3,3], [3,3,3]]
            num_classes: the target class number
            block: 'ConvNormAct' for origin UNet, 'BasicBlock' for ResUNet
            num_block: number of blocks in each stage
            pool: use maxpool or use strided conv for downsample
            norm: the norm layer type, bn or in
            tn: the number of task priors
            mn: the number of modality priors

        '''
        block = get_block(block)
        norm = get_norm(norm)
    
        self.inc = inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)

        self.down1 = down_block(base_ch, 2*base_ch, num_block=num_block[0], block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[1], norm=norm)
        
        self.down2 = down_block(2*base_ch, 4*base_ch, num_block=num_block[1], block=block, pool=pool, down_scale=scale[1], kernel_size=kernel_size[2], norm=norm)
        self.prior_init_fuse_2 = PriorInitFusionLayer(4*base_ch, 4*base_ch, block_num=2, task_prior_num=tn, modality_prior_num=mn)
        
        self.down3 = down_block(4*base_ch, 8*base_ch, num_block=num_block[2], block=block, pool=pool, down_scale=scale[2], kernel_size=kernel_size[3], norm=norm)
        self.prior_init_fuse_3 = PriorInitFusionLayer(8*base_ch, 8*base_ch, block_num=2, task_prior_num=tn, modality_prior_num=mn)
        
        self.down4 = down_block(8*base_ch, 10*base_ch, num_block=num_block[3], block=block, pool=pool, down_scale=scale[3], kernel_size=kernel_size[4], norm=norm)
        self.prior_init_fuse_4 = PriorInitFusionLayer(10*base_ch, 10*base_ch, block_num=4, task_prior_num=tn, modality_prior_num=mn)

        self.up1 = up_block(10*base_ch, 8*base_ch, num_block=num_block[2], block=block, up_scale=scale[3], kernel_size=kernel_size[3], norm=norm)
        self.prior_fuse_5 = PriorInitFusionLayer(8*base_ch, 8*base_ch, block_num=2, task_prior_num=tn, modality_prior_num=mn)
        
        self.up2 = up_block(8*base_ch, 4*base_ch, num_block=num_block[1], block=block, up_scale=scale[2], kernel_size=kernel_size[2], norm=norm)
        self.prior_fuse_6 = PriorInitFusionLayer(4*base_ch, 4*base_ch, block_num=2, task_prior_num=tn, modality_prior_num=mn)

        self.up3 = up_block(4*base_ch, 2*base_ch, num_block=num_block[0], block=block, up_scale=scale[1], kernel_size=kernel_size[1], norm=norm)
        self.up4 = up_block(2*base_ch, base_ch, num_block=2, block=block, up_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
    
        self.out = HierarchyPriorClassifier(34*base_ch, base_ch)
        self.mod_out = ModalityClassifier(34*base_ch, mn)



    def forward(self, x, tgt_idx, mod_idx): 
    
        tn = tgt_idx.shape[1] # the number of task prior tokens in the batch
        mn = mod_idx.shape[1] # the number of modality piror tokens in the batch

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2) 
        x3, priors_2 = self.prior_init_fuse_2(x3, tgt_idx, mod_idx)

        x4 = self.down3(x3)
        x4, priors_3 = self.prior_init_fuse_3(x4, tgt_idx, mod_idx)

        x5 = self.down4(x4)
        x5, priors_4 = self.prior_init_fuse_4(x5, tgt_idx, mod_idx)

        
        out = self.up1(x5, x4)
        out, priors_5 = self.prior_fuse_5(out, tgt_idx, mod_idx)

        out = self.up2(out, x3) 
        out, priors_6 = self.prior_fuse_6(out, tgt_idx, mod_idx)

        out = self.up3(out, x2)
        out = self.up4(out, x1)
        
        # only select the task posterior tokens for segmentation
        task_priors_6 = priors_6[:, :tn, :]
        task_priors_5 = priors_5[:, :tn, :]
        task_priors_4 = priors_4[:, :tn, :]
        task_priors_3 = priors_3[:, :tn, :]
        task_priors_2 = priors_2[:, :tn, :]

        task_prior_list = [task_priors_2, task_priors_3, task_priors_4, task_priors_5, task_priors_6]
        out = self.out(out, task_prior_list)
        
        # only select the modality posterior tokens for modality classification
        mod_priors_6 = priors_6[:, tn:, :]
        mod_priors_5 = priors_5[:, tn:, :]
        mod_priors_4 = priors_4[:, tn:, :]
        mod_priors_3 = priors_3[:, tn:, :]
        mod_priors_2 = priors_2[:, tn:, :]

        mod_prior_list = [mod_priors_2, mod_priors_3, mod_priors_4, mod_priors_5, mod_priors_6]
        mod_out =  self.mod_out(mod_prior_list)

        return out, mod_out


