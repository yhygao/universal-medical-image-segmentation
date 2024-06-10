import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_block, get_norm, get_act
from .hermes_medformer_utils import down_block, up_block, inconv, SemanticMapFusion
from .hermes_utils import HierarchyPriorClassifier, ModalityClassifier
import pdb



class Hermes_MedFormer(nn.Module):
    
    def __init__(self, 
        in_chan, 
        base_chan=32, 
        map_size=[4,4,4], 
        conv_block='BasicBlock', 
        conv_num=[2,0,0,0, 0,0,2,2], 
        trans_num=[0,2,4,6, 4,2,0,0], 
        chan_num=[64,128,256,320,256,128,64,32], 
        num_heads=[1,4,8,10, 8,4,1,1], 
        fusion_depth=2, 
        fusion_dim=320, 
        fusion_heads=10, 
        expansion=4, 
        attn_drop=0., 
        proj_drop=0., 
        proj_type='depthwise', 
        norm='in', 
        act='relu', 
        kernel_size=[[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]], 
        scale=[[2,2,2],[2,2,2],[2,2,2],[2,2,2]], 
        aux_loss=True,
        tn=72,
        mn=6
        ):
        super().__init__()

        if conv_block == 'BasicBlock':
            dim_head = [chan_num[i]//num_heads[i] for i in range(8)]

        
        conv_block = get_block(conv_block)
        norm = get_norm(norm)
        act = get_act(act)
        
        # self.inc and self.down1 forms the conv stem
        self.inc = inconv(in_chan, base_chan, block=conv_block, kernel_size=kernel_size[0], norm=norm, act=act)
        self.down1 = down_block(base_chan, chan_num[0], conv_num[0], trans_num[0], conv_block=conv_block, kernel_size=kernel_size[1], down_scale=scale[0], norm=norm, act=act, map_generate=False)
        
        # down2 down3 down4 up1 up2 apply the B-MHA blocks modified with the priors
        self.down2 = down_block(chan_num[0], chan_num[1], conv_num[1], trans_num[1], conv_block=conv_block, kernel_size=kernel_size[2], down_scale=scale[1], heads=num_heads[1], dim_head=dim_head[1], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_generate=True, task_prior_num=tn, modality_prior_num=mn)

        self.down3 = down_block(chan_num[1], chan_num[2], conv_num[2], trans_num[2], conv_block=conv_block, kernel_size=kernel_size[3], down_scale=scale[2], heads=num_heads[2], dim_head=dim_head[2], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_generate=True, task_prior_num=tn, modality_prior_num=mn)

        self.down4 = down_block(chan_num[2], chan_num[3], conv_num[3], trans_num[3], conv_block=conv_block, kernel_size=kernel_size[4], down_scale=scale[3], heads=num_heads[3], dim_head=dim_head[3], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_generate=True, task_prior_num=tn, modality_prior_num=mn)


        self.map_fusion = SemanticMapFusion(chan_num[1:4], fusion_dim, fusion_heads, depth=fusion_depth, norm=norm)

        self.up1 = up_block(chan_num[3], chan_num[4], conv_num[4], trans_num[4], conv_block=conv_block, kernel_size=kernel_size[3], up_scale=scale[3], heads=num_heads[4], dim_head=dim_head[4], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_shortcut=True, task_prior_num=tn, modality_prior_num=mn)

        self.up2 = up_block(chan_num[4], chan_num[5], conv_num[5], trans_num[5], conv_block=conv_block, kernel_size=kernel_size[2], up_scale=scale[2], heads=num_heads[5], dim_head=dim_head[5], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_shortcut=True, no_map_out=True, task_prior_num=tn, modality_prior_num=mn)

        self.up3 = up_block(chan_num[5], chan_num[6], conv_num[6], trans_num[6], conv_block=conv_block, kernel_size=kernel_size[1], up_scale=scale[1], norm=norm, act=act, map_shortcut=False)

        self.up4 = up_block(chan_num[6], chan_num[7], conv_num[7], trans_num[7], conv_block=conv_block, kernel_size=kernel_size[0], up_scale=scale[0], norm=norm, act=act, map_shortcut=False)
        
    
        self.aux_loss = aux_loss
        
        if aux_loss:
            self.aux_classifier = HierarchyPriorClassifier(sum(chan_num[1:6]), chan_num[5])
        

        self.out_classifier = HierarchyPriorClassifier(sum(chan_num[1:6]), chan_num[7])
        self.mod_out = ModalityClassifier(sum(chan_num[1:6]), mn)

    def forward(self, x, tgt_idx, mod_idx):
        tn = tgt_idx.shape[1] # the number of task prior tokens in the batch
        mn = mod_idx.shape[1] # the number of modality piror tokens in the batch


        x0 = self.inc(x)
        x1, _ = self.down1(x0)
        x2, map2, priors_2 = self.down2(x1, tgt_idx, mod_idx)
        x3, map3, priors_3 = self.down3(x2, tgt_idx, mod_idx)
        x4, map4, priors_4 = self.down4(x3, tgt_idx, mod_idx)
        
        map_list = [map2, map3, map4]
        map_list = self.map_fusion(map_list)

        
        out, semantic_map, priors_5 = self.up1(x4, x3, map_list[2], map_list[1], tgt_idx, mod_idx)
        out, semantic_map, priors_6 = self.up2(out, x2, semantic_map, map_list[0], tgt_idx, mod_idx)

        # only select the task posterior tokens for segmentation
        task_priors_6 = priors_6[:, :tn, :]
        task_priors_5 = priors_5[:, :tn, :]
        task_priors_4 = priors_4[:, :tn, :]
        task_priors_3 = priors_3[:, :tn, :]
        task_priors_2 = priors_2[:, :tn, :]

        task_prior_list = [task_priors_2, task_priors_3, task_priors_4, task_priors_5, task_priors_6] 
        
        if self.aux_loss:
            aux_out = self.aux_classifier(out, task_prior_list)
            aux_out = F.interpolate(aux_out, size=x.shape[-3:], mode='trilinear', align_corners=True)
        

        out, semantic_map = self.up3(out, x1, semantic_map, None)
        out, semantic_map = self.up4(out, x0, semantic_map, None)
        
        out = self.out_classifier(out, task_prior_list)

        # only select the modality posterior tokens for modality classification
        mod_priors_6 = priors_6[:, tn:, :]
        mod_priors_5 = priors_5[:, tn:, :]
        mod_priors_4 = priors_4[:, tn:, :]
        mod_priors_3 = priors_3[:, tn:, :]
        mod_priors_2 = priors_2[:, tn:, :]
        
        mod_prior_list = [mod_priors_2, mod_priors_3, mod_priors_4, mod_priors_5, mod_priors_6]
        mod_out =  self.mod_out(mod_prior_list)

        if self.aux_loss:
            return [out, aux_out], mod_out
        else:
            return out, mod_out

