import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv_layers import BasicBlock, Bottleneck, ConvNormAct
from .trans_layers import Attention, CrossAttention, LayerNorm, Mlp, PreNorm
import pdb
from einops import rearrange

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=[3,3,3], block=BasicBlock, norm=nn.BatchNorm3d):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * 3
        pad_size = [i//2 for i in kernel_size]
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=pad_size, bias=False)
        self.conv2 = block(out_ch, out_ch, kernel_size=kernel_size, norm=norm)

    def forward(self, x): 
        out = self.conv1(x)
        out = self.conv2(out)

        return out 


class down_block(nn.Module):
    def __init__(self, in_ch, out_ch, num_block, block=BasicBlock, kernel_size=[3,3,3], down_scale=[2,2,2], pool=True, norm=nn.BatchNorm3d):
        super().__init__() 
        
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * 3
        if isinstance(down_scale, int):
            down_scale = [down_scale] * 3

        block_list = []

        if pool:
            block_list.append(nn.MaxPool3d(down_scale))
            block_list.append(block(in_ch, out_ch, kernel_size=kernel_size, norm=norm))
        else:
            block_list.append(block(in_ch, out_ch, stride=down_scale, kernel_size=kernel_size, norm=norm))

        for i in range(num_block-1):
            block_list.append(block(out_ch, out_ch, stride=1, kernel_size=kernel_size, norm=norm))

        self.conv = nn.Sequential(*block_list)
    def forward(self, x):
        return self.conv(x)

class up_block(nn.Module):
    def __init__(self, in_ch, out_ch, num_block, block=BasicBlock, kernel_size=[3,3,3], up_scale=[2,2,2], norm=nn.BatchNorm3d):
        super().__init__()
        
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * 3
        if isinstance(up_scale, int):
            up_scale = [up_scale] * 3

        self.up_scale = up_scale


        block_list = []

        block_list.append(block(in_ch+out_ch, out_ch, kernel_size=kernel_size, norm=norm))
        for i in range(num_block-1):
            block_list.append(block(out_ch, out_ch, kernel_size=kernel_size, norm=norm))

        self.conv = nn.Sequential(*block_list)

    def forward(self, x1, x2):
        input_dtype = x1.dtype
        # F.interpolate trilinear doesn't support bfloat16, so need to cast to float32 for upsampling then cast back if using amp training
        x1 = F.interpolate(x1.float(), size=x2.shape[2:], mode='trilinear', align_corners=True)
        x1 = x1.to(input_dtype)
        out = torch.cat([x2, x1], dim=1)

        out = self.conv(out)

        return out

class DualPreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x1, x2, **kwargs):
        return self.fn(self.norm1(x1), self.norm2(x2), **kwargs)


class PriorAttentionBlock(nn.Module):
    def __init__(self, feat_dim, heads=4, dim_head=64, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.feat_dim = feat_dim
        self.heads = heads
        self.scale = dim_head ** (-0.5)
        self.dim_head = dim_head

        dim = feat_dim
        mlp_dim = dim * 4

        # update priors by aggregating from the feature map
        self.prior_aggregate_block = DualPreNorm(dim, CrossAttention(dim, heads, dim_head, attn_drop, proj_drop))
        self.prior_ffn = PreNorm(dim, Mlp(dim, mlp_dim, dim, drop=proj_drop))

        # update the feature map by injecting knowledge from the priors
        self.feat_aggregate_block = DualPreNorm(dim, CrossAttention(dim, heads, dim_head, attn_drop, proj_drop))
        self.feat_ffn = PreNorm(dim, Mlp(dim, mlp_dim, dim, drop=proj_drop))


    def forward(self, x1, x2):
        # x1: image feature map, x2: priors

        x2 = self.prior_aggregate_block(x2, x1) + x2
        x2 = self.prior_ffn(x2) + x2

        x1 = self.feat_aggregate_block(x1, x2) + x1
        x1 = self.feat_ffn(x1) + x1

        return x1, x2


class PriorInitFusionLayer(nn.Module):
    def __init__(self, feat_dim, prior_dim, block_num=2, task_prior_num=42, modality_prior_num=2, l=10):
        super().__init__()
        
        # random initialize the priors
        self.task_prior = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(task_prior_num+1, prior_dim))) # +1 for null token
        self.modality_prior = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(modality_prior_num, l, prior_dim)))

        self.attn_layers = nn.ModuleList([])
        for i in range(block_num):
            self.attn_layers.append(PriorAttentionBlock(feat_dim, heads=feat_dim//32, dim_head=32, attn_drop=0, proj_drop=0))

    def forward(self, x, tgt_idx, mod_idx):
        # x: image feature map, tgt_idx: target task index, mod_idx: modality index
        B, C, D, H, W = x.shape
        
        task_prior_list = []
        modality_prior_list = []
        # prior selection
        for i in range(B):
            idxs = tgt_idx[i]
            task_prior_list.append(self.task_prior[idxs, :])
            modality_prior_list.append(self.modality_prior[mod_idx[i], :, :])
        

        task_priors = torch.stack(task_prior_list)
        modality_priors = torch.stack(modality_prior_list)
        modality_priors = modality_priors.squeeze(1)

        priors = torch.cat([task_priors, modality_priors], dim=1)
        
        #x = rearrange(x, 'b c d h w -> b (d h w) c', d=D, h=H, w=W)
        b, c, d, h, w = x.shape
        x = x.view(b, c, -1)
        x = x.permute(0, 2, 1).contiguous()

        
        for layer in self.attn_layers:
            x, priors = layer(x, priors)
        
        #x = rearrange(x, 'b (d h w) c -> b c d h w', d=D, h=H, w=W, c=C)
        x = x.permute(0, 2, 1)
        x = x.view(b, c, d, h, w).contiguous()

        return x, priors
