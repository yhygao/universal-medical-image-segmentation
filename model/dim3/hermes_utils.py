import torch
import torch.nn as nn
import torch.nn.functional as F
from .trans_layers import Mlp

import pdb

class HierarchyPriorClassifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
            
        self.norm = nn.LayerNorm(in_dim)
        self.classifier_pred = nn.Sequential(
            Mlp(in_dim=in_dim, out_dim=out_dim),
            Mlp(in_dim=out_dim, out_dim=out_dim)
            )   
                
        self.classifier_pred.apply(self.init_weights)
    def init_weights(self, m): 
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0)

    def forward(self, x, prior_list):
        priors = torch.cat(prior_list, dim=2)
        priors = self.norm(priors)
        weights = self.classifier_pred(priors) # B, n, dim
            
        B, C, D, H, W = x.shape
        #x = rearrange(x, 'b c d h w -> b (d h w) c', b=B, c=C, d=D, h=H, w=W)
        x = x.view(B, C, -1) 
        x = x.permute(0, 2, 1).contiguous()
            
        weights = torch.permute(weights, (0, 2, 1)) 
            
        output = torch.bmm(x, weights)
            
        #output = rearrange(output, 'b (d h w) c -> b c d h w', b=B, c=weights.shape[2], d=D, h=H, w=W)
        c = weights.shape[2]
        output = output.permute(0, 2, 1).contiguous()
        output = output.view(B, c, D, H, W).contiguous()

        return output



class ModalityClassifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        
        self.norm = nn.LayerNorm(in_dim)
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = Mlp(in_dim=in_dim, out_dim=out_dim)

    def forward(self, prior_list):

        priors = torch.cat(prior_list, dim=2)
        priors = self.norm(priors)

        priors = torch.permute(priors, (0, 2, 1))
        priors = self.avg_pool(priors).squeeze(2)


        pred = self.classifier(priors)

        return pred

