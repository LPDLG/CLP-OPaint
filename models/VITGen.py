import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops import get_sinusoid_encoding_table, CorssAttnBlock, LandscapeDecoder
from .VIT import *
from .QEM import QueryExpansionModule
from .UPSM import PatchSmoothingModule

import torch
import torch.nn as nn
import torch.nn.functional as F



class TransGen(nn.Module):
    def __init__(self, opts, enc_ckpt_path=None):
        super(TransGen, self).__init__()
        self.output_size = opts.output_size
        self.input_size = opts.input_size
        self.patch_size = 16
        hidden_num = 1024
        
        self.output_query_width = self.output_size // self.patch_size
        self.input_query_width = self.input_size // self.patch_size
        self.current_stage = 1
        self.stage_patch_expand = [1, 2, 3]
        self.qem = QueryExpansionModule(hidden_num=hidden_num, input_size=self.input_size, output_size=self.output_size, patch_size=self.patch_size)

        self.transformer_decoder = LandscapeDecoder(
            dim=hidden_num,
            depth=opts.dec_depth,
            num_heads=8,
            mlp_ratio=2.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            window_size=4
        )



        self.psm = PatchSmoothingModule(patch_size=16, out_chans=3, embed_dim=hidden_num)
        self.apply(self._init_weights)

        self.transformer_encoder = vit_large_patch16(pretrained=True, img_size=224, init_ckpt=enc_ckpt_path)

        self.enc_image_size = 224

        self.pos_embed = get_sinusoid_encoding_table(12**2, hidden_num)
        self.inner_index, self.outer_index = self.get_index()

    def get_index(self):
        output_patches = self.output_size // self.patch_size
        input_patches = self.input_size // self.patch_size 
    
        mask = torch.ones(output_patches, output_patches)
        
        inner_pad = (output_patches - input_patches) // 2 
        mask[inner_pad:inner_pad+input_patches, inner_pad:inner_pad+input_patches] = 0
        
        if self.current_stage == 1:
            outer_pad = 1 
            mask_full = torch.ones_like(mask)
            mask_inner = torch.zeros(output_patches-2*outer_pad, output_patches-2*outer_pad)
            mask_full[outer_pad:-outer_pad, outer_pad:-outer_pad] = mask_inner
            mask = mask * (1 - mask_full)  
        
        return (mask == 0).view(-1), (mask == 1).view(-1)

    def update_stage(self, new_stage):
        prev_stage = self.current_stage
        self.current_stage = min(max(new_stage, 1), 3)

        if hasattr(self, 'qem'):
            self.qem.special_reset_flag = True 
        self.inner_index, self.outer_index = self.get_index()
        
        if hasattr(self, 'qem'):
            self.qem.current_stage = self.current_stage
            self.qem.inner_query_index, self.qem.outer_query_index = self.get_index()
            self.qem.update_stage(self.current_stage)
            

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def generate_vit_mask(self):
        vit_patches = 14
        mask = torch.ones(vit_patches, vit_patches)
        
        center_start = 3 
        center_end = vit_patches - center_start
        mask[center_start:center_end, center_start:center_end] = 0
        
        if self.current_stage == 1:
            expand_start = center_start - 1  
            expand_end = center_end + 1  
            mask[expand_start:expand_end, expand_start:expand_end] = 0
            
        return mask

    def forward(self, samples):
        if type(samples) is not dict:
            samples = {'input': samples, 'gt_inner': F.pad(samples, (32, 32, 32, 32))}
        x = samples['input']
        gt_inner = samples['gt_inner']

        b, c, w, h = x.size()
        assert w == 128 and h == 128
        padded_x = F.pad(x, (48, 48, 48, 48), mode='reflect')
        
        vit_mask = self.generate_vit_mask()
        vit_mask = vit_mask.view(-1).expand(b, -1).contiguous().bool()

        src = self.transformer_encoder.forward_features(padded_x, vit_mask)  # b n c
        
        query_embed = self.qem(src)
        
        center_mask = torch.zeros(self.output_query_width**2).bool()
        center_start = (self.output_query_width - self.input_query_width) // 2
        for i in range(self.input_query_width):
            for j in range(self.input_query_width):
                idx = (i+center_start) * self.output_query_width + (j+center_start)
                center_mask[idx] = True
        
        decode_index = ~center_mask
        decode_index = decode_index.to(src.device)
        
        full_pos = self.pos_embed.type_as(x).to(x.device).clone().detach().expand(x.size(0), -1, -1)
        tgt_all = query_embed[:, decode_index, :] + full_pos[:, decode_index, :]
        
        for i, dec in enumerate(self.transformer_decoder):
            tgt_all = dec(tgt_all, src)
        
        tgt = torch.zeros_like(query_embed, dtype=torch.float32)
        tgt[:, decode_index] = tgt_all
        
        center_indices = torch.nonzero(center_mask).squeeze(1)
        for i, idx in enumerate(center_indices):
            input_idx = i % (self.input_query_width**2)
            tgt[:, idx, :] = src[:, input_idx, :]
        
        fake = self.psm(tgt, gt_inner)
        return fake
