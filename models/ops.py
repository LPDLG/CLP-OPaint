import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
            # cls to token & token 2 cls & cls to cls

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = \
                torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer("relative_position_index", relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.relative_position_bias_table is not None:
            relative_position_bias = \
                self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, all_head_dim * 1, bias=False)
        self.kv = nn.Linear(dim, all_head_dim * 2, bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
            # cls to token & token 2 cls & cls to cls

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = \
                torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer("relative_position_index", relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y, rel_pos_bias=None):
        B, N1, C = x.shape
        B, N2, C = y.shape
        q_bias = None
        kv_bias = None
        if self.q_bias is not None:
            q_bias=self.q_bias
            kv_bias = torch.cat((torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q=F.linear(input=x,weight=self.q.weight,bias=q_bias)
        kv = F.linear(input=y, weight=self.kv.weight, bias=kv_bias)
        q = q.reshape(B, N1, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q = q[0]
        kv = kv.reshape(B, N2, 2, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.relative_position_bias_table is not None:
            relative_position_bias = \
                self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N1, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CorssAttnBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)
        self.cross_attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm3= norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_3 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2, self.gamma_3 = None, None, None

    def forward(self, x, y, rel_pos_bias=None):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.cross_attn(self.norm2(x),y, rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.mlp(self.norm3(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.gamma_2 * self.cross_attn(self.norm2(x),y, rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.gamma_3 * self.mlp(self.norm3(x)))
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, rel_pos_bias=None):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class RelativePositionBias(nn.Module):
    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        # cls to token & token 2 cls & cls to cls

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = \
            torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer("relative_position_index", relative_position_index)

        # trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self):
        relative_position_bias = \
            self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] + 1,
                self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
        return relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

import numpy as np
def get_sinusoid_encoding_table(n_position, d_hid):
    ''' Sinusoid position encoding table '''
    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)



class LandscapeDecoderBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=2.0, qkv_bias=True, qk_scale=None, drop=0., 
                 attn_drop=0., drop_path=0., init_values=None, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, window_size=16, memory_efficient=True):
        super().__init__()
        self.memory_efficient = memory_efficient
        
        self.norm_global = norm_layer(dim)
        self.global_attn = Attention(
            dim, num_heads=num_heads//2,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        
        self.norm_sparse = norm_layer(dim)
        self.sparse_attn = SparseAttention( 
            dim, num_heads=num_heads//2, stride=2,  
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        
        self.norm_window = norm_layer(dim)
        self.window_attn = WindowAttention(
            dim, window_size=(window_size, window_size),
            num_heads=num_heads//2, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)
            
        self.norm_cross = norm_layer(dim)
        self.cross_attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
        self.norm_fusion = norm_layer(dim)
        self.fusion_mlp = Mlp(in_features=dim, hidden_features=int(dim*mlp_ratio), 
                            act_layer=act_layer, drop=drop)
        

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if init_values > 0:
            self.gamma_global = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_sparse = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_window = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_cross = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_fusion = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_global = self.gamma_sparse = self.gamma_window = \
                self.gamma_cross = self.gamma_fusion = None
    
    def forward(self, x, src):
        B, N, C = x.shape
        
        if self.memory_efficient and self.training:
            from torch.utils.checkpoint import checkpoint
            
            norm_x = self.norm_global(x)
            global_out = checkpoint(self.global_attn, norm_x)
            x_global = x + self.drop_path(self.gamma_global * global_out if self.gamma_global is not None else global_out)
            
            norm_x = self.norm_sparse(x_global)
            sparse_out = checkpoint(self.sparse_attn, norm_x)
            x_struct = x_global + self.drop_path(self.gamma_sparse * sparse_out if self.gamma_sparse is not None else sparse_out)
            
            norm_x = self.norm_window(x)
            window_out = checkpoint(self.window_attn, norm_x)
            x_detail = x + self.drop_path(self.gamma_window * window_out if self.gamma_window is not None else window_out)
            
            norm_x = self.norm_cross(x_detail)
            cross_out = checkpoint(self.cross_attn, norm_x, src)
            x_refined = x_detail + self.drop_path(self.gamma_cross * cross_out if self.gamma_cross is not None else cross_out)
            
            x_combined = x_struct + x_refined
            norm_x = self.norm_fusion(x_combined)
            fusion_out = checkpoint(self.fusion_mlp, norm_x)
            x_final = x_combined + self.drop_path(self.gamma_fusion * fusion_out if self.gamma_fusion is not None else fusion_out)
            
        else:
            norm_x = self.norm_global(x)
            global_out = self.global_attn(norm_x)
            x_global = x + self.drop_path(self.gamma_global * global_out if self.gamma_global is not None else global_out)
            
            norm_x = self.norm_sparse(x_global)
            sparse_out = self.sparse_attn(norm_x)
            x_struct = x_global + self.drop_path(self.gamma_sparse * sparse_out if self.gamma_sparse is not None else sparse_out)
            
            norm_x = self.norm_window(x)
            window_out = self.window_attn(norm_x)
            x_detail = x + self.drop_path(self.gamma_window * window_out if self.gamma_window is not None else window_out)
            
            norm_x = self.norm_cross(x_detail)
            cross_out = self.cross_attn(norm_x, src)
            x_refined = x_detail + self.drop_path(self.gamma_cross * cross_out if self.gamma_cross is not None else cross_out)
            
            x_combined = x_struct + x_refined
            norm_x = self.norm_fusion(x_combined)
            fusion_out = self.fusion_mlp(norm_x)
            x_final = x_combined + self.drop_path(self.gamma_fusion * fusion_out if self.gamma_fusion is not None else fusion_out)
            
        return x_final
    
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True,
                attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size if isinstance(window_size, tuple) else (window_size, window_size)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class SparseAttention(nn.Module):
    def __init__(self, dim, num_heads=8, stride=2, qkv_bias=False,
                qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.stride = stride 
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q = qkv[0]
        indices = torch.arange(0, N, self.stride, device=x.device)
        k_sparse = qkv[1][:, :, indices]
        v_sparse = qkv[2][:, :, indices]
        q = q * self.scale
        attn = (q @ k_sparse.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v_sparse).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class LandscapeDecoder(nn.Module):
    def __init__(self, dim=1024, depth=4, num_heads=8, mlp_ratio=2.0, qkv_bias=True,
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                norm_layer=nn.LayerNorm, window_size=16, memory_efficient=True):
        super().__init__()
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        self.blocks = nn.ModuleList([
            LandscapeDecoderBlock(
                dim=dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, 
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],  
                norm_layer=norm_layer, 
                window_size=window_size,
                init_values=0.1, 
                memory_efficient=memory_efficient
            )
            for i in range(depth)
        ])
        
        self.norm = norm_layer(dim)
        
    def __iter__(self):
        return iter(self.blocks)
    
    def __len__(self):
        return len(self.blocks)
    
    def forward(self, x, src=None):
        for block in self.blocks:
            x = block(x, src)
        return self.norm(x)
    
class DualPathAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., 
                proj_drop=0., window_size=16, stride=2):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.stride = stride
        
        self.global_heads = num_heads // 2
        self.local_heads = num_heads - self.global_heads
        
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        self.proj_global = nn.Linear(self.global_heads * head_dim, dim // 2)
        self.proj_local = nn.Linear(self.local_heads * head_dim, dim // 2)
        
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.attn_drop = nn.Dropout(attn_drop)
        
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), self.local_heads))
        
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        trunc_normal_(self.relative_position_bias_table, std=.02)
    
    def _get_sparse_attention(self, q, k, v):
        B, N, H_g, D = q.shape
        
        indices = torch.arange(0, N, self.stride, device=q.device)
        k_sparse = k[:, indices]
        v_sparse = v[:, indices]
        
        q = q * self.scale
        attn = (q @ k_sparse.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        return (attn @ v_sparse)
    
    def _get_window_attention(self, q, k, v):
        B, N, H_l, D = q.shape
        
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, 
            self.window_size * self.window_size, -1)
        
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn + relative_position_bias.unsqueeze(0)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        return (attn @ v)
    
    def forward(self, x):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        
        qkv_global = qkv[:, :, :, :self.global_heads]
        qkv_local = qkv[:, :, :, self.global_heads:]
        
        q_g, k_g, v_g = qkv_global[:, :, 0], qkv_global[:, :, 1], qkv_global[:, :, 2]
        q_l, k_l, v_l = qkv_local[:, :, 0], qkv_local[:, :, 1], qkv_local[:, :, 2]
        
        if self.training:
            from torch.utils.checkpoint import checkpoint
            global_out = checkpoint(self._get_sparse_attention, q_g, k_g, v_g)
        else:
            global_out = self._get_sparse_attention(q_g, k_g, v_g)
        
        if self.training:
            from torch.utils.checkpoint import checkpoint
            local_out = checkpoint(self._get_window_attention, q_l, k_l, v_l)
        else:
            local_out = self._get_window_attention(q_l, k_l, v_l)
        
        global_out = global_out.transpose(1, 2).reshape(B, N, -1)
        local_out = local_out.transpose(1, 2).reshape(B, N, -1)
        
        global_out = self.proj_global(global_out)
        local_out = self.proj_local(local_out)
        
        out = torch.cat([global_out, local_out], dim=-1)
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return out
    
class MemoryEfficientCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def _attn_chunk(self, q, k, v, chunk_size=1024):
        B, N, _ = q.shape
        attn_chunks = []
        
        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)
            q_chunk = q[:, i:end]
            
            attn_chunk = torch.bmm(q_chunk, k.transpose(1, 2)) * self.scale
            attn_chunk = F.softmax(attn_chunk, dim=-1)
            attn_chunk = self.attn_drop(attn_chunk)
            
            output_chunk = torch.bmm(attn_chunk, v)
            attn_chunks.append(output_chunk)
            
        return torch.cat(attn_chunks, dim=1)
    
    def forward(self, x, src):
        B, N, C = x.shape
        B, M, C = src.shape
        
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(src).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(src).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        q = q.reshape(B * self.num_heads, N, C // self.num_heads)
        k = k.reshape(B * self.num_heads, M, C // self.num_heads)
        v = v.reshape(B * self.num_heads, M, C // self.num_heads)
        
        if self.training and (N*M > 2048*2048):
            attn_output = self._attn_chunk(q, k, v)
        else:
            attn = torch.bmm(q, k.transpose(1, 2)) * self.scale
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            
            attn_output = torch.bmm(attn, v)
            
        attn_output = attn_output.reshape(B, self.num_heads, N, C // self.num_heads)
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(B, N, C)
        
        output = self.proj(attn_output)
        output = self.proj_drop(output)
        
        return output
    
