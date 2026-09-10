import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.nn import ReLU, LeakyReLU
import sys
import itertools
from functools import partial
# import pytorch_wavelets.dwt.transform3d as dwt3d
# from kymatio.torch import Scattering3D

from . import block

# sys.path.append('/home/boys/project/voxelmorph/voxelmorph_code/voxelmorph/torch')
# from typing import Optional
# from torch import nn, Tensor
# from torch.nn.init import trunc_normal_

# # from .context import is_fna_enabled
# # from .experimental import na3d as experimental_na3d
# from .functional import na3d, na3d_av, na3d_qk
# from .types import CausalArg3DTypeOrDed, Dimension3DTypeOrDed

from .. import default_unet_features
from . import layers
from .modelio import LoadableModel, store_config_args
# from patchify import patchify
import random
from timm.models.layers import DropPath, trunc_normal_, to_3tuple
import torch.utils.checkpoint as checkpoint
from functools import reduce, lru_cache
from operator import mul
from einops import rearrange
import einops
# from mamba_ssm import Mamba
from timm.models.swin_transformer import swin_large_patch4_window7_224,swin_large_patch4_window12_384


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, kernel_size=3,stride=1,padding = 1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, kernel_size, stride, padding=padding)
        # self.norm = nn.InstanceNorm3d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        # out = self.norm(out)
        out = self.activation(out)
        return out


def compute_mask(dims, window_size, shift_size, device): #([80,100,115]),([5,5,5]) ([2,2,2])
    
    cnt = 0
    d, h, w = dims
    img_mask = torch.zeros((1, d, h, w, 1), device=device)
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1

    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask


def window_partition(x_in, window_size):

    b, d, h, w, c = x_in.shape
    x = x_in.view(b,
                  d // window_size[0],
                  window_size[0],
                  h // window_size[1],
                  window_size[1],
                  w // window_size[2],
                  window_size[2],
                  c)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0] * window_size[1] * window_size[2], c)
        
    return windows


def window_reverse(windows, window_size, dims):

    b, d, h, w = dims
    x = windows.view(b,
                     d // window_size[0],
                     h // window_size[1],
                     w // window_size[2],
                     window_size[0],
                     window_size[1],
                     window_size[2],
                     -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d, h, w, -1)

    return x


def get_window_size(x_size, window_size, shift_size=None):

    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)
    
    
def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
        
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b) 
        return tensor
    

    
    
class PatchExpanding_block(nn.Module):
    
    def __init__(self, embed_dim: int):
        super().__init__()
        
        self.up_conv = nn.ConvTranspose3d(embed_dim, embed_dim//2, kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(embed_dim//2)

    def forward(self, x_in):

        x = self.up_conv(x_in)
        x = einops.rearrange(x, 'b c d h w -> b d h w c')
        x = self.norm(x)
        x_out = einops.rearrange(x, 'b d h w c -> b c d h w')
        
        return x_out
    

class SwinTrans_stage_block(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 num_layers: int,
                 num_heads: int,
                 window_size: list,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 drop: float = 0.0,
                 attn_drop: float = 0.0,
                 use_checkpoint: bool = False):
        super().__init__()
        
        self.window_size = window_size  # 5
        self.shift_size = tuple(i // 2 for i in window_size)  # 2
        self.no_shift = tuple(0 for i in window_size)  # 0
        
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            block = SwinTrans_Block(embed_dim=embed_dim,
                                    num_heads=num_heads,
                                    window_size=self.window_size,
                                    shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias,
                                    drop=drop,
                                    attn_drop=attn_drop,
                                    use_checkpoint=use_checkpoint)
            self.blocks.append(block)
        
    def forward(self, x_in):
        
        b, c, d, h, w = x_in.shape  # 1,16,80,96,112
        window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)  # ([5,5,5]) ([2,2,2])
        dp = int(np.ceil(d / window_size[0])) * window_size[0] # 80
        hp = int(np.ceil(h / window_size[1])) * window_size[1] # 100
        wp = int(np.ceil(w / window_size[2])) * window_size[2] # 115
        attn_mask = compute_mask([dp, hp, wp], window_size, shift_size, x_in.device)
        
        x = einops.rearrange(x_in, 'b c d h w -> b d h w c')
        for block in self.blocks:
            x = block(x, mask_matrix=attn_mask)
        x_out = einops.rearrange(x, 'b d h w c -> b c d h w')

        return x_out
    

class SwinTrans_Block(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 window_size: list,
                 shift_size: list,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 drop: float = 0.0,
                 attn_drop: float = 0.0,
                 use_checkpoint: bool = False):
        super().__init__()
                         
        self.window_size = window_size
        self.shift_size = shift_size
        self.use_checkpoint = use_checkpoint
                         
        self.norm1 = nn.LayerNorm(embed_dim)  
        self.attn = MSA_block(embed_dim,
                              window_size=window_size,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              attn_drop=attn_drop,
                              proj_drop=drop)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP_block(hidden_size=embed_dim, 
                             mlp_dim=int(embed_dim * mlp_ratio), 
                             dropout_rate=drop)

    def forward_part1(self, x_in, mask_matrix):
        
        x = self.norm1(x_in)
        
        b, d, h, w, c = x.shape
        window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - h % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - w % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, dp, hp, wp, _ = x.shape
        dims = [b, dp, hp, wp]
        
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None  
         
        x_windows = window_partition(shifted_x, window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        
        attn_windows = attn_windows.view(-1, *(window_size + (c,)))
        shifted_x = window_reverse(attn_windows, window_size, dims)
        
        if any(i > 0 for i in shift_size):
            x_out = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x_out = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x_out = x_out[:, :d, :h, :w, :].contiguous()

        return x_out

    def forward_part2(self, x_in):
        
        x = self.norm2(x_in)
        x_out = self.mlp(x)
        return x_out

    def forward(self, x_in, mask_matrix=None):
        
        if self.use_checkpoint and x_in.requires_grad:
            x = x_in + checkpoint.checkpoint(self.forward_part1, x_in, mask_matrix)
        else:
            x = x_in + self.forward_part1(x_in, mask_matrix)
                         
        if self.use_checkpoint and x.requires_grad:
            x_out = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x_out = x + self.forward_part2(x)
        
        return x_out


class MSA_block(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 window_size: list,
                 qkv_bias: bool = False,
                 attn_drop: float = 0.0,
                 proj_drop: float = 0.0):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = head_dim**-0.5
        mesh_args = torch.meshgrid.__kwdefaults__

        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * self.window_size[0] - 1) * 
                                                                     (2 * self.window_size[1] - 1) * 
                                                                     (2 * self.window_size[2] - 1), num_heads))
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        if mesh_args is not None:
            coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
        else:
            coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.Softmax = nn.Softmax(dim=-1)

    def forward(self, x_in, mask=None):
        
        b, n, c = x_in.shape
        qkv = self.qkv(x_in).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.clone()[:n, :n].reshape(-1)
        ].reshape(n, n, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
        attn = self.Softmax(attn)
        attn = self.attn_drop(attn).to(v.dtype)
        
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x_out = self.proj_drop(x)
        
        return x_out
    
    
class MLP_block(nn.Module):

    def __init__(self, hidden_size: int, mlp_dim: int, dropout_rate: float = 0.0):
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
        
        self.linear1 = nn.Linear(hidden_size, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim, hidden_size)
        
        self.drop1 = nn.Dropout(dropout_rate)
        self.drop2 = nn.Dropout(dropout_rate)
        
        self.GELU = nn.GELU()

    def forward(self, x_in):
        
        x = self.linear1(x_in)
        x = self.GELU(x)
        x = self.drop1(x)
        
        x = self.linear2(x)
        x_out = self.drop2(x)
        
        return x_out


# ------------------------------  以下是正式实验：-------------------------------


class GatingFunction(nn.Module):
    def __init__(self, epsilon=0.1,decay=1):
        super(GatingFunction, self).__init__()
        # epsilon 是一个小常数，以防止除零错误
        self.epsilon = epsilon
        self.decay = decay

    def forward(self, x):
        # 计算门控函数
        epsilon = self.epsilon * self.decay
        # print(epsilon)
        return x**2 / (x**2 + epsilon)

#(with FFM) 直接融合 + 俩卷积 (剪枝后的网络) encoder和FFM 都×3
class dual_pyramid_VxmDense_FFM_huge(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    自己写的, 两个权重共享的编码器来各自提取img_concat_wavelet的特征,
    然后特征融合, 再送入同一个解码器
    在multi-vxmdense的基础上改的

    配准网络本身配准的是输入model的source和target的尺寸, 而SPT则是new_shape的尺寸
    特别需要注意的就是输入的source和target的尺寸是否能完成四次下采样, 不能的话要注意调整网络下采样的次数
    最后生成的flow可以通过self.flow里面的卷积步长stride来改变尺寸(下采样等等)
    """

    @store_config_args
    def __init__(self,
                 inshape=(192, 160, 192)):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this 
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. 
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2. 
                Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)

        # # print(new_shape)
        # self.transformer1 = layers.SpatialTransformer((20,24,28))
        # self.transformer2 = layers.SpatialTransformer((40,48,56))
        # self.transformer3 = layers.SpatialTransformer((80,96,112))
        # self.transformer4 = layers.SpatialTransformer((160,192,224))

        # self.transformer1 = layers.SpatialTransformer((24,20,24))
        # self.transformer2 = layers.SpatialTransformer((48,40,48))
        # self.transformer3 = layers.SpatialTransformer((96,80,96))
        # self.transformer4 = layers.SpatialTransformer((192, 160, 192))

        self.transformer1 = layers.SpatialTransformer((16,16,16))
        self.transformer2 = layers.SpatialTransformer((32,32,32))
        self.transformer3 = layers.SpatialTransformer((64,64,64))
        self.transformer4 = layers.SpatialTransformer((128,128,128))

        # self.transformer1 = layers.SpatialTransformer((24,20,20))
        # self.transformer2 = layers.SpatialTransformer((48,40,40))
        # self.transformer3 = layers.SpatialTransformer((96,80,80))
        # self.transformer4 = layers.SpatialTransformer((192, 160, 160))
        # self.input_model = input_model
        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = MaxPooling(2)
        self.pooling_4 =MaxPooling(4)
        self.pooling_8 =MaxPooling(8)

        self.upsampling = nn.Upsample(scale_factor=2, mode='trilinear') 


        self.encoder1_m = ConvBlock(3,1,48)  # 权重共享的编码器, 提取moving
        self.encoder2_m = ConvBlock(3,48,96)
        self.encoder3_m = ConvBlock(3,96,96)
        self.encoder4_m = ConvBlock(3,96,96)
        self.encoder5_m = ConvBlock(3,96,96)

        
        self.encoder1_f = self.encoder1_m  # 权重共享的编码器, 提取fixed
        self.encoder2_f = self.encoder2_m
        self.encoder3_f = self.encoder3_m
        self.encoder4_f = self.encoder4_m
        self.encoder5_f = self.encoder5_m

        
        self.decoder1 = ConvBlock(ndims,192,32)
        self.decoder2 = ConvBlock(ndims,224,32)
        self.decoder3 = ConvBlock(ndims,224,32)
        self.decoder4 = ConvBlock(ndims,224,16)
        self.decoder5 = ConvBlock(ndims,112,16)

        self.output_block = nn.Sequential(ConvBlock(ndims,16,16),ConvBlock(ndims,16,16))
        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(16, ndims, kernel_size=3, padding=1)

        # # init flow layer with small weights and bias
        # self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        # self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        self.output_block_0 = nn.Sequential(ConvBlock(ndims,32,16),ConvBlock(ndims,16,3))
        self.output_block_1 = nn.Sequential(ConvBlock(ndims,32,16),ConvBlock(ndims,16,3))
        self.output_block_2 = nn.Sequential(ConvBlock(ndims,32,16),ConvBlock(ndims,16,3))
        self.output_block_3 = nn.Sequential(ConvBlock(ndims,16,16),ConvBlock(ndims,16,3))

        # 整合特征
        # # ------------------------------Scale4--------------------------------------------
        # self.fusion_f1_scale4 = ConvBlock(ndims,16,16)
        # self.fusion_f2_scale4 = ConvBlock(ndims,32,32)
        # self.fusion_f3_scale4 = ConvBlock(ndims,32,32)
        # self.fusion_f4_scale4 = ConvBlock(ndims,32,32)
        self.fusion_f_scale4 = ConvBlock(ndims,336,96)
        self.fusion_f_scale4_2 = ConvBlock(ndims,96,96)

        # self.fusion_m1_scale4 = self.fusion_f1_scale4
        # self.fusion_m2_scale4 = self.fusion_f2_scale4
        # self.fusion_m3_scale4 = self.fusion_f3_scale4
        # self.fusion_m4_scale4 = self.fusion_f4_scale4
        self.fusion_m_scale4 = self.fusion_f_scale4
        self.fusion_m_scale4_2 = self.fusion_f_scale4_2

        # # ------------------------------Scale3--------------------------------------------
        # self.fusion_f1_scale3 = ConvBlock(ndims,16,16)
        # self.fusion_f2_scale3 = ConvBlock(ndims,32,32)
        # self.fusion_f3_scale3 = ConvBlock(ndims,32,32)
        # self.fusion_f4_scale3 = ConvBlock(ndims,32,32)
        self.fusion_f_scale3 = ConvBlock(ndims,336,96)
        self.fusion_f_scale3_2 = ConvBlock(ndims,96,96)

        # self.fusion_m1_scale3 = self.fusion_f1_scale3
        # self.fusion_m2_scale3 = self.fusion_f2_scale3
        # self.fusion_m3_scale3 = self.fusion_f3_scale3
        # self.fusion_m4_scale3 = self.fusion_f4_scale3
        self.fusion_m_scale3 = self.fusion_f_scale3
        self.fusion_m_scale3_2 = self.fusion_f_scale3_2

        # # ------------------------------Scale2--------------------------------------------
        # self.fusion_f1_scale2 = ConvBlock(ndims,16,16)
        # self.fusion_f2_scale2 = ConvBlock(ndims,32,32)
        # self.fusion_f3_scale2 = ConvBlock(ndims,32,32)
        # self.fusion_f4_scale2 = ConvBlock(ndims,32,32)
        self.fusion_f_scale2 = ConvBlock(ndims,336,96)
        self.fusion_f_scale2_2 = ConvBlock(ndims,96,96)

        # self.fusion_m1_scale2 = self.fusion_f1_scale2
        # self.fusion_m2_scale2 = self.fusion_f2_scale2
        # self.fusion_m3_scale2 = self.fusion_f3_scale2
        # self.fusion_m4_scale2 = self.fusion_f4_scale2
        self.fusion_m_scale2 = self.fusion_f_scale2
        self.fusion_m_scale2_2 = self.fusion_f_scale2_2

        
    def resblock_seq(self, in_channels, out_channels, bias_opt=False):
        '''
        一个resblock_seq作为一个编码器, 不需要太多,
        借鉴LapIRN
        '''
        layer = nn.Sequential(
            PreActBlock(in_channels, out_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
        )
        return layer
    
    
    
    def forward(self, source, target):
        '''
        Parameters:
            source: Source image tensor.F (moving)
            target: Target image tensor.  (fixed)
            registration: Return transformed image and flow. Default is False.
        '''
        xm = source
        xf = target
        xm_skip_1 = self.encoder1_m(xm) # (1,16,160,192,224)
        xf_skip_1 = self.encoder1_f(xf) # (1,16,160,192,224)

        # moving encoder
        #下采样到1/2,第二个encoder
        xm = self.pooling(xm_skip_1) # (1,16,80,96,112)
        xm_skip_2 = self.encoder2_m(xm) # (1,32,80,96,112)
        # 下采样到1/4,第三个encoder
        xm = self.pooling(xm_skip_2) # (1,32,40,48,56)
        xm_skip_3 = self.encoder3_m(xm) # (1,32,40,48,56)
        # 下采样到1/8,第四个encoder
        xm = self.pooling(xm_skip_3)  # (1,32,20,24,28)
        xm_skip_4 = self.encoder4_m(xm) # (1,32,20,24,28)
        # 下采样到1/16
        xm = self.pooling(xm_skip_4) # (1,32,10,12,14)
        xm = self.encoder5_m(xm) # (1,32,10,12,14)

        features_xm = [xm_skip_1,xm_skip_2,xm_skip_3,xm_skip_4,xm]


        # fixed encoder
        #下采样到1/2,第二个encoder
        xf = self.pooling(xf_skip_1)
        xf_skip_2 = self.encoder2_f(xf)
        # 下采样到1/4,第三个encoder
        xf = self.pooling(xf_skip_2)
        xf_skip_3 = self.encoder3_f(xf)
        # 下采样到1/8,第四个encoder
        xf = self.pooling(xf_skip_3)  
        xf_skip_4 = self.encoder4_f(xf)
        # 下采样到1/16
        xf = self.pooling(xf_skip_4)
        xf = self.encoder5_f(xf) # (1,32,10,12,14)

        features_xf = [xf_skip_1,xf_skip_2,xf_skip_3,xf_skip_4,xf]

        #-------------------------------Scale5--------------------------------
        # 第一个decoder
        x = torch.cat([xm, xf], dim=1) # (1,64,10,12,14)
        x = self.decoder1(x) # (1,32,10,12,14)


        # ------------------------------Scale4--------------------------------------------
        # 融合多尺度特征：(20,24,28)
        xm_skip_1_fusion4 = self.pooling_8(xm_skip_1) # (1,16,20,24,28)
        # xm_skip_1_fusion4 = self.fusion_m1_scale4(xm_skip_1_fusion4) 
        xm_skip_2_fusion4 = self.pooling_4(xm_skip_2) # (1,32,20,24,28)
        # xm_skip_2_fusion4 = self.fusion_m2_scale4(xm_skip_2_fusion4)
        xm_skip_3_fusion4 = self.pooling(xm_skip_3) # (1,32,20,24,28)
        # xm_skip_3_fusion4 = self.fusion_m3_scale4(xm_skip_3_fusion4)
        xm_skip_4_fusion4 = xm_skip_4 # (1,32,20,24,28)

        xm_fusion4 = torch.cat([xm_skip_1_fusion4, xm_skip_2_fusion4], dim=1)
        xm_fusion4 = torch.cat([xm_fusion4, xm_skip_3_fusion4], dim=1)
        xm_fusion4 = torch.cat([xm_fusion4, xm_skip_4_fusion4], dim=1)

        xm_fusion4 = self.fusion_m_scale4(xm_fusion4) # (1,32,20,24,28)
        xm_fusion4 = self.fusion_m_scale4_2(xm_fusion4) # (1,32,20,24,28)
    

        xf_skip_1_fusion4 = self.pooling_8(xf_skip_1) # (1,16,20,24,28)
        # xf_skip_1_fusion4 = self.fusion_f1_scale4(xf_skip_1_fusion4)
        xf_skip_2_fusion4 = self.pooling_4(xf_skip_2) # (1,32,20,24,28)
        # xf_skip_2_fusion4 = self.fusion_f2_scale4(xf_skip_2_fusion4)
        xf_skip_3_fusion4 = self.pooling(xf_skip_3) # (1,32,20,24,28)
        # xf_skip_3_fusion4 = self.fusion_f3_scale4(xf_skip_3_fusion4)
        xf_skip_4_fusion4 = xf_skip_4 # (1,32,20,24,28)

        xf_fusion4 = torch.cat([xf_skip_1_fusion4, xf_skip_2_fusion4], dim=1) 
        xf_fusion4 = torch.cat([xf_fusion4, xf_skip_3_fusion4], dim=1)
        xf_fusion4 = torch.cat([xf_fusion4, xf_skip_4_fusion4], dim=1)
   
        xf_fusion4 = self.fusion_f_scale4(xf_fusion4)
        xf_fusion4 = self.fusion_f_scale4_2(xf_fusion4) # (1,32,20,24,28) 

        
        x_fusion4 = torch.cat([xm_fusion4, xf_fusion4], dim=1) # (1,64,20,24,28)

    
        # 第二个decoder
        x = self.decoder2(torch.cat([self.upsampling(x), x_fusion4], dim=1)) # (1,32,20,24,28)
        # x = self.trans_3(x)

        # ----输出1/8分辨率的变形场-----
        flow_1 = self.output_block_1(x) # (1,3,20,24,28) 
        flow_1_up = nn.functional.interpolate(flow_1, scale_factor=2,mode="trilinear")*2 # (1,3,40,48,56)


        ## ------------------------------Scale3--------------------------------------------
        # 融合多尺度特征：(40,48,56)
        xm_skip_1_fusion3 = self.pooling_4(xm_skip_1) # (1,16,40,48,56)
        # xm_skip_1_fusion3 = self.fusion_m1_scale3(xm_skip_1_fusion3) 
        xm_skip_2_fusion3 = self.pooling(xm_skip_2) # (1,32,40,48,56)
        # xm_skip_2_fusion3 = self.fusion_m2_scale3(xm_skip_2_fusion3)
        xm_skip_3_fusion3 = xm_skip_3 # (1,32,40,48,56)
        # xm_skip_3_fusion3 = self.fusion_m3_scale3(xm_skip_3_fusion3)

        # xm_skip_4_fusion3 = self.fusion_m4_scale3(xm_skip_4) # (1,32,40,48,56)
        xm_skip_4_fusion3 = nn.functional.interpolate(xm_skip_4, scale_factor=2,mode="trilinear")
        

        xm_fusion3 = torch.cat([xm_skip_1_fusion3, xm_skip_2_fusion3], dim=1)
        xm_fusion3 = torch.cat([xm_fusion3, xm_skip_3_fusion3], dim=1)
        xm_fusion3 = torch.cat([xm_fusion3, xm_skip_4_fusion3], dim=1) # (1,112,40,48,56)

        xm_fusion3 = self.fusion_m_scale3(xm_fusion3) # (1,32,40,48,56)
        xm_fusion3 = self.fusion_m_scale3_2(xm_fusion3) # (1,32,40,48,56)
        # print("xm_fusion3",xm_fusion3.shape)

        xw_fusion3 = self.transformer2(xm_fusion3, flow_1_up)


        xf_skip_1_fusion3 = self.pooling_4(xf_skip_1) # (1,16,40,48,56)
        # xf_skip_1_fusion3 = self.fusion_f1_scale3(xf_skip_1_fusion3) 
        xf_skip_2_fusion3 = self.pooling(xf_skip_2) # (1,32,40,48,56)
        # xf_skip_2_fusion3 = self.fusion_f2_scale3(xf_skip_2_fusion3)
        xf_skip_3_fusion3 = xf_skip_3 # (1,32,40,48,56)
        # xf_skip_3_fusion3 = self.fusion_f3_scale3(xf_skip_3_fusion3)

        # xf_skip_4_fusion3 = self.fusion_f4_scale3(xf_skip_4) # (1,32,40,48,56)
        xf_skip_4_fusion3 = nn.functional.interpolate(xf_skip_4, scale_factor=2,mode="trilinear")
        

        xf_fusion3 = torch.cat([xf_skip_1_fusion3, xf_skip_2_fusion3], dim=1)
        xf_fusion3 = torch.cat([xf_fusion3, xf_skip_3_fusion3], dim=1)
        xf_fusion3 = torch.cat([xf_fusion3, xf_skip_4_fusion3], dim=1) # (1,112,40,48,56)

        xf_fusion3 = self.fusion_f_scale3(xf_fusion3) # (1,32,40,48,56)
        xf_fusion3 = self.fusion_f_scale3_2(xf_fusion3) # (1,32,40,48,56)

        # concat
        x_fusion3 = torch.cat([xw_fusion3, xf_fusion3], dim=1)  # (1,64,40,48,56)

        # 第三个decoder
        x = self.decoder3(torch.cat([self.upsampling(x), x_fusion3], dim=1)) # (1,32,40,48,56)
        # x = self.trans_4(x)

        # ----输出1/4分辨率的变形场---
        delta_flow_2 = self.output_block_2(x) # (1,3,40,48,56)
        flow_2 = delta_flow_2 + flow_1_up # (1,3,40,48,56)
        flow_2_up = nn.functional.interpolate(flow_2, scale_factor=2,mode="trilinear")*2 # (1,3,80,96,112)

        ## ------------------------------Scale2--------------------------------------------
        # 融合多尺度特征：(80,96,112)
        xm_skip_1_fusion2 = self.pooling(xm_skip_1) # (1,16,80,96,112)
        # xm_skip_1_fusion2 = self.fusion_m1_scale2(xm_skip_1_fusion2) 
        xm_skip_2_fusion2 = xm_skip_2 # (1,32,80,96,112)
        # xm_skip_2_fusion2 = self.fusion_m2_scale2(xm_skip_2_fusion2)

        # xm_skip_3_fusion2 = self.fusion_m3_scale2(xm_skip_3)
        xm_skip_3_fusion2 = nn.functional.interpolate(xm_skip_3, scale_factor=2,mode="trilinear")  # (1,32,80,96,112)
        
        # xm_skip_4_fusion2 = self.fusion_m4_scale2(xm_skip_4) # (1,32,80,96,112)
        xm_skip_4_fusion2 = nn.functional.interpolate(xm_skip_4, scale_factor=4,mode="trilinear")
        

        xm_fusion2 = torch.cat([xm_skip_1_fusion2, xm_skip_2_fusion2], dim=1)
        xm_fusion2 = torch.cat([xm_fusion2, xm_skip_3_fusion2], dim=1)
        xm_fusion2 = torch.cat([xm_fusion2, xm_skip_4_fusion2], dim=1) # (1,112,80,96,112)


        xm_fusion2 = self.fusion_m_scale2(xm_fusion2) # (1,32,80,96,112)
        xm_fusion2 = self.fusion_m_scale2_2(xm_fusion2) # (1,32,80,96,112)


        xw_fusion2 = self.transformer3(xm_fusion2, flow_2_up)

        xf_skip_1_fusion2 = self.pooling(xf_skip_1) # (1,16,80,96,112)
        # xf_skip_1_fusion2 = self.fusion_f1_scale2(xf_skip_1_fusion2) 
        xf_skip_2_fusion2 = xf_skip_2 # (1,32,80,96,112)
        # xf_skip_2_fusion2 = self.fusion_f2_scale2(xf_skip_2_fusion2)

        # xf_skip_3_fusion2 = self.fusion_f3_scale2(xf_skip_3)
        xf_skip_3_fusion2 = nn.functional.interpolate(xf_skip_3, scale_factor=2,mode="trilinear")  # (1,32,80,96,112)
        
        # xf_skip_4_fusion2 = self.fusion_f4_scale2(xf_skip_4) # (1,32,80,96,112)
        xf_skip_4_fusion2 = nn.functional.interpolate(xf_skip_4, scale_factor=4,mode="trilinear")
        

        xf_fusion2 = torch.cat([xf_skip_1_fusion2, xf_skip_2_fusion2], dim=1)
        xf_fusion2 = torch.cat([xf_fusion2, xf_skip_3_fusion2], dim=1)
        xf_fusion2 = torch.cat([xf_fusion2, xf_skip_4_fusion2], dim=1) # (1,112,80,96,112)

        xf_fusion2 = self.fusion_f_scale2(xf_fusion2) # (1,32,80,96,112)
        xf_fusion2 = self.fusion_f_scale2_2(xf_fusion2) # (1,32,80,96,112)

        # concat
        x_fusion2 = torch.cat([xw_fusion2, xf_fusion2], dim=1)  # (1,32,80,96,112)


        # 第四个decoder
        x = self.decoder4(torch.cat([self.upsampling(x), x_fusion2], dim=1)) # (1,32,80,96,112)
        # x = self.trans_5(x)
        # print(x.shape)

        # ----输出1/2分辨率的变形场-------
        delta_flow_3 = self.output_block_3(x) # (1,3,80,96,112)
        flow_3 = delta_flow_3 + flow_2_up # (1,3,80,96,112)
        flow_3_up = nn.functional.interpolate(flow_3, scale_factor=2,mode="trilinear")*2 # (1,3,160,112,224)

        # ------------------------------Scale1--------------------------------------------
        # # 对moving feature进行warp得到moved feature
        xw_skip_1 = self.transformer4(xm_skip_1, flow_3_up)
        # concat
        x_skip_1 = torch.cat([xw_skip_1, xf_skip_1], dim=1)  # (1,32,160,192,224)
        # print(x_skip_1.shape)


        # 第四个decoder
        x = self.decoder5(torch.cat([self.upsampling(x), x_skip_1], dim=1)) # (1,32,160,192,224)

        # ----------------------输出最终的变形场------------------------------------------
        # output block
        x = self.output_block(x) # (1,16,160,192,224)
        # 生成flow场
        delta_flow_final = self.flow(x) # (1,3,160,192,224)
        flow_final = delta_flow_final + flow_3_up
            

        return flow_1,flow_2,flow_3,flow_final,delta_flow_2,delta_flow_3,delta_flow_final,features_xf,features_xm

#(with FFM) 直接融合 + 俩卷积 维度扩展 encoder和FFM 都×2(剪枝前的网络)
class dual_pyramid_VxmDense_FFM_large(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    自己写的, 两个权重共享的编码器来各自提取img_concat_wavelet的特征,
    然后特征融合, 再送入同一个解码器
    在multi-vxmdense的基础上改的

    配准网络本身配准的是输入model的source和target的尺寸, 而SPT则是new_shape的尺寸
    特别需要注意的就是输入的source和target的尺寸是否能完成四次下采样, 不能的话要注意调整网络下采样的次数
    最后生成的flow可以通过self.flow里面的卷积步长stride来改变尺寸(下采样等等)
    """

    @store_config_args
    def __init__(self,
                 inshape=(192, 160, 192)):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this 
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. 
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2. 
                Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)

        # # print(new_shape)
        self.transformer1 = layers.SpatialTransformer((16,16,16))
        self.transformer2 = layers.SpatialTransformer((32,32,32))
        self.transformer3 = layers.SpatialTransformer((64,64,64))
        self.transformer4 = layers.SpatialTransformer((128,128,128))

        # self.transformer1 = layers.SpatialTransformer((20,24,28))
        # self.transformer2 = layers.SpatialTransformer((40,48,56))
        # self.transformer3 = layers.SpatialTransformer((80,96,112))
        # self.transformer4 = layers.SpatialTransformer((160,192,224))

        # self.transformer1 = layers.SpatialTransformer((24,20,24))
        # self.transformer2 = layers.SpatialTransformer((48,40,48))
        # self.transformer3 = layers.SpatialTransformer((96,80,96))
        # self.transformer4 = layers.SpatialTransformer((192, 160, 192))

        # self.transformer1 = layers.SpatialTransformer((24,20,20))
        # self.transformer2 = layers.SpatialTransformer((48,40,40))
        # self.transformer3 = layers.SpatialTransformer((96,80,80))
        # self.transformer4 = layers.SpatialTransformer((192, 160, 160))

        # self.input_model = input_model
        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = MaxPooling(2)
        self.pooling_4 =MaxPooling(4)
        self.pooling_8 =MaxPooling(8)

        self.upsampling = nn.Upsample(scale_factor=2, mode='trilinear') 


        self.encoder1_m = ConvBlock(3,1,32)  # 权重共享的编码器, 提取moving
        self.encoder2_m = ConvBlock(3,32,64)
        self.encoder3_m = ConvBlock(3,64,64)
        self.encoder4_m = ConvBlock(3,64,64)
        self.encoder5_m = ConvBlock(3,64,64)

        
        self.encoder1_f = self.encoder1_m  # 权重共享的编码器, 提取fixed
        self.encoder2_f = self.encoder2_m
        self.encoder3_f = self.encoder3_m
        self.encoder4_f = self.encoder4_m
        self.encoder5_f = self.encoder5_m

        
        self.decoder1 = ConvBlock(ndims,128,32)
        self.decoder2 = ConvBlock(ndims,160,32)
        self.decoder3 = ConvBlock(ndims,160,32)
        self.decoder4 = ConvBlock(ndims,160,16)
        self.decoder5 = ConvBlock(ndims,80,16)

        self.output_block = nn.Sequential(ConvBlock(ndims,16,16),ConvBlock(ndims,16,16))
        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(16, ndims, kernel_size=3, padding=1)

        # # init flow layer with small weights and bias
        # self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        # self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        self.output_block_0 = nn.Sequential(ConvBlock(ndims,32,16),ConvBlock(ndims,16,3))
        self.output_block_1 = nn.Sequential(ConvBlock(ndims,32,16),ConvBlock(ndims,16,3))
        self.output_block_2 = nn.Sequential(ConvBlock(ndims,32,16),ConvBlock(ndims,16,3))
        self.output_block_3 = nn.Sequential(ConvBlock(ndims,16,16),ConvBlock(ndims,16,3))

        # 整合特征
        # # ------------------------------Scale4--------------------------------------------
        # self.fusion_f1_scale4 = ConvBlock(ndims,16,16)
        # self.fusion_f2_scale4 = ConvBlock(ndims,32,32)
        # self.fusion_f3_scale4 = ConvBlock(ndims,32,32)
        # self.fusion_f4_scale4 = ConvBlock(ndims,32,32)
        self.fusion_f_scale4 = ConvBlock(ndims,224,64)
        self.fusion_f_scale4_2 = ConvBlock(ndims,64,64)

        # self.fusion_m1_scale4 = self.fusion_f1_scale4
        # self.fusion_m2_scale4 = self.fusion_f2_scale4
        # self.fusion_m3_scale4 = self.fusion_f3_scale4
        # self.fusion_m4_scale4 = self.fusion_f4_scale4
        self.fusion_m_scale4 = self.fusion_f_scale4
        self.fusion_m_scale4_2 = self.fusion_f_scale4_2

        # # ------------------------------Scale3--------------------------------------------
        # self.fusion_f1_scale3 = ConvBlock(ndims,16,16)
        # self.fusion_f2_scale3 = ConvBlock(ndims,32,32)
        # self.fusion_f3_scale3 = ConvBlock(ndims,32,32)
        # self.fusion_f4_scale3 = ConvBlock(ndims,32,32)
        self.fusion_f_scale3 = ConvBlock(ndims,224,64)
        self.fusion_f_scale3_2 = ConvBlock(ndims,64,64)

        # self.fusion_m1_scale3 = self.fusion_f1_scale3
        # self.fusion_m2_scale3 = self.fusion_f2_scale3
        # self.fusion_m3_scale3 = self.fusion_f3_scale3
        # self.fusion_m4_scale3 = self.fusion_f4_scale3
        self.fusion_m_scale3 = self.fusion_f_scale3
        self.fusion_m_scale3_2 = self.fusion_f_scale3_2

        # # ------------------------------Scale2--------------------------------------------
        # self.fusion_f1_scale2 = ConvBlock(ndims,16,16)
        # self.fusion_f2_scale2 = ConvBlock(ndims,32,32)
        # self.fusion_f3_scale2 = ConvBlock(ndims,32,32)
        # self.fusion_f4_scale2 = ConvBlock(ndims,32,32)
        self.fusion_f_scale2 = ConvBlock(ndims,224,64)
        self.fusion_f_scale2_2 = ConvBlock(ndims,64,64)

        # self.fusion_m1_scale2 = self.fusion_f1_scale2
        # self.fusion_m2_scale2 = self.fusion_f2_scale2
        # self.fusion_m3_scale2 = self.fusion_f3_scale2
        # self.fusion_m4_scale2 = self.fusion_f4_scale2
        self.fusion_m_scale2 = self.fusion_f_scale2
        self.fusion_m_scale2_2 = self.fusion_f_scale2_2

        
    def resblock_seq(self, in_channels, out_channels, bias_opt=False):
        '''
        一个resblock_seq作为一个编码器, 不需要太多,
        借鉴LapIRN
        '''
        layer = nn.Sequential(
            PreActBlock(in_channels, out_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
        )
        return layer
    
    
    
    def forward(self, source, target):
        '''
        Parameters:
            source: Source image tensor.F (moving)
            target: Target image tensor.  (fixed)
            registration: Return transformed image and flow. Default is False.
        '''
        xm = source
        xf = target
        xm_skip_1 = self.encoder1_m(xm) # (1,32,160,192,224)
        xf_skip_1 = self.encoder1_f(xf) # (1,32,160,192,224)

        # moving encoder
        #下采样到1/2,第二个encoder
        xm = self.pooling(xm_skip_1) # (1,32,80,96,112)
        xm_skip_2 = self.encoder2_m(xm) # (1,64,80,96,112)
        # 下采样到1/4,第三个encoder
        xm = self.pooling(xm_skip_2) # (1,64,40,48,56)
        xm_skip_3 = self.encoder3_m(xm) # (1,64,40,48,56)
        # 下采样到1/8,第四个encoder
        xm = self.pooling(xm_skip_3)  # (1,64,20,24,28)
        xm_skip_4 = self.encoder4_m(xm) # (1,64,20,24,28)
        # 下采样到1/16
        xm = self.pooling(xm_skip_4) # (1,64,10,12,14)
        xm_skip_5 = self.encoder5_m(xm) # (1,64,10,12,14)


        # fixed encoder
        #下采样到1/2,第二个encoder
        xf = self.pooling(xf_skip_1)
        xf_skip_2 = self.encoder2_f(xf)
        # 下采样到1/4,第三个encoder
        xf = self.pooling(xf_skip_2)
        xf_skip_3 = self.encoder3_f(xf)
        # 下采样到1/8,第四个encoder
        xf = self.pooling(xf_skip_3)  
        xf_skip_4 = self.encoder4_f(xf)
        # 下采样到1/16
        xf = self.pooling(xf_skip_4)
        xf_skip_5 = self.encoder5_f(xf) # (1,64,10,12,14)

        

        #-------------------------------Scale5--------------------------------
        # 第一个decoder
        x = torch.cat([xm_skip_5, xf_skip_5], dim=1) # (1,128,10,12,14)
        x = self.decoder1(x) # (1,64,10,12,14)
        x_decoder1_feature = x


        # ------------------------------Scale4--------------------------------------------
        # 融合多尺度特征：(20,24,28)
        xm_skip_1_fusion4 = self.pooling_8(xm_skip_1) # (1,32,20,24,28)
        # xm_skip_1_fusion4 = self.fusion_m1_scale4(xm_skip_1_fusion4) 
        xm_skip_2_fusion4 = self.pooling_4(xm_skip_2) # (1,64,20,24,28)
        # xm_skip_2_fusion4 = self.fusion_m2_scale4(xm_skip_2_fusion4)
        xm_skip_3_fusion4 = self.pooling(xm_skip_3) # (1,64,20,24,28)
        # xm_skip_3_fusion4 = self.fusion_m3_scale4(xm_skip_3_fusion4)
        xm_skip_4_fusion4 = xm_skip_4 # (1,64,20,24,28)

        xm_fusion4 = torch.cat([xm_skip_1_fusion4, xm_skip_2_fusion4], dim=1)
        xm_fusion4 = torch.cat([xm_fusion4, xm_skip_3_fusion4], dim=1)
        xm_fusion4 = torch.cat([xm_fusion4, xm_skip_4_fusion4], dim=1)

        xm_fusion4 = self.fusion_m_scale4(xm_fusion4) # (1,64,20,24,28)
        xm_fusion4_feature_1 = xm_fusion4

        xm_fusion4 = self.fusion_m_scale4_2(xm_fusion4) # (1,64,20,24,28)
        xm_fusion4_feature_2 = xm_fusion4
    

        xf_skip_1_fusion4 = self.pooling_8(xf_skip_1) # (1,32,20,24,28)
        # xf_skip_1_fusion4 = self.fusion_f1_scale4(xf_skip_1_fusion4)
        xf_skip_2_fusion4 = self.pooling_4(xf_skip_2) # (1,64,20,24,28)
        # xf_skip_2_fusion4 = self.fusion_f2_scale4(xf_skip_2_fusion4)
        xf_skip_3_fusion4 = self.pooling(xf_skip_3) # (1,64,20,24,28)
        # xf_skip_3_fusion4 = self.fusion_f3_scale4(xf_skip_3_fusion4)
        xf_skip_4_fusion4 = xf_skip_4 # (1,64,20,24,28)

        xf_fusion4 = torch.cat([xf_skip_1_fusion4, xf_skip_2_fusion4], dim=1) 
        xf_fusion4 = torch.cat([xf_fusion4, xf_skip_3_fusion4], dim=1)
        xf_fusion4 = torch.cat([xf_fusion4, xf_skip_4_fusion4], dim=1)
   
        xf_fusion4 = self.fusion_f_scale4(xf_fusion4)
        xf_fusion4_feature_1 = xf_fusion4

        xf_fusion4 = self.fusion_f_scale4_2(xf_fusion4) # (1,64,20,24,28) 
        xf_fusion4_feature_2 = xf_fusion4

        
        x_fusion4 = torch.cat([xm_fusion4, xf_fusion4], dim=1) # (1,128,20,24,28)

    
        # 第二个decoder
        x = self.decoder2(torch.cat([self.upsampling(x), x_fusion4], dim=1)) # (1,64,20,24,28)

        x_decoder2_feature = x
        # x = self.trans_3(x)

        # ----输出1/8分辨率的变形场-----
        flow_1 = self.output_block_1(x) # (1,3,20,24,28) 
        flow_1_up = nn.functional.interpolate(flow_1, scale_factor=2,mode="trilinear")*2 # (1,3,40,48,56)


        ## ------------------------------Scale3--------------------------------------------
        # 融合多尺度特征：(40,48,56)
        xm_skip_1_fusion3 = self.pooling_4(xm_skip_1) # (1,32,40,48,56)
        # xm_skip_1_fusion3 = self.fusion_m1_scale3(xm_skip_1_fusion3) 
        xm_skip_2_fusion3 = self.pooling(xm_skip_2) # (1,64,40,48,56)
        # xm_skip_2_fusion3 = self.fusion_m2_scale3(xm_skip_2_fusion3)
        xm_skip_3_fusion3 = xm_skip_3 # (1,64,40,48,56)
        # xm_skip_3_fusion3 = self.fusion_m3_scale3(xm_skip_3_fusion3)

        # xm_skip_4_fusion3 = self.fusion_m4_scale3(xm_skip_4) # (1,32,40,48,56)
        xm_skip_4_fusion3 = nn.functional.interpolate(xm_skip_4, scale_factor=2,mode="trilinear")
        

        xm_fusion3 = torch.cat([xm_skip_1_fusion3, xm_skip_2_fusion3], dim=1)
        xm_fusion3 = torch.cat([xm_fusion3, xm_skip_3_fusion3], dim=1)
        xm_fusion3 = torch.cat([xm_fusion3, xm_skip_4_fusion3], dim=1) # (1,224,40,48,56)

        xm_fusion3 = self.fusion_m_scale3(xm_fusion3) # (1,64,40,48,56)
        xm_fusion3_feature_1 = xm_fusion3

        xm_fusion3 = self.fusion_m_scale3_2(xm_fusion3) # (1,64,40,48,56)
        xm_fusion3_feature_2 = xm_fusion3

        xw_fusion3 = self.transformer2(xm_fusion3, flow_1_up)


        xf_skip_1_fusion3 = self.pooling_4(xf_skip_1) # (1,32,40,48,56)
        # xf_skip_1_fusion3 = self.fusion_f1_scale3(xf_skip_1_fusion3) 
        xf_skip_2_fusion3 = self.pooling(xf_skip_2) # (1,64,40,48,56)
        # xf_skip_2_fusion3 = self.fusion_f2_scale3(xf_skip_2_fusion3)
        xf_skip_3_fusion3 = xf_skip_3 # (1,64,40,48,56)
        # xf_skip_3_fusion3 = self.fusion_f3_scale3(xf_skip_3_fusion3)

        # xf_skip_4_fusion3 = self.fusion_f4_scale3(xf_skip_4) # (1,32,40,48,56)
        xf_skip_4_fusion3 = nn.functional.interpolate(xf_skip_4, scale_factor=2,mode="trilinear")
        

        xf_fusion3 = torch.cat([xf_skip_1_fusion3, xf_skip_2_fusion3], dim=1)
        xf_fusion3 = torch.cat([xf_fusion3, xf_skip_3_fusion3], dim=1)
        xf_fusion3 = torch.cat([xf_fusion3, xf_skip_4_fusion3], dim=1) # (1,224,40,48,56)

        xf_fusion3 = self.fusion_f_scale3(xf_fusion3) # (1,64,40,48,56)
        xf_fusion3_feature_1 = xf_fusion3

        xf_fusion3 = self.fusion_f_scale3_2(xf_fusion3) # (1,64,40,48,56)
        xf_fusion3_feature_2 = xf_fusion3

        # concat
        x_fusion3 = torch.cat([xw_fusion3, xf_fusion3], dim=1)  # (1,128,40,48,56)

        # 第三个decoder
        x = self.decoder3(torch.cat([self.upsampling(x), x_fusion3], dim=1)) # (1,64,40,48,56)
        x_decoder3_feature = x
        # x = self.trans_4(x)

        # ----输出1/4分辨率的变形场---
        delta_flow_2 = self.output_block_2(x) # (1,3,40,48,56)
        flow_2 = delta_flow_2 + flow_1_up # (1,3,40,48,56)
        flow_2_up = nn.functional.interpolate(flow_2, scale_factor=2,mode="trilinear")*2 # (1,3,80,96,112)

        ## ------------------------------Scale2--------------------------------------------
        # 融合多尺度特征：(80,96,112)
        xm_skip_1_fusion2 = self.pooling(xm_skip_1) # (1,32,80,96,112)
        # xm_skip_1_fusion2 = self.fusion_m1_scale2(xm_skip_1_fusion2) 
        xm_skip_2_fusion2 = xm_skip_2 # (1,64,80,96,112)
        # xm_skip_2_fusion2 = self.fusion_m2_scale2(xm_skip_2_fusion2)

        # xm_skip_3_fusion2 = self.fusion_m3_scale2(xm_skip_3)
        xm_skip_3_fusion2 = nn.functional.interpolate(xm_skip_3, scale_factor=2,mode="trilinear")  # (1,64,80,96,112)
        
        # xm_skip_4_fusion2 = self.fusion_m4_scale2(xm_skip_4) # (1,64,80,96,112)
        xm_skip_4_fusion2 = nn.functional.interpolate(xm_skip_4, scale_factor=4,mode="trilinear")
        

        xm_fusion2 = torch.cat([xm_skip_1_fusion2, xm_skip_2_fusion2], dim=1)
        xm_fusion2 = torch.cat([xm_fusion2, xm_skip_3_fusion2], dim=1)
        xm_fusion2 = torch.cat([xm_fusion2, xm_skip_4_fusion2], dim=1) # (1,224,80,96,112)


        xm_fusion2 = self.fusion_m_scale2(xm_fusion2) # (1,64,80,96,112)
        xm_fusion2_feature_1 = xm_fusion2

        xm_fusion2 = self.fusion_m_scale2_2(xm_fusion2) # (1,64,80,96,112)
        xm_fusion2_feature_2 = xm_fusion2


        xw_fusion2 = self.transformer3(xm_fusion2, flow_2_up)

        xf_skip_1_fusion2 = self.pooling(xf_skip_1) # (1,16,80,96,112)
        # xf_skip_1_fusion2 = self.fusion_f1_scale2(xf_skip_1_fusion2) 
        xf_skip_2_fusion2 = xf_skip_2 # (1,32,80,96,112)
        # xf_skip_2_fusion2 = self.fusion_f2_scale2(xf_skip_2_fusion2)

        # xf_skip_3_fusion2 = self.fusion_f3_scale2(xf_skip_3)
        xf_skip_3_fusion2 = nn.functional.interpolate(xf_skip_3, scale_factor=2,mode="trilinear")  # (1,32,80,96,112)
        
        # xf_skip_4_fusion2 = self.fusion_f4_scale2(xf_skip_4) # (1,32,80,96,112)
        xf_skip_4_fusion2 = nn.functional.interpolate(xf_skip_4, scale_factor=4,mode="trilinear")
        

        xf_fusion2 = torch.cat([xf_skip_1_fusion2, xf_skip_2_fusion2], dim=1)
        xf_fusion2 = torch.cat([xf_fusion2, xf_skip_3_fusion2], dim=1)
        xf_fusion2 = torch.cat([xf_fusion2, xf_skip_4_fusion2], dim=1) # (1,112,80,96,112)

        xf_fusion2 = self.fusion_f_scale2(xf_fusion2) # (1,64,80,96,112)
        xf_fusion2_feature_1 = xf_fusion2

        xf_fusion2 = self.fusion_f_scale2_2(xf_fusion2) # (1,64,80,96,112)
        xf_fusion2_feature_2 = xf_fusion2

        # concat
        x_fusion2 = torch.cat([xw_fusion2, xf_fusion2], dim=1)  # (1,32,80,96,112)


        # 第四个decoder
        x = self.decoder4(torch.cat([self.upsampling(x), x_fusion2], dim=1)) # (1,32,80,96,112)
        x_decoder4_feature = x
        # x = self.trans_5(x)
        # print(x.shape)

        # ----输出1/2分辨率的变形场-------
        delta_flow_3 = self.output_block_3(x) # (1,3,80,96,112)
        flow_3 = delta_flow_3 + flow_2_up # (1,3,80,96,112)
        flow_3_up = nn.functional.interpolate(flow_3, scale_factor=2,mode="trilinear")*2 # (1,3,160,112,224)

        # ------------------------------Scale1--------------------------------------------
        # # 对moving feature进行warp得到moved feature
        xw_skip_1 = self.transformer4(xm_skip_1, flow_3_up)
        # concat
        x_skip_1 = torch.cat([xw_skip_1, xf_skip_1], dim=1)  # (1,64,160,192,224)
        # print(x_skip_1.shape)


        # 第四个decoder
        x = self.decoder5(torch.cat([self.upsampling(x), x_skip_1], dim=1)) # (1,32,160,192,224)
        x_decoder5_feature = x

        encoder_features = [xf_skip_1,xf_skip_2,xf_skip_3,xf_skip_4,xf_skip_5,xm_skip_1,xm_skip_2,xm_skip_3,xm_skip_4,xm_skip_5]
        decoder_features = [x_decoder1_feature,x_decoder2_feature,x_decoder3_feature,x_decoder4_feature,x_decoder5_feature]
        FFM_features = [xm_fusion4_feature_1,xm_fusion4_feature_2,xf_fusion4_feature_1,xf_fusion4_feature_2,xm_fusion3_feature_1,xm_fusion3_feature_2,xf_fusion3_feature_1,xf_fusion3_feature_2,xm_fusion2_feature_1,xm_fusion2_feature_2,xf_fusion2_feature_1,xf_fusion2_feature_2]
        features = [encoder_features, decoder_features,FFM_features]
        # ----------------------输出最终的变形场------------------------------------------
        # output block
        x = self.output_block(x) # (1,16,160,192,224)
        # 生成flow场
        delta_flow_final = self.flow(x) # (1,3,160,192,224)
        flow_final = delta_flow_final + flow_3_up
            

        return flow_1,flow_2,flow_3,flow_final,delta_flow_2,delta_flow_3,delta_flow_final

#(with FFM) 直接融合 + 俩卷积 (剪枝后的网络) 
class dual_pyramid_VxmDense_FFM_normal(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    自己写的, 两个权重共享的编码器来各自提取img_concat_wavelet的特征,
    然后特征融合, 再送入同一个解码器
    在multi-vxmdense的基础上改的

    配准网络本身配准的是输入model的source和target的尺寸, 而SPT则是new_shape的尺寸
    特别需要注意的就是输入的source和target的尺寸是否能完成四次下采样, 不能的话要注意调整网络下采样的次数
    最后生成的flow可以通过self.flow里面的卷积步长stride来改变尺寸(下采样等等)
    """

    @store_config_args
    def __init__(self,
                 inshape=(192, 160, 192)):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this 
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. 
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2. 
                Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)

        # # print(new_shape)
        self.transformer1 = layers.SpatialTransformer((16,16,16))
        self.transformer2 = layers.SpatialTransformer((32,32,32))
        self.transformer3 = layers.SpatialTransformer((64,64,64))
        self.transformer4 = layers.SpatialTransformer((128,128,128))

        # self.transformer1 = layers.SpatialTransformer((24,20,24))
        # self.transformer2 = layers.SpatialTransformer((48,40,48))
        # self.transformer3 = layers.SpatialTransformer((96,80,96))
        # self.transformer4 = layers.SpatialTransformer((192, 160, 192))

        # self.transformer1 = layers.SpatialTransformer((24,20,20))
        # self.transformer2 = layers.SpatialTransformer((48,40,40))
        # self.transformer3 = layers.SpatialTransformer((96,80,80))
        # self.transformer4 = layers.SpatialTransformer((192, 160, 160))

        # self.transformer1 = layers.SpatialTransformer((20,24,28))
        # self.transformer2 = layers.SpatialTransformer((40,48,56))
        # self.transformer3 = layers.SpatialTransformer((80,96,112))
        # self.transformer4 = layers.SpatialTransformer((160,192,224))
        # self.input_model = input_model
        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = MaxPooling(2)
        self.pooling_4 =MaxPooling(4)
        self.pooling_8 =MaxPooling(8)

        self.upsampling = nn.Upsample(scale_factor=2, mode='trilinear') 


        self.encoder1_m = ConvBlock(3,1,16)  # 权重共享的编码器, 提取moving
        self.encoder2_m = ConvBlock(3,16,32)
        self.encoder3_m = ConvBlock(3,32,32)
        self.encoder4_m = ConvBlock(3,32,32)
        self.encoder5_m = ConvBlock(3,32,32)

        
        self.encoder1_f = self.encoder1_m  # 权重共享的编码器, 提取fixed
        self.encoder2_f = self.encoder2_m
        self.encoder3_f = self.encoder3_m
        self.encoder4_f = self.encoder4_m
        self.encoder5_f = self.encoder5_m

        
        self.decoder1 = ConvBlock(ndims,64,32)
        self.decoder2 = ConvBlock(ndims,96,32)
        self.decoder3 = ConvBlock(ndims,96,32)
        self.decoder4 = ConvBlock(ndims,96,16)
        self.decoder5 = ConvBlock(ndims,48,16)

        self.output_block = nn.Sequential(ConvBlock(ndims,16,16),ConvBlock(ndims,16,16))
        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(16, ndims, kernel_size=3, padding=1)

        # # init flow layer with small weights and bias
        # self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        # self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        self.output_block_0 = nn.Sequential(ConvBlock(ndims,32,16),ConvBlock(ndims,16,3))
        self.output_block_1 = nn.Sequential(ConvBlock(ndims,32,16),ConvBlock(ndims,16,3))
        self.output_block_2 = nn.Sequential(ConvBlock(ndims,32,16),ConvBlock(ndims,16,3))
        self.output_block_3 = nn.Sequential(ConvBlock(ndims,16,16),ConvBlock(ndims,16,3))

        # 整合特征
        # # ------------------------------Scale4--------------------------------------------
        # self.fusion_f1_scale4 = ConvBlock(ndims,16,16)
        # self.fusion_f2_scale4 = ConvBlock(ndims,32,32)
        # self.fusion_f3_scale4 = ConvBlock(ndims,32,32)
        # self.fusion_f4_scale4 = ConvBlock(ndims,32,32)
        self.fusion_f_scale4 = ConvBlock(ndims,112,32)
        self.fusion_f_scale4_2 = ConvBlock(ndims,32,32)

        # self.fusion_m1_scale4 = self.fusion_f1_scale4
        # self.fusion_m2_scale4 = self.fusion_f2_scale4
        # self.fusion_m3_scale4 = self.fusion_f3_scale4
        # self.fusion_m4_scale4 = self.fusion_f4_scale4
        self.fusion_m_scale4 = self.fusion_f_scale4
        self.fusion_m_scale4_2 = self.fusion_f_scale4_2

        # # ------------------------------Scale3--------------------------------------------
        # self.fusion_f1_scale3 = ConvBlock(ndims,16,16)
        # self.fusion_f2_scale3 = ConvBlock(ndims,32,32)
        # self.fusion_f3_scale3 = ConvBlock(ndims,32,32)
        # self.fusion_f4_scale3 = ConvBlock(ndims,32,32)
        self.fusion_f_scale3 = ConvBlock(ndims,112,32)
        self.fusion_f_scale3_2 = ConvBlock(ndims,32,32)

        # self.fusion_m1_scale3 = self.fusion_f1_scale3
        # self.fusion_m2_scale3 = self.fusion_f2_scale3
        # self.fusion_m3_scale3 = self.fusion_f3_scale3
        # self.fusion_m4_scale3 = self.fusion_f4_scale3
        self.fusion_m_scale3 = self.fusion_f_scale3
        self.fusion_m_scale3_2 = self.fusion_f_scale3_2

        # # ------------------------------Scale2--------------------------------------------
        # self.fusion_f1_scale2 = ConvBlock(ndims,16,16)
        # self.fusion_f2_scale2 = ConvBlock(ndims,32,32)
        # self.fusion_f3_scale2 = ConvBlock(ndims,32,32)
        # self.fusion_f4_scale2 = ConvBlock(ndims,32,32)
        self.fusion_f_scale2 = ConvBlock(ndims,112,32)
        self.fusion_f_scale2_2 = ConvBlock(ndims,32,32)

        # self.fusion_m1_scale2 = self.fusion_f1_scale2
        # self.fusion_m2_scale2 = self.fusion_f2_scale2
        # self.fusion_m3_scale2 = self.fusion_f3_scale2
        # self.fusion_m4_scale2 = self.fusion_f4_scale2
        self.fusion_m_scale2 = self.fusion_f_scale2
        self.fusion_m_scale2_2 = self.fusion_f_scale2_2

        
    def resblock_seq(self, in_channels, out_channels, bias_opt=False):
        '''
        一个resblock_seq作为一个编码器, 不需要太多,
        借鉴LapIRN
        '''
        layer = nn.Sequential(
            PreActBlock(in_channels, out_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
        )
        return layer
    
    
    
    def forward(self, source, target):
        '''
        Parameters:
            source: Source image tensor.F (moving)
            target: Target image tensor.  (fixed)
            registration: Return transformed image and flow. Default is False.
        '''
        xm = source
        xf = target
        xm_skip_1 = self.encoder1_m(xm) # (1,16,160,192,224)
        xf_skip_1 = self.encoder1_f(xf) # (1,16,160,192,224)

        # moving encoder
        #下采样到1/2,第二个encoder
        xm = self.pooling(xm_skip_1) # (1,16,80,96,112)
        xm_skip_2 = self.encoder2_m(xm) # (1,32,80,96,112)
        # 下采样到1/4,第三个encoder
        xm = self.pooling(xm_skip_2) # (1,32,40,48,56)
        xm_skip_3 = self.encoder3_m(xm) # (1,32,40,48,56)
        # 下采样到1/8,第四个encoder
        xm = self.pooling(xm_skip_3)  # (1,32,20,24,28)
        xm_skip_4 = self.encoder4_m(xm) # (1,32,20,24,28)
        # 下采样到1/16
        xm = self.pooling(xm_skip_4) # (1,32,10,12,14)
        xm = self.encoder5_m(xm) # (1,32,10,12,14)


        # fixed encoder
        #下采样到1/2,第二个encoder
        xf = self.pooling(xf_skip_1)
        xf_skip_2 = self.encoder2_f(xf)
        # 下采样到1/4,第三个encoder
        xf = self.pooling(xf_skip_2)
        xf_skip_3 = self.encoder3_f(xf)
        # 下采样到1/8,第四个encoder
        xf = self.pooling(xf_skip_3)  
        xf_skip_4 = self.encoder4_f(xf)
        # 下采样到1/16
        xf = self.pooling(xf_skip_4)
        xf = self.encoder5_f(xf) # (1,32,10,12,14)

        #-------------------------------Scale5--------------------------------
        # 第一个decoder
        x = torch.cat([xm, xf], dim=1) # (1,64,10,12,14)
        x = self.decoder1(x) # (1,32,10,12,14)


        # ------------------------------Scale4--------------------------------------------
        # 融合多尺度特征：(20,24,28)
        xm_skip_1_fusion4 = self.pooling_8(xm_skip_1) # (1,16,20,24,28)
        # xm_skip_1_fusion4 = self.fusion_m1_scale4(xm_skip_1_fusion4) 
        xm_skip_2_fusion4 = self.pooling_4(xm_skip_2) # (1,32,20,24,28)
        # xm_skip_2_fusion4 = self.fusion_m2_scale4(xm_skip_2_fusion4)
        xm_skip_3_fusion4 = self.pooling(xm_skip_3) # (1,32,20,24,28)
        # xm_skip_3_fusion4 = self.fusion_m3_scale4(xm_skip_3_fusion4)
        xm_skip_4_fusion4 = xm_skip_4 # (1,32,20,24,28)

        xm_fusion4 = torch.cat([xm_skip_1_fusion4, xm_skip_2_fusion4], dim=1)
        xm_fusion4 = torch.cat([xm_fusion4, xm_skip_3_fusion4], dim=1)
        xm_fusion4 = torch.cat([xm_fusion4, xm_skip_4_fusion4], dim=1)

        xm_fusion4 = self.fusion_m_scale4(xm_fusion4) # (1,32,20,24,28)
        xm_fusion4 = self.fusion_m_scale4_2(xm_fusion4) # (1,32,20,24,28)
    

        xf_skip_1_fusion4 = self.pooling_8(xf_skip_1) # (1,16,20,24,28)
        # xf_skip_1_fusion4 = self.fusion_f1_scale4(xf_skip_1_fusion4)
        xf_skip_2_fusion4 = self.pooling_4(xf_skip_2) # (1,32,20,24,28)
        # xf_skip_2_fusion4 = self.fusion_f2_scale4(xf_skip_2_fusion4)
        xf_skip_3_fusion4 = self.pooling(xf_skip_3) # (1,32,20,24,28)
        # xf_skip_3_fusion4 = self.fusion_f3_scale4(xf_skip_3_fusion4)
        xf_skip_4_fusion4 = xf_skip_4 # (1,32,20,24,28)

        xf_fusion4 = torch.cat([xf_skip_1_fusion4, xf_skip_2_fusion4], dim=1) 
        xf_fusion4 = torch.cat([xf_fusion4, xf_skip_3_fusion4], dim=1)
        xf_fusion4 = torch.cat([xf_fusion4, xf_skip_4_fusion4], dim=1)
   
        xf_fusion4 = self.fusion_f_scale4(xf_fusion4)
        xf_fusion4 = self.fusion_f_scale4_2(xf_fusion4) # (1,32,20,24,28) 

        
        x_fusion4 = torch.cat([xm_fusion4, xf_fusion4], dim=1) # (1,64,20,24,28)

    
        # 第二个decoder
        x = self.decoder2(torch.cat([self.upsampling(x), x_fusion4], dim=1)) # (1,32,20,24,28)
        # x = self.trans_3(x)

        # ----输出1/8分辨率的变形场-----
        flow_1 = self.output_block_1(x) # (1,3,20,24,28) 
        flow_1_up = nn.functional.interpolate(flow_1, scale_factor=2,mode="trilinear")*2 # (1,3,40,48,56)


        ## ------------------------------Scale3--------------------------------------------
        # 融合多尺度特征：(40,48,56)
        xm_skip_1_fusion3 = self.pooling_4(xm_skip_1) # (1,16,40,48,56)
        # xm_skip_1_fusion3 = self.fusion_m1_scale3(xm_skip_1_fusion3) 
        xm_skip_2_fusion3 = self.pooling(xm_skip_2) # (1,32,40,48,56)
        # xm_skip_2_fusion3 = self.fusion_m2_scale3(xm_skip_2_fusion3)
        xm_skip_3_fusion3 = xm_skip_3 # (1,32,40,48,56)
        # xm_skip_3_fusion3 = self.fusion_m3_scale3(xm_skip_3_fusion3)

        # xm_skip_4_fusion3 = self.fusion_m4_scale3(xm_skip_4) # (1,32,40,48,56)
        xm_skip_4_fusion3 = nn.functional.interpolate(xm_skip_4, scale_factor=2,mode="trilinear")
        

        xm_fusion3 = torch.cat([xm_skip_1_fusion3, xm_skip_2_fusion3], dim=1)
        xm_fusion3 = torch.cat([xm_fusion3, xm_skip_3_fusion3], dim=1)
        xm_fusion3 = torch.cat([xm_fusion3, xm_skip_4_fusion3], dim=1) # (1,112,40,48,56)

        xm_fusion3 = self.fusion_m_scale3(xm_fusion3) # (1,32,40,48,56)
        xm_fusion3 = self.fusion_m_scale3_2(xm_fusion3) # (1,32,40,48,56)

        xw_fusion3 = self.transformer2(xm_fusion3, flow_1_up)


        xf_skip_1_fusion3 = self.pooling_4(xf_skip_1) # (1,16,40,48,56)
        # xf_skip_1_fusion3 = self.fusion_f1_scale3(xf_skip_1_fusion3) 
        xf_skip_2_fusion3 = self.pooling(xf_skip_2) # (1,32,40,48,56)
        # xf_skip_2_fusion3 = self.fusion_f2_scale3(xf_skip_2_fusion3)
        xf_skip_3_fusion3 = xf_skip_3 # (1,32,40,48,56)
        # xf_skip_3_fusion3 = self.fusion_f3_scale3(xf_skip_3_fusion3)

        # xf_skip_4_fusion3 = self.fusion_f4_scale3(xf_skip_4) # (1,32,40,48,56)
        xf_skip_4_fusion3 = nn.functional.interpolate(xf_skip_4, scale_factor=2,mode="trilinear")
        

        xf_fusion3 = torch.cat([xf_skip_1_fusion3, xf_skip_2_fusion3], dim=1)
        xf_fusion3 = torch.cat([xf_fusion3, xf_skip_3_fusion3], dim=1)
        xf_fusion3 = torch.cat([xf_fusion3, xf_skip_4_fusion3], dim=1) # (1,112,40,48,56)

        xf_fusion3 = self.fusion_f_scale3(xf_fusion3) # (1,32,40,48,56)
        xf_fusion3 = self.fusion_f_scale3_2(xf_fusion3) # (1,32,40,48,56)

        # concat
        x_fusion3 = torch.cat([xw_fusion3, xf_fusion3], dim=1)  # (1,64,40,48,56)

        # 第三个decoder
        x = self.decoder3(torch.cat([self.upsampling(x), x_fusion3], dim=1)) # (1,32,40,48,56)
        # x = self.trans_4(x)

        # ----输出1/4分辨率的变形场---
        delta_flow_2 = self.output_block_2(x) # (1,3,40,48,56)
        flow_2 = delta_flow_2 + flow_1_up # (1,3,40,48,56)
        flow_2_up = nn.functional.interpolate(flow_2, scale_factor=2,mode="trilinear")*2 # (1,3,80,96,112)

        ## ------------------------------Scale2--------------------------------------------
        # 融合多尺度特征：(80,96,112)
        xm_skip_1_fusion2 = self.pooling(xm_skip_1) # (1,16,80,96,112)
        # xm_skip_1_fusion2 = self.fusion_m1_scale2(xm_skip_1_fusion2) 
        xm_skip_2_fusion2 = xm_skip_2 # (1,32,80,96,112)
        # xm_skip_2_fusion2 = self.fusion_m2_scale2(xm_skip_2_fusion2)

        # xm_skip_3_fusion2 = self.fusion_m3_scale2(xm_skip_3)
        xm_skip_3_fusion2 = nn.functional.interpolate(xm_skip_3, scale_factor=2,mode="trilinear")  # (1,32,80,96,112)
        
        # xm_skip_4_fusion2 = self.fusion_m4_scale2(xm_skip_4) # (1,32,80,96,112)
        xm_skip_4_fusion2 = nn.functional.interpolate(xm_skip_4, scale_factor=4,mode="trilinear")
        

        xm_fusion2 = torch.cat([xm_skip_1_fusion2, xm_skip_2_fusion2], dim=1)
        xm_fusion2 = torch.cat([xm_fusion2, xm_skip_3_fusion2], dim=1)
        xm_fusion2 = torch.cat([xm_fusion2, xm_skip_4_fusion2], dim=1) # (1,112,80,96,112)


        xm_fusion2 = self.fusion_m_scale2(xm_fusion2) # (1,32,80,96,112)
        xm_fusion2 = self.fusion_m_scale2_2(xm_fusion2) # (1,32,80,96,112)


        xw_fusion2 = self.transformer3(xm_fusion2, flow_2_up)

        xf_skip_1_fusion2 = self.pooling(xf_skip_1) # (1,16,80,96,112)
        # xf_skip_1_fusion2 = self.fusion_f1_scale2(xf_skip_1_fusion2) 
        xf_skip_2_fusion2 = xf_skip_2 # (1,32,80,96,112)
        # xf_skip_2_fusion2 = self.fusion_f2_scale2(xf_skip_2_fusion2)

        # xf_skip_3_fusion2 = self.fusion_f3_scale2(xf_skip_3)
        xf_skip_3_fusion2 = nn.functional.interpolate(xf_skip_3, scale_factor=2,mode="trilinear")  # (1,32,80,96,112)
        
        # xf_skip_4_fusion2 = self.fusion_f4_scale2(xf_skip_4) # (1,32,80,96,112)
        xf_skip_4_fusion2 = nn.functional.interpolate(xf_skip_4, scale_factor=4,mode="trilinear")
        

        xf_fusion2 = torch.cat([xf_skip_1_fusion2, xf_skip_2_fusion2], dim=1)
        xf_fusion2 = torch.cat([xf_fusion2, xf_skip_3_fusion2], dim=1)
        xf_fusion2 = torch.cat([xf_fusion2, xf_skip_4_fusion2], dim=1) # (1,112,80,96,112)

        xf_fusion2 = self.fusion_f_scale2(xf_fusion2) # (1,32,80,96,112)
        xf_fusion2 = self.fusion_f_scale2_2(xf_fusion2) # (1,32,80,96,112)

        # concat
        x_fusion2 = torch.cat([xw_fusion2, xf_fusion2], dim=1)  # (1,32,80,96,112)


        # 第四个decoder
        x = self.decoder4(torch.cat([self.upsampling(x), x_fusion2], dim=1)) # (1,32,80,96,112)
        # x = self.trans_5(x)
        # print(x.shape)

        # ----输出1/2分辨率的变形场-------
        delta_flow_3 = self.output_block_3(x) # (1,3,80,96,112)
        flow_3 = delta_flow_3 + flow_2_up # (1,3,80,96,112)
        flow_3_up = nn.functional.interpolate(flow_3, scale_factor=2,mode="trilinear")*2 # (1,3,160,112,224)

        # ------------------------------Scale1--------------------------------------------
        # # 对moving feature进行warp得到moved feature
        xw_skip_1 = self.transformer4(xm_skip_1, flow_3_up)
        # concat
        x_skip_1 = torch.cat([xw_skip_1, xf_skip_1], dim=1)  # (1,32,160,192,224)
        # print(x_skip_1.shape)


        # 第四个decoder
        x = self.decoder5(torch.cat([self.upsampling(x), x_skip_1], dim=1)) # (1,32,160,192,224)

        # ----------------------输出最终的变形场------------------------------------------
        # output block
        x = self.output_block(x) # (1,16,160,192,224)
        # 生成flow场
        delta_flow_final = self.flow(x) # (1,3,160,192,224)
        flow_final = delta_flow_final + flow_3_up
            

        return flow_1,flow_2,flow_3,flow_final,delta_flow_2,delta_flow_3,delta_flow_final


# #(with FFM) 直接融合 + 俩卷积 (剪枝后的网络) 
class dual_pyramid_VxmDense_FFM_normal_GDP(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    自己写的, 两个权重共享的编码器来各自提取img_concat_wavelet的特征,
    然后特征融合, 再送入同一个解码器
    在multi-vxmdense的基础上改的

    配准网络本身配准的是输入model的source和target的尺寸, 而SPT则是new_shape的尺寸
    特别需要注意的就是输入的source和target的尺寸是否能完成四次下采样, 不能的话要注意调整网络下采样的次数
    最后生成的flow可以通过self.flow里面的卷积步长stride来改变尺寸(下采样等等)
    """

    @store_config_args
    def __init__(self,
                 inshape=(160,192,224)):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this 
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. 
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2. 
                Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)

        # # # print(new_shape)
        # self.transformer1 = layers.SpatialTransformer((16,16,16))
        # self.transformer2 = layers.SpatialTransformer((32,32,32))
        # self.transformer3 = layers.SpatialTransformer((64,64,64))
        # self.transformer4 = layers.SpatialTransformer((128,128,128))

        # self.transformer1 = layers.SpatialTransformer((24,20,20))
        # self.transformer2 = layers.SpatialTransformer((48,40,40))
        # self.transformer3 = layers.SpatialTransformer((96,80,80))
        # self.transformer4 = layers.SpatialTransformer((192,160,160))

        # self.transformer1 = layers.SpatialTransformer((20,24,28))
        # self.transformer2 = layers.SpatialTransformer((40,48,56))
        # self.transformer3 = layers.SpatialTransformer((80,96,112))
        # self.transformer4 = layers.SpatialTransformer((160,192,224))

        self.transformer1 = layers.SpatialTransformer((24,20,24))
        self.transformer2 = layers.SpatialTransformer((48,40,48))
        self.transformer3 = layers.SpatialTransformer((96,80,96))
        self.transformer4 = layers.SpatialTransformer((192,160,192))
        # self.input_model = input_model
        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = MaxPooling(2)
        self.pooling_4 =MaxPooling(4)
        self.pooling_8 =MaxPooling(8)

        self.upsampling = nn.Upsample(scale_factor=2, mode='trilinear') 


        self.encoder1_m = ConvBlock(3,1,16)  # 权重共享的编码器, 提取moving
        self.encoder2_m = ConvBlock(3,16,32)
        self.encoder3_m = ConvBlock(3,32,32)
        self.encoder4_m = ConvBlock(3,32,32)
        self.encoder5_m = ConvBlock(3,32,32)

        
        self.encoder1_f = self.encoder1_m  # 权重共享的编码器, 提取fixed
        self.encoder2_f = self.encoder2_m
        self.encoder3_f = self.encoder3_m
        self.encoder4_f = self.encoder4_m
        self.encoder5_f = self.encoder5_m

        
        self.decoder1 = ConvBlock(ndims,64,32)
        self.decoder2 = ConvBlock(ndims,96,32)
        self.decoder3 = ConvBlock(ndims,96,32)
        self.decoder4 = ConvBlock(ndims,96,16)
        self.decoder5 = ConvBlock(ndims,48,16)

        self.output_block = nn.Sequential(ConvBlock(ndims,16,16),ConvBlock(ndims,16,16))
        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(16, ndims, kernel_size=3, padding=1)

        # # init flow layer with small weights and bias
        # self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        # self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        self.output_block_0 = nn.Sequential(ConvBlock(ndims,32,16),ConvBlock(ndims,16,3))
        self.output_block_1 = nn.Sequential(ConvBlock(ndims,32,16),ConvBlock(ndims,16,3))
        self.output_block_2 = nn.Sequential(ConvBlock(ndims,32,16),ConvBlock(ndims,16,3))
        self.output_block_3 = nn.Sequential(ConvBlock(ndims,16,16),ConvBlock(ndims,16,3))

        # 整合特征
        # # ------------------------------Scale4--------------------------------------------
        # self.fusion_f1_scale4 = ConvBlock(ndims,16,16)
        # self.fusion_f2_scale4 = ConvBlock(ndims,32,32)
        # self.fusion_f3_scale4 = ConvBlock(ndims,32,32)
        # self.fusion_f4_scale4 = ConvBlock(ndims,32,32)
        self.fusion_f_scale4 = ConvBlock(ndims,112,32)
        self.fusion_f_scale4_2 = ConvBlock(ndims,32,32)

        # self.fusion_m1_scale4 = self.fusion_f1_scale4
        # self.fusion_m2_scale4 = self.fusion_f2_scale4
        # self.fusion_m3_scale4 = self.fusion_f3_scale4
        # self.fusion_m4_scale4 = self.fusion_f4_scale4
        self.fusion_m_scale4 = self.fusion_f_scale4
        self.fusion_m_scale4_2 = self.fusion_f_scale4_2

        # # ------------------------------Scale3--------------------------------------------
        # self.fusion_f1_scale3 = ConvBlock(ndims,16,16)
        # self.fusion_f2_scale3 = ConvBlock(ndims,32,32)
        # self.fusion_f3_scale3 = ConvBlock(ndims,32,32)
        # self.fusion_f4_scale3 = ConvBlock(ndims,32,32)
        self.fusion_f_scale3 = ConvBlock(ndims,112,32)
        self.fusion_f_scale3_2 = ConvBlock(ndims,32,32)

        # self.fusion_m1_scale3 = self.fusion_f1_scale3
        # self.fusion_m2_scale3 = self.fusion_f2_scale3
        # self.fusion_m3_scale3 = self.fusion_f3_scale3
        # self.fusion_m4_scale3 = self.fusion_f4_scale3
        self.fusion_m_scale3 = self.fusion_f_scale3
        self.fusion_m_scale3_2 = self.fusion_f_scale3_2

        # # ------------------------------Scale2--------------------------------------------
        # self.fusion_f1_scale2 = ConvBlock(ndims,16,16)
        # self.fusion_f2_scale2 = ConvBlock(ndims,32,32)
        # self.fusion_f3_scale2 = ConvBlock(ndims,32,32)
        # self.fusion_f4_scale2 = ConvBlock(ndims,32,32)
        self.fusion_f_scale2 = ConvBlock(ndims,112,32)
        self.fusion_f_scale2_2 = ConvBlock(ndims,32,32)

        # self.fusion_m1_scale2 = self.fusion_f1_scale2
        # self.fusion_m2_scale2 = self.fusion_f2_scale2
        # self.fusion_m3_scale2 = self.fusion_f3_scale2
        # self.fusion_m4_scale2 = self.fusion_f4_scale2
        self.fusion_m_scale2 = self.fusion_f_scale2
        self.fusion_m_scale2_2 = self.fusion_f_scale2_2

        # 定义筛选特征的向量
        self.alpha_1 = nn.Parameter(torch.ones(1, 16))
        self.alpha_2 = nn.Parameter(torch.ones(1, 32))
        self.alpha_3 = nn.Parameter(torch.ones(1, 32))
        self.alpha_4 = nn.Parameter(torch.ones(1, 32))
        self.alpha_5 = nn.Parameter(torch.ones(1, 32))
        self.alpha_6_1 = nn.Parameter(torch.ones(1, 32))
        self.alpha_6_2 = nn.Parameter(torch.ones(1, 32))
        self.alpha_7_1 = nn.Parameter(torch.ones(1, 32))
        self.alpha_7_2 = nn.Parameter(torch.ones(1, 32))
        self.alpha_8_1 = nn.Parameter(torch.ones(1, 32))
        self.alpha_8_2 = nn.Parameter(torch.ones(1, 32))

        # 定义门函数
        self.gate_activation = GatingFunction()

        # 更新decay的系数
    def set_decay(self, new_decay_value):
        self.gate_activation.decay = new_decay_value

    
    
    
    def forward(self, source, target):
        '''
        Parameters:
            source: Source image tensor.F (moving)
            target: Target image tensor.  (fixed)
            registration: Return transformed image and flow. Default is False.
        '''
        ori_alpha_1 = self.alpha_1
        ori_alpha_2 = self.alpha_2
        ori_alpha_3 = self.alpha_3
        ori_alpha_4 = self.alpha_4
        ori_alpha_5 = self.alpha_5
        ori_alpha_6_1 = self.alpha_6_1
        ori_alpha_6_2 = self.alpha_6_2
        ori_alpha_7_1 = self.alpha_7_1
        ori_alpha_7_2 = self.alpha_7_2
        ori_alpha_8_1 = self.alpha_8_1
        ori_alpha_8_2 = self.alpha_8_2
        
        # 输入门函数
        alpha_1 = self.gate_activation(ori_alpha_1)
        alpha_2 = self.gate_activation(ori_alpha_2)
        alpha_3 = self.gate_activation(ori_alpha_3)
        alpha_4 = self.gate_activation(ori_alpha_4)
        alpha_5 = self.gate_activation(ori_alpha_5)
        alpha_6_1 = self.gate_activation(ori_alpha_6_1)
        alpha_6_2 = self.gate_activation(ori_alpha_6_2)
        alpha_7_1 = self.gate_activation(ori_alpha_7_1)
        alpha_7_2 = self.gate_activation(ori_alpha_7_2)
        alpha_8_1 = self.gate_activation(ori_alpha_8_1)
        alpha_8_2 = self.gate_activation(ori_alpha_8_2)
        ori_alpha = [ori_alpha_1,ori_alpha_2,ori_alpha_3,ori_alpha_4,ori_alpha_5,ori_alpha_6_1,ori_alpha_6_2,ori_alpha_7_1,ori_alpha_7_2,ori_alpha_8_1,ori_alpha_8_2]
        gate_alpha = [alpha_1,alpha_2,alpha_3,alpha_4,alpha_5,alpha_6_1,alpha_6_2,alpha_7_1,alpha_7_2,alpha_8_1,alpha_8_2]

        # 维度扩展
        alpha_1 = alpha_1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  
        alpha_2 = alpha_2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        alpha_3 = alpha_3.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        alpha_4 = alpha_4.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        alpha_5 = alpha_5.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        alpha_6_1 = alpha_6_1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) 
        alpha_6_2 = alpha_6_2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        alpha_7_1 = alpha_7_1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) 
        alpha_7_2 = alpha_7_2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        alpha_8_1 = alpha_8_1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) 
        alpha_8_2 = alpha_8_2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  


        xm = source
        xf = target
        xm_skip_1 = self.encoder1_m(xm) # (1,16,160,192,224)
        xm_skip_1 = xm_skip_1 * alpha_1
        xf_skip_1 = self.encoder1_f(xf) # (1,16,160,192,224)
        xf_skip_1 = xf_skip_1 * alpha_1

        # moving encoder
        #下采样到1/2,第二个encoder
        xm = self.pooling(xm_skip_1) # (1,16,80,96,112)
        xm_skip_2 = self.encoder2_m(xm) # (1,32,80,96,112)
        xm_skip_2 = xm_skip_2 * alpha_2
        # 下采样到1/4,第三个encoder
        xm = self.pooling(xm_skip_2) # (1,32,40,48,56)
        xm_skip_3 = self.encoder3_m(xm) # (1,32,40,48,56)
        xm_skip_3 = xm_skip_3 * alpha_3
        # 下采样到1/8,第四个encoder
        xm = self.pooling(xm_skip_3)  # (1,32,20,24,28)
        xm_skip_4 = self.encoder4_m(xm) # (1,32,20,24,28)
        xm_skip_4 = xm_skip_4 * alpha_4
        # 下采样到1/16
        xm = self.pooling(xm_skip_4) # (1,32,10,12,14)
        xm_skip_5 = self.encoder5_m(xm) # (1,32,10,12,14)
        xm_skip_5 = xm_skip_5 * alpha_5

        # fixed encoder
        #下采样到1/2,第二个encoder
        xf = self.pooling(xf_skip_1)
        xf_skip_2 = self.encoder2_f(xf)
        xf_skip_2 = xf_skip_2 * alpha_2
        # 下采样到1/4,第三个encoder
        xf = self.pooling(xf_skip_2)
        xf_skip_3 = self.encoder3_f(xf)
        xf_skip_3 = xf_skip_3 * alpha_3
        # 下采样到1/8,第四个encoder
        xf = self.pooling(xf_skip_3)  
        xf_skip_4 = self.encoder4_f(xf)
        xf_skip_4 = xf_skip_4 * alpha_4
        # 下采样到1/16
        xf = self.pooling(xf_skip_4)
        xf_skip_5 = self.encoder5_f(xf) # (1,32,10,12,14)
        xf_skip_5 = xf_skip_5 * alpha_5

        #-------------------------------Scale5--------------------------------
        # 第一个decoder
        x = torch.cat([xm_skip_5, xf_skip_5], dim=1) # (1,64,10,12,14)
        x = self.decoder1(x) # (1,32,10,12,14)


        # ------------------------------Scale4--------------------------------------------
        # 融合多尺度特征：(20,24,28)
        xm_skip_1_fusion4 = self.pooling_8(xm_skip_1) # (1,16,20,24,28)
        # xm_skip_1_fusion4 = self.fusion_m1_scale4(xm_skip_1_fusion4) 
        xm_skip_2_fusion4 = self.pooling_4(xm_skip_2) # (1,32,20,24,28)
        # xm_skip_2_fusion4 = self.fusion_m2_scale4(xm_skip_2_fusion4)
        xm_skip_3_fusion4 = self.pooling(xm_skip_3) # (1,32,20,24,28)
        # xm_skip_3_fusion4 = self.fusion_m3_scale4(xm_skip_3_fusion4)
        xm_skip_4_fusion4 = xm_skip_4 # (1,32,20,24,28)

        xm_fusion4 = torch.cat([xm_skip_1_fusion4, xm_skip_2_fusion4], dim=1)
        xm_fusion4 = torch.cat([xm_fusion4, xm_skip_3_fusion4], dim=1)
        xm_fusion4 = torch.cat([xm_fusion4, xm_skip_4_fusion4], dim=1)

        xm_fusion4 = self.fusion_m_scale4(xm_fusion4) # (1,32,20,24,28)
        xm_fusion4 = xm_fusion4 * alpha_6_1
        xm_fusion4 = self.fusion_m_scale4_2(xm_fusion4) # (1,32,20,24,28)
        xm_fusion4 = xm_fusion4 * alpha_6_2
    

        xf_skip_1_fusion4 = self.pooling_8(xf_skip_1) # (1,16,20,24,28)
        # xf_skip_1_fusion4 = self.fusion_f1_scale4(xf_skip_1_fusion4)
        xf_skip_2_fusion4 = self.pooling_4(xf_skip_2) # (1,32,20,24,28)
        # xf_skip_2_fusion4 = self.fusion_f2_scale4(xf_skip_2_fusion4)
        xf_skip_3_fusion4 = self.pooling(xf_skip_3) # (1,32,20,24,28)
        # xf_skip_3_fusion4 = self.fusion_f3_scale4(xf_skip_3_fusion4)
        xf_skip_4_fusion4 = xf_skip_4 # (1,32,20,24,28)

        xf_fusion4 = torch.cat([xf_skip_1_fusion4, xf_skip_2_fusion4], dim=1) 
        xf_fusion4 = torch.cat([xf_fusion4, xf_skip_3_fusion4], dim=1)
        xf_fusion4 = torch.cat([xf_fusion4, xf_skip_4_fusion4], dim=1)
   
        xf_fusion4 = self.fusion_f_scale4(xf_fusion4)
        xf_fusion4 = xf_fusion4 * alpha_6_1
        xf_fusion4 = self.fusion_f_scale4_2(xf_fusion4) # (1,32,20,24,28) 
        xf_fusion4 = xf_fusion4 * alpha_6_2

        
        x_fusion4 = torch.cat([xm_fusion4, xf_fusion4], dim=1) # (1,64,20,24,28)

    
        # 第二个decoder
        x = self.decoder2(torch.cat([self.upsampling(x), x_fusion4], dim=1)) # (1,32,20,24,28)
        # x = self.trans_3(x)

        # ----输出1/8分辨率的变形场-----
        flow_1 = self.output_block_1(x) # (1,3,20,24,28) 
        flow_1_up = nn.functional.interpolate(flow_1, scale_factor=2,mode="trilinear")*2 # (1,3,40,48,56)


        ## ------------------------------Scale3--------------------------------------------
        # 融合多尺度特征：(40,48,56)
        xm_skip_1_fusion3 = self.pooling_4(xm_skip_1) # (1,16,40,48,56)
        # xm_skip_1_fusion3 = self.fusion_m1_scale3(xm_skip_1_fusion3) 
        xm_skip_2_fusion3 = self.pooling(xm_skip_2) # (1,32,40,48,56)
        # xm_skip_2_fusion3 = self.fusion_m2_scale3(xm_skip_2_fusion3)
        xm_skip_3_fusion3 = xm_skip_3 # (1,32,40,48,56)
        # xm_skip_3_fusion3 = self.fusion_m3_scale3(xm_skip_3_fusion3)

        # xm_skip_4_fusion3 = self.fusion_m4_scale3(xm_skip_4) # (1,32,40,48,56)
        xm_skip_4_fusion3 = nn.functional.interpolate(xm_skip_4, scale_factor=2,mode="trilinear")
        

        xm_fusion3 = torch.cat([xm_skip_1_fusion3, xm_skip_2_fusion3], dim=1)
        xm_fusion3 = torch.cat([xm_fusion3, xm_skip_3_fusion3], dim=1)
        xm_fusion3 = torch.cat([xm_fusion3, xm_skip_4_fusion3], dim=1) # (1,112,40,48,56)

        xm_fusion3 = self.fusion_m_scale3(xm_fusion3) # (1,32,40,48,56)
        xm_fusion3 = xm_fusion3 * alpha_7_1
        xm_fusion3 = self.fusion_m_scale3_2(xm_fusion3) # (1,32,40,48,56)
        xm_fusion3 = xm_fusion3 * alpha_7_2

        xw_fusion3 = self.transformer2(xm_fusion3, flow_1_up)


        xf_skip_1_fusion3 = self.pooling_4(xf_skip_1) # (1,16,40,48,56)
        # xf_skip_1_fusion3 = self.fusion_f1_scale3(xf_skip_1_fusion3) 
        xf_skip_2_fusion3 = self.pooling(xf_skip_2) # (1,32,40,48,56)
        # xf_skip_2_fusion3 = self.fusion_f2_scale3(xf_skip_2_fusion3)
        xf_skip_3_fusion3 = xf_skip_3 # (1,32,40,48,56)
        # xf_skip_3_fusion3 = self.fusion_f3_scale3(xf_skip_3_fusion3)

        # xf_skip_4_fusion3 = self.fusion_f4_scale3(xf_skip_4) # (1,32,40,48,56)
        xf_skip_4_fusion3 = nn.functional.interpolate(xf_skip_4, scale_factor=2,mode="trilinear")
        

        xf_fusion3 = torch.cat([xf_skip_1_fusion3, xf_skip_2_fusion3], dim=1)
        xf_fusion3 = torch.cat([xf_fusion3, xf_skip_3_fusion3], dim=1)
        xf_fusion3 = torch.cat([xf_fusion3, xf_skip_4_fusion3], dim=1) # (1,112,40,48,56)

        xf_fusion3 = self.fusion_f_scale3(xf_fusion3) # (1,32,40,48,56)
        xf_fusion3 = xf_fusion3 * alpha_7_1
        xf_fusion3 = self.fusion_f_scale3_2(xf_fusion3) # (1,32,40,48,56)
        xf_fusion3 = xf_fusion3 * alpha_7_2

        # concat
        x_fusion3 = torch.cat([xw_fusion3, xf_fusion3], dim=1)  # (1,64,40,48,56)

        # 第三个decoder
        x = self.decoder3(torch.cat([self.upsampling(x), x_fusion3], dim=1)) # (1,32,40,48,56)
        # x = self.trans_4(x)

        # ----输出1/4分辨率的变形场---
        delta_flow_2 = self.output_block_2(x) # (1,3,40,48,56)
        flow_2 = delta_flow_2 + flow_1_up # (1,3,40,48,56)
        flow_2_up = nn.functional.interpolate(flow_2, scale_factor=2,mode="trilinear")*2 # (1,3,80,96,112)

        ## ------------------------------Scale2--------------------------------------------
        # 融合多尺度特征：(80,96,112)
        xm_skip_1_fusion2 = self.pooling(xm_skip_1) # (1,16,80,96,112)
        # xm_skip_1_fusion2 = self.fusion_m1_scale2(xm_skip_1_fusion2) 
        xm_skip_2_fusion2 = xm_skip_2 # (1,32,80,96,112)
        # xm_skip_2_fusion2 = self.fusion_m2_scale2(xm_skip_2_fusion2)

        # xm_skip_3_fusion2 = self.fusion_m3_scale2(xm_skip_3)
        xm_skip_3_fusion2 = nn.functional.interpolate(xm_skip_3, scale_factor=2,mode="trilinear")  # (1,32,80,96,112)
        
        # xm_skip_4_fusion2 = self.fusion_m4_scale2(xm_skip_4) # (1,32,80,96,112)
        xm_skip_4_fusion2 = nn.functional.interpolate(xm_skip_4, scale_factor=4,mode="trilinear")
        

        xm_fusion2 = torch.cat([xm_skip_1_fusion2, xm_skip_2_fusion2], dim=1)
        xm_fusion2 = torch.cat([xm_fusion2, xm_skip_3_fusion2], dim=1)
        xm_fusion2 = torch.cat([xm_fusion2, xm_skip_4_fusion2], dim=1) # (1,112,80,96,112)


        xm_fusion2 = self.fusion_m_scale2(xm_fusion2) # (1,32,80,96,112)
        xm_fusion2 = xm_fusion2 * alpha_8_1
        xm_fusion2 = self.fusion_m_scale2_2(xm_fusion2) # (1,32,80,96,112)
        xm_fusion2 = xm_fusion2 * alpha_8_2


        xw_fusion2 = self.transformer3(xm_fusion2, flow_2_up)

        xf_skip_1_fusion2 = self.pooling(xf_skip_1) # (1,16,80,96,112)
        # xf_skip_1_fusion2 = self.fusion_f1_scale2(xf_skip_1_fusion2) 
        xf_skip_2_fusion2 = xf_skip_2 # (1,32,80,96,112)
        # xf_skip_2_fusion2 = self.fusion_f2_scale2(xf_skip_2_fusion2)

        # xf_skip_3_fusion2 = self.fusion_f3_scale2(xf_skip_3)
        xf_skip_3_fusion2 = nn.functional.interpolate(xf_skip_3, scale_factor=2,mode="trilinear")  # (1,32,80,96,112)
        
        # xf_skip_4_fusion2 = self.fusion_f4_scale2(xf_skip_4) # (1,32,80,96,112)
        xf_skip_4_fusion2 = nn.functional.interpolate(xf_skip_4, scale_factor=4,mode="trilinear")
        

        xf_fusion2 = torch.cat([xf_skip_1_fusion2, xf_skip_2_fusion2], dim=1)
        xf_fusion2 = torch.cat([xf_fusion2, xf_skip_3_fusion2], dim=1)
        xf_fusion2 = torch.cat([xf_fusion2, xf_skip_4_fusion2], dim=1) # (1,112,80,96,112)

        xf_fusion2 = self.fusion_f_scale2(xf_fusion2) # (1,32,80,96,112)
        xf_fusion2 = xf_fusion2 * alpha_8_1
        xf_fusion2 = self.fusion_f_scale2_2(xf_fusion2) # (1,32,80,96,112)
        xf_fusion2 = xf_fusion2 * alpha_8_2

        # concat
        x_fusion2 = torch.cat([xw_fusion2, xf_fusion2], dim=1)  # (1,32,80,96,112)


        # 第四个decoder
        x = self.decoder4(torch.cat([self.upsampling(x), x_fusion2], dim=1)) # (1,32,80,96,112)
        # x = self.trans_5(x)
        # print(x.shape)

        # ----输出1/2分辨率的变形场-------
        delta_flow_3 = self.output_block_3(x) # (1,3,80,96,112)
        flow_3 = delta_flow_3 + flow_2_up # (1,3,80,96,112)
        flow_3_up = nn.functional.interpolate(flow_3, scale_factor=2,mode="trilinear")*2 # (1,3,160,112,224)

        # ------------------------------Scale1--------------------------------------------
        # # 对moving feature进行warp得到moved feature
        xw_skip_1 = self.transformer4(xm_skip_1, flow_3_up)
        # concat
        x_skip_1 = torch.cat([xw_skip_1, xf_skip_1], dim=1)  # (1,32,160,192,224)
        # print(x_skip_1.shape)


        # 第四个decoder
        x = self.decoder5(torch.cat([self.upsampling(x), x_skip_1], dim=1)) # (1,32,160,192,224)

        # ----------------------输出最终的变形场------------------------------------------
        # output block
        x = self.output_block(x) # (1,16,160,192,224)
        # 生成flow场
        delta_flow_final = self.flow(x) # (1,3,160,192,224)
        flow_final = delta_flow_final + flow_3_up
            

        return flow_1,flow_2,flow_3,flow_final,delta_flow_2,delta_flow_3,delta_flow_final,ori_alpha,gate_alpha

#(with FFM) 直接融合 + 俩卷积 (剪枝后的网络) encoder和FFM 都×3
class dual_pyramid_VxmDense_FFM_huge_GDP(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    自己写的, 两个权重共享的编码器来各自提取img_concat_wavelet的特征,
    然后特征融合, 再送入同一个解码器
    在multi-vxmdense的基础上改的

    配准网络本身配准的是输入model的source和target的尺寸, 而SPT则是new_shape的尺寸
    特别需要注意的就是输入的source和target的尺寸是否能完成四次下采样, 不能的话要注意调整网络下采样的次数
    最后生成的flow可以通过self.flow里面的卷积步长stride来改变尺寸(下采样等等)
    """

    @store_config_args
    def __init__(self,
                 inshape=(160,192,224)):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this 
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. 
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2. 
                Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)

        # # print(new_shape)
        # self.transformer1 = layers.SpatialTransformer((20,24,28))
        # self.transformer2 = layers.SpatialTransformer((40,48,56))
        # self.transformer3 = layers.SpatialTransformer((80,96,112))
        # self.transformer4 = layers.SpatialTransformer((160,192,224))

        # self.transformer1 = layers.SpatialTransformer((16,16,16))
        # self.transformer2 = layers.SpatialTransformer((32,32,32))
        # self.transformer3 = layers.SpatialTransformer((64,64,64))
        # self.transformer4 = layers.SpatialTransformer((128,128,128))

        self.transformer1 = layers.SpatialTransformer((24,20,24))
        self.transformer2 = layers.SpatialTransformer((48,40,48))
        self.transformer3 = layers.SpatialTransformer((96,80,96))
        self.transformer4 = layers.SpatialTransformer((192,160,192))

        # self.transformer1 = layers.SpatialTransformer((24,20,20))
        # self.transformer2 = layers.SpatialTransformer((48,40,40))
        # self.transformer3 = layers.SpatialTransformer((96,80,80))
        # self.transformer4 = layers.SpatialTransformer((192,160,160))

        # self.input_model = input_model
        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = MaxPooling(2)
        self.pooling_4 =MaxPooling(4)
        self.pooling_8 =MaxPooling(8)

        self.upsampling = nn.Upsample(scale_factor=2, mode='trilinear') 


        self.encoder1_m = ConvBlock(3,1,48)  # 权重共享的编码器, 提取moving
        self.encoder2_m = ConvBlock(3,48,96)
        self.encoder3_m = ConvBlock(3,96,96)
        self.encoder4_m = ConvBlock(3,96,96)
        self.encoder5_m = ConvBlock(3,96,96)

        
        self.encoder1_f = self.encoder1_m  # 权重共享的编码器, 提取fixed
        self.encoder2_f = self.encoder2_m
        self.encoder3_f = self.encoder3_m
        self.encoder4_f = self.encoder4_m
        self.encoder5_f = self.encoder5_m

        
        self.decoder1 = ConvBlock(ndims,192,32)
        self.decoder2 = ConvBlock(ndims,224,32)
        self.decoder3 = ConvBlock(ndims,224,32)
        self.decoder4 = ConvBlock(ndims,224,16)
        self.decoder5 = ConvBlock(ndims,112,16)

        self.output_block = nn.Sequential(ConvBlock(ndims,16,16),ConvBlock(ndims,16,16))
        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(16, ndims, kernel_size=3, padding=1)

        # # init flow layer with small weights and bias
        # self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        # self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        self.output_block_0 = nn.Sequential(ConvBlock(ndims,32,16),ConvBlock(ndims,16,3))
        self.output_block_1 = nn.Sequential(ConvBlock(ndims,32,16),ConvBlock(ndims,16,3))
        self.output_block_2 = nn.Sequential(ConvBlock(ndims,32,16),ConvBlock(ndims,16,3))
        self.output_block_3 = nn.Sequential(ConvBlock(ndims,16,16),ConvBlock(ndims,16,3))

        # 整合特征
        # # ------------------------------Scale4--------------------------------------------
        self.fusion_f_scale4 = ConvBlock(ndims,336,96)
        self.fusion_f_scale4_2 = ConvBlock(ndims,96,96)


        self.fusion_m_scale4 = self.fusion_f_scale4
        self.fusion_m_scale4_2 = self.fusion_f_scale4_2

        # # ------------------------------Scale3--------------------------------------------
        self.fusion_f_scale3 = ConvBlock(ndims,336,96)
        self.fusion_f_scale3_2 = ConvBlock(ndims,96,96)


        self.fusion_m_scale3 = self.fusion_f_scale3
        self.fusion_m_scale3_2 = self.fusion_f_scale3_2

        # # ------------------------------Scale2--------------------------------------------
        self.fusion_f_scale2 = ConvBlock(ndims,336,96)
        self.fusion_f_scale2_2 = ConvBlock(ndims,96,96)


        self.fusion_m_scale2 = self.fusion_f_scale2
        self.fusion_m_scale2_2 = self.fusion_f_scale2_2

        # 定义筛选特征的向量
        self.alpha_1 = nn.Parameter(torch.ones(1, 48))
        self.alpha_2 = nn.Parameter(torch.ones(1, 96))
        self.alpha_3 = nn.Parameter(torch.ones(1, 96))
        self.alpha_4 = nn.Parameter(torch.ones(1, 96))
        self.alpha_5 = nn.Parameter(torch.ones(1, 96))
        self.alpha_6_1 = nn.Parameter(torch.ones(1, 96))
        self.alpha_6_2 = nn.Parameter(torch.ones(1, 96))
        self.alpha_7_1 = nn.Parameter(torch.ones(1, 96))
        self.alpha_7_2 = nn.Parameter(torch.ones(1, 96))
        self.alpha_8_1 = nn.Parameter(torch.ones(1, 96))
        self.alpha_8_2 = nn.Parameter(torch.ones(1, 96))

        # 定义门函数
        self.gate_activation = GatingFunction()

        # 更新decay的系数
    def set_decay(self, new_decay_value):
        self.gate_activation.decay = new_decay_value
    
    def forward(self, source, target):
        '''
        Parameters:
            source: Source image tensor.F (moving)
            target: Target image tensor.  (fixed)
            registration: Return transformed image and flow. Default is False.
        '''

        ori_alpha_1 = self.alpha_1
        ori_alpha_2 = self.alpha_2
        ori_alpha_3 = self.alpha_3
        ori_alpha_4 = self.alpha_4
        ori_alpha_5 = self.alpha_5
        ori_alpha_6_1 = self.alpha_6_1
        ori_alpha_6_2 = self.alpha_6_2
        ori_alpha_7_1 = self.alpha_7_1
        ori_alpha_7_2 = self.alpha_7_2
        ori_alpha_8_1 = self.alpha_8_1
        ori_alpha_8_2 = self.alpha_8_2
        
        # 输入门函数
        alpha_1 = self.gate_activation(ori_alpha_1)
        alpha_2 = self.gate_activation(ori_alpha_2)
        alpha_3 = self.gate_activation(ori_alpha_3)
        alpha_4 = self.gate_activation(ori_alpha_4)
        alpha_5 = self.gate_activation(ori_alpha_5)
        alpha_6_1 = self.gate_activation(ori_alpha_6_1)
        alpha_6_2 = self.gate_activation(ori_alpha_6_2)
        alpha_7_1 = self.gate_activation(ori_alpha_7_1)
        alpha_7_2 = self.gate_activation(ori_alpha_7_2)
        alpha_8_1 = self.gate_activation(ori_alpha_8_1)
        alpha_8_2 = self.gate_activation(ori_alpha_8_2)
        ori_alpha = [ori_alpha_1,ori_alpha_2,ori_alpha_3,ori_alpha_4,ori_alpha_5,ori_alpha_6_1,ori_alpha_6_2,ori_alpha_7_1,ori_alpha_7_2,ori_alpha_8_1,ori_alpha_8_2]
        gate_alpha = [alpha_1,alpha_2,alpha_3,alpha_4,alpha_5,alpha_6_1,alpha_6_2,alpha_7_1,alpha_7_2,alpha_8_1,alpha_8_2]

        # 维度扩展
        alpha_1 = alpha_1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  
        alpha_2 = alpha_2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        alpha_3 = alpha_3.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        alpha_4 = alpha_4.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        alpha_5 = alpha_5.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        alpha_6_1 = alpha_6_1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) 
        alpha_6_2 = alpha_6_2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        alpha_7_1 = alpha_7_1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) 
        alpha_7_2 = alpha_7_2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        alpha_8_1 = alpha_8_1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) 
        alpha_8_2 = alpha_8_2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)   
        
        xm = source
        xf = target
        xm_skip_1 = self.encoder1_m(xm) # (1,48,160,192,224)
        xm_skip_1 = xm_skip_1 * alpha_1
        xf_skip_1 = self.encoder1_f(xf) # (1,48,160,192,224)
        xf_skip_1 = xf_skip_1 * alpha_1

        # moving encoder
        #下采样到1/2,第二个encoder
        xm = self.pooling(xm_skip_1) # (1,16,80,96,112)
        xm_skip_2 = self.encoder2_m(xm) # (1,32,80,96,112)
        xm_skip_2 = xm_skip_2 * alpha_2
        # 下采样到1/4,第三个encoder
        xm = self.pooling(xm_skip_2) # (1,32,40,48,56)
        xm_skip_3 = self.encoder3_m(xm) # (1,32,40,48,56)
        xm_skip_3 = xm_skip_3 * alpha_3
        # 下采样到1/8,第四个encoder
        xm = self.pooling(xm_skip_3)  # (1,32,20,24,28)
        xm_skip_4 = self.encoder4_m(xm) # (1,32,20,24,28)
        xm_skip_4 = xm_skip_4 * alpha_4
        # 下采样到1/16
        xm = self.pooling(xm_skip_4) # (1,32,10,12,14)
        xm_skip_5 = self.encoder5_m(xm) # (1,32,10,12,14)
        xm_skip_5 = xm_skip_5 * alpha_5


        # fixed encoder
        #下采样到1/2,第二个encoder
        xf = self.pooling(xf_skip_1)
        xf_skip_2 = self.encoder2_f(xf)
        xf_skip_2 = xf_skip_2 * alpha_2
        # 下采样到1/4,第三个encoder
        xf = self.pooling(xf_skip_2)
        xf_skip_3 = self.encoder3_f(xf)
        xf_skip_3 = xf_skip_3 * alpha_3
        # 下采样到1/8,第四个encoder
        xf = self.pooling(xf_skip_3)  
        xf_skip_4 = self.encoder4_f(xf)
        xf_skip_4 = xf_skip_4 * alpha_4
        # 下采样到1/16
        xf = self.pooling(xf_skip_4)
        xf_skip_5 = self.encoder5_f(xf) # (1,32,10,12,14)
        xf_skip_5 = xf_skip_5 * alpha_5

        #-------------------------------Scale5--------------------------------
        # 第一个decoder
        x = torch.cat([xm_skip_5, xf_skip_5], dim=1) # (1,64,10,12,14)
        x = self.decoder1(x) # (1,32,10,12,14)


        # ------------------------------Scale4--------------------------------------------
        # 融合多尺度特征：(20,24,28)
        xm_skip_1_fusion4 = self.pooling_8(xm_skip_1) # (1,16,20,24,28)
        xm_skip_2_fusion4 = self.pooling_4(xm_skip_2) # (1,32,20,24,28)
        xm_skip_3_fusion4 = self.pooling(xm_skip_3) # (1,32,20,24,28)
        xm_skip_4_fusion4 = xm_skip_4 # (1,32,20,24,28)

        xm_fusion4 = torch.cat([xm_skip_1_fusion4, xm_skip_2_fusion4], dim=1)
        xm_fusion4 = torch.cat([xm_fusion4, xm_skip_3_fusion4], dim=1)
        xm_fusion4 = torch.cat([xm_fusion4, xm_skip_4_fusion4], dim=1)

        xm_fusion4 = self.fusion_m_scale4(xm_fusion4) # (1,32,20,24,28)
        xm_fusion4 = xm_fusion4 * alpha_6_1
        xm_fusion4 = self.fusion_m_scale4_2(xm_fusion4) # (1,32,20,24,28)
        xm_fusion4 = xm_fusion4 * alpha_6_2
    

        xf_skip_1_fusion4 = self.pooling_8(xf_skip_1) # (1,16,20,24,28)
        xf_skip_2_fusion4 = self.pooling_4(xf_skip_2) # (1,32,20,24,28)
        xf_skip_3_fusion4 = self.pooling(xf_skip_3) # (1,32,20,24,28)
        xf_skip_4_fusion4 = xf_skip_4 # (1,32,20,24,28)

        xf_fusion4 = torch.cat([xf_skip_1_fusion4, xf_skip_2_fusion4], dim=1) 
        xf_fusion4 = torch.cat([xf_fusion4, xf_skip_3_fusion4], dim=1)
        xf_fusion4 = torch.cat([xf_fusion4, xf_skip_4_fusion4], dim=1)
   
        xf_fusion4 = self.fusion_f_scale4(xf_fusion4)
        xf_fusion4 = xf_fusion4 * alpha_6_1
        xf_fusion4 = self.fusion_f_scale4_2(xf_fusion4) # (1,32,20,24,28) 
        xf_fusion4 = xf_fusion4 * alpha_6_2

        
        x_fusion4 = torch.cat([xm_fusion4, xf_fusion4], dim=1) # (1,64,20,24,28)

    
        # 第二个decoder
        x = self.decoder2(torch.cat([self.upsampling(x), x_fusion4], dim=1)) # (1,32,20,24,28)
        # x = self.trans_3(x)

        # ----输出1/8分辨率的变形场-----
        flow_1 = self.output_block_1(x) # (1,3,20,24,28) 
        flow_1_up = nn.functional.interpolate(flow_1, scale_factor=2,mode="trilinear")*2 # (1,3,40,48,56)


        ## ------------------------------Scale3--------------------------------------------
        # 融合多尺度特征：(40,48,56)
        xm_skip_1_fusion3 = self.pooling_4(xm_skip_1) # (1,16,40,48,56)
        xm_skip_2_fusion3 = self.pooling(xm_skip_2) # (1,32,40,48,56)
        xm_skip_3_fusion3 = xm_skip_3 # (1,32,40,48,56)
        xm_skip_4_fusion3 = nn.functional.interpolate(xm_skip_4, scale_factor=2,mode="trilinear")
        

        xm_fusion3 = torch.cat([xm_skip_1_fusion3, xm_skip_2_fusion3], dim=1)
        xm_fusion3 = torch.cat([xm_fusion3, xm_skip_3_fusion3], dim=1)
        xm_fusion3 = torch.cat([xm_fusion3, xm_skip_4_fusion3], dim=1) # (1,112,40,48,56)

        xm_fusion3 = self.fusion_m_scale3(xm_fusion3) # (1,32,40,48,56)
        xm_fusion3 = xm_fusion3 * alpha_7_1
        xm_fusion3 = self.fusion_m_scale3_2(xm_fusion3) # (1,32,40,48,56)
        xm_fusion3 = xm_fusion3 * alpha_7_2

        xw_fusion3 = self.transformer2(xm_fusion3, flow_1_up)


        xf_skip_1_fusion3 = self.pooling_4(xf_skip_1) # (1,16,40,48,56)
        xf_skip_2_fusion3 = self.pooling(xf_skip_2) # (1,32,40,48,56)
        xf_skip_3_fusion3 = xf_skip_3 # (1,32,40,48,56)

        xf_skip_4_fusion3 = nn.functional.interpolate(xf_skip_4, scale_factor=2,mode="trilinear")
        

        xf_fusion3 = torch.cat([xf_skip_1_fusion3, xf_skip_2_fusion3], dim=1)
        xf_fusion3 = torch.cat([xf_fusion3, xf_skip_3_fusion3], dim=1)
        xf_fusion3 = torch.cat([xf_fusion3, xf_skip_4_fusion3], dim=1) # (1,112,40,48,56)

        xf_fusion3 = self.fusion_f_scale3(xf_fusion3) # (1,32,40,48,56)
        xf_fusion3 = xf_fusion3 * alpha_7_1
        xf_fusion3 = self.fusion_f_scale3_2(xf_fusion3) # (1,32,40,48,56)
        xf_fusion3 = xf_fusion3 * alpha_7_2

        # concat
        x_fusion3 = torch.cat([xw_fusion3, xf_fusion3], dim=1)  # (1,64,40,48,56)

        # 第三个decoder
        x = self.decoder3(torch.cat([self.upsampling(x), x_fusion3], dim=1)) # (1,32,40,48,56)
        # x = self.trans_4(x)

        # ----输出1/4分辨率的变形场---
        delta_flow_2 = self.output_block_2(x) # (1,3,40,48,56)
        flow_2 = delta_flow_2 + flow_1_up # (1,3,40,48,56)
        flow_2_up = nn.functional.interpolate(flow_2, scale_factor=2,mode="trilinear")*2 # (1,3,80,96,112)

        ## ------------------------------Scale2--------------------------------------------
        # 融合多尺度特征：(80,96,112)
        xm_skip_1_fusion2 = self.pooling(xm_skip_1) # (1,16,80,96,112)
        xm_skip_2_fusion2 = xm_skip_2 # (1,32,80,96,112)

        xm_skip_3_fusion2 = nn.functional.interpolate(xm_skip_3, scale_factor=2,mode="trilinear")  # (1,32,80,96,112)
        
        xm_skip_4_fusion2 = nn.functional.interpolate(xm_skip_4, scale_factor=4,mode="trilinear")
        

        xm_fusion2 = torch.cat([xm_skip_1_fusion2, xm_skip_2_fusion2], dim=1)
        xm_fusion2 = torch.cat([xm_fusion2, xm_skip_3_fusion2], dim=1)
        xm_fusion2 = torch.cat([xm_fusion2, xm_skip_4_fusion2], dim=1) # (1,112,80,96,112)


        xm_fusion2 = self.fusion_m_scale2(xm_fusion2) # (1,32,80,96,112)
        xm_fusion2 = xm_fusion2 * alpha_8_1
        xm_fusion2 = self.fusion_m_scale2_2(xm_fusion2) # (1,32,80,96,112)
        xm_fusion2 = xm_fusion2 * alpha_8_2


        xw_fusion2 = self.transformer3(xm_fusion2, flow_2_up)

        xf_skip_1_fusion2 = self.pooling(xf_skip_1) # (1,16,80,96,112)
        xf_skip_2_fusion2 = xf_skip_2 # (1,32,80,96,112)

        xf_skip_3_fusion2 = nn.functional.interpolate(xf_skip_3, scale_factor=2,mode="trilinear")  # (1,32,80,96,112)
        
        xf_skip_4_fusion2 = nn.functional.interpolate(xf_skip_4, scale_factor=4,mode="trilinear")
        

        xf_fusion2 = torch.cat([xf_skip_1_fusion2, xf_skip_2_fusion2], dim=1)
        xf_fusion2 = torch.cat([xf_fusion2, xf_skip_3_fusion2], dim=1)
        xf_fusion2 = torch.cat([xf_fusion2, xf_skip_4_fusion2], dim=1) # (1,112,80,96,112)

        xf_fusion2 = self.fusion_f_scale2(xf_fusion2) # (1,32,80,96,112)
        xf_fusion2 = xf_fusion2 * alpha_8_1
        xf_fusion2 = self.fusion_f_scale2_2(xf_fusion2) # (1,32,80,96,112)
        xf_fusion2 = xf_fusion2 * alpha_8_2

        # concat
        x_fusion2 = torch.cat([xw_fusion2, xf_fusion2], dim=1)  # (1,32,80,96,112)


        # 第四个decoder
        x = self.decoder4(torch.cat([self.upsampling(x), x_fusion2], dim=1)) # (1,32,80,96,112)
        # x = self.trans_5(x)
        # print(x.shape)

        # ----输出1/2分辨率的变形场-------
        delta_flow_3 = self.output_block_3(x) # (1,3,80,96,112)
        flow_3 = delta_flow_3 + flow_2_up # (1,3,80,96,112)
        flow_3_up = nn.functional.interpolate(flow_3, scale_factor=2,mode="trilinear")*2 # (1,3,160,112,224)

        # ------------------------------Scale1--------------------------------------------
        # # 对moving feature进行warp得到moved feature
        xw_skip_1 = self.transformer4(xm_skip_1, flow_3_up)
        # concat
        x_skip_1 = torch.cat([xw_skip_1, xf_skip_1], dim=1)  # (1,32,160,192,224)
        # print(x_skip_1.shape)


        # 第四个decoder
        x = self.decoder5(torch.cat([self.upsampling(x), x_skip_1], dim=1)) # (1,32,160,192,224)

        # ----------------------输出最终的变形场------------------------------------------
        # output block
        x = self.output_block(x) # (1,16,160,192,224)
        # 生成flow场
        delta_flow_final = self.flow(x) # (1,3,160,192,224)
        flow_final = delta_flow_final + flow_3_up
            

        return flow_1,flow_2,flow_3,flow_final,delta_flow_2,delta_flow_3,delta_flow_final,ori_alpha,gate_alpha

#(with FFM) 直接融合 + 俩卷积 维度扩展 encoder和FFM 都×2(剪枝前的网络)
class dual_pyramid_VxmDense_FFM_large_GDP(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    自己写的, 两个权重共享的编码器来各自提取img_concat_wavelet的特征,
    然后特征融合, 再送入同一个解码器
    在multi-vxmdense的基础上改的

    配准网络本身配准的是输入model的source和target的尺寸, 而SPT则是new_shape的尺寸
    特别需要注意的就是输入的source和target的尺寸是否能完成四次下采样, 不能的话要注意调整网络下采样的次数
    最后生成的flow可以通过self.flow里面的卷积步长stride来改变尺寸(下采样等等)
    """

    @store_config_args
    def __init__(self,
                 inshape=(160,192,224)):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this 
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. 
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2. 
                Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)

        # # print(new_shape)
        # self.transformer1 = layers.SpatialTransformer((16,16,16))
        # self.transformer2 = layers.SpatialTransformer((32,32,32))
        # self.transformer3 = layers.SpatialTransformer((64,64,64))
        # self.transformer4 = layers.SpatialTransformer((128,128,128))

        # self.transformer1 = layers.SpatialTransformer((20,24,28))
        # self.transformer2 = layers.SpatialTransformer((40,48,56))
        # self.transformer3 = layers.SpatialTransformer((80,96,112))
        # self.transformer4 = layers.SpatialTransformer((160,192,224))

        # self.transformer1 = layers.SpatialTransformer((24,20,20))
        # self.transformer2 = layers.SpatialTransformer((48,40,40))
        # self.transformer3 = layers.SpatialTransformer((96,80,80))
        # self.transformer4 = layers.SpatialTransformer((192,160,160))

        self.transformer1 = layers.SpatialTransformer((24,20,24))
        self.transformer2 = layers.SpatialTransformer((48,40,48))
        self.transformer3 = layers.SpatialTransformer((96,80,96))
        self.transformer4 = layers.SpatialTransformer((192,160,192))
        # self.input_model = input_model
        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = MaxPooling(2)
        self.pooling_4 =MaxPooling(4)
        self.pooling_8 =MaxPooling(8)

        self.upsampling = nn.Upsample(scale_factor=2, mode='trilinear') 


        self.encoder1_m = ConvBlock(3,1,32)  # 权重共享的编码器, 提取moving
        self.encoder2_m = ConvBlock(3,32,64)
        self.encoder3_m = ConvBlock(3,64,64)
        self.encoder4_m = ConvBlock(3,64,64)
        self.encoder5_m = ConvBlock(3,64,64)

        
        self.encoder1_f = self.encoder1_m  # 权重共享的编码器, 提取fixed
        self.encoder2_f = self.encoder2_m
        self.encoder3_f = self.encoder3_m
        self.encoder4_f = self.encoder4_m
        self.encoder5_f = self.encoder5_m

        
        self.decoder1 = ConvBlock(ndims,128,32)
        self.decoder2 = ConvBlock(ndims,160,32)
        self.decoder3 = ConvBlock(ndims,160,32)
        self.decoder4 = ConvBlock(ndims,160,16)
        self.decoder5 = ConvBlock(ndims,80,16)

        self.output_block = nn.Sequential(ConvBlock(ndims,16,16),ConvBlock(ndims,16,16))
        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(16, ndims, kernel_size=3, padding=1)

        # # init flow layer with small weights and bias
        # self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        # self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        self.output_block_0 = nn.Sequential(ConvBlock(ndims,32,16),ConvBlock(ndims,16,3))
        self.output_block_1 = nn.Sequential(ConvBlock(ndims,32,16),ConvBlock(ndims,16,3))
        self.output_block_2 = nn.Sequential(ConvBlock(ndims,32,16),ConvBlock(ndims,16,3))
        self.output_block_3 = nn.Sequential(ConvBlock(ndims,16,16),ConvBlock(ndims,16,3))

        # 整合特征
        # # ------------------------------Scale4--------------------------------------------
        # self.fusion_f1_scale4 = ConvBlock(ndims,16,16)
        # self.fusion_f2_scale4 = ConvBlock(ndims,32,32)
        # self.fusion_f3_scale4 = ConvBlock(ndims,32,32)
        # self.fusion_f4_scale4 = ConvBlock(ndims,32,32)
        self.fusion_f_scale4 = ConvBlock(ndims,224,64)
        self.fusion_f_scale4_2 = ConvBlock(ndims,64,64)

        # self.fusion_m1_scale4 = self.fusion_f1_scale4
        # self.fusion_m2_scale4 = self.fusion_f2_scale4
        # self.fusion_m3_scale4 = self.fusion_f3_scale4
        # self.fusion_m4_scale4 = self.fusion_f4_scale4
        self.fusion_m_scale4 = self.fusion_f_scale4
        self.fusion_m_scale4_2 = self.fusion_f_scale4_2

        # # ------------------------------Scale3--------------------------------------------
        # self.fusion_f1_scale3 = ConvBlock(ndims,16,16)
        # self.fusion_f2_scale3 = ConvBlock(ndims,32,32)
        # self.fusion_f3_scale3 = ConvBlock(ndims,32,32)
        # self.fusion_f4_scale3 = ConvBlock(ndims,32,32)
        self.fusion_f_scale3 = ConvBlock(ndims,224,64)
        self.fusion_f_scale3_2 = ConvBlock(ndims,64,64)

        # self.fusion_m1_scale3 = self.fusion_f1_scale3
        # self.fusion_m2_scale3 = self.fusion_f2_scale3
        # self.fusion_m3_scale3 = self.fusion_f3_scale3
        # self.fusion_m4_scale3 = self.fusion_f4_scale3
        self.fusion_m_scale3 = self.fusion_f_scale3
        self.fusion_m_scale3_2 = self.fusion_f_scale3_2

        # # ------------------------------Scale2--------------------------------------------
        # self.fusion_f1_scale2 = ConvBlock(ndims,16,16)
        # self.fusion_f2_scale2 = ConvBlock(ndims,32,32)
        # self.fusion_f3_scale2 = ConvBlock(ndims,32,32)
        # self.fusion_f4_scale2 = ConvBlock(ndims,32,32)
        self.fusion_f_scale2 = ConvBlock(ndims,224,64)
        self.fusion_f_scale2_2 = ConvBlock(ndims,64,64)

        # self.fusion_m1_scale2 = self.fusion_f1_scale2
        # self.fusion_m2_scale2 = self.fusion_f2_scale2
        # self.fusion_m3_scale2 = self.fusion_f3_scale2
        # self.fusion_m4_scale2 = self.fusion_f4_scale2
        self.fusion_m_scale2 = self.fusion_f_scale2
        self.fusion_m_scale2_2 = self.fusion_f_scale2_2

        # 定义筛选特征的向量
        self.alpha_1 = nn.Parameter(torch.ones(1, 32))
        self.alpha_2 = nn.Parameter(torch.ones(1, 64))
        self.alpha_3 = nn.Parameter(torch.ones(1, 64))
        self.alpha_4 = nn.Parameter(torch.ones(1, 64))
        self.alpha_5 = nn.Parameter(torch.ones(1, 64))
        self.alpha_6_1 = nn.Parameter(torch.ones(1, 64))
        self.alpha_6_2 = nn.Parameter(torch.ones(1, 64))
        self.alpha_7_1 = nn.Parameter(torch.ones(1, 64))
        self.alpha_7_2 = nn.Parameter(torch.ones(1, 64))
        self.alpha_8_1 = nn.Parameter(torch.ones(1, 64))
        self.alpha_8_2 = nn.Parameter(torch.ones(1, 64))

        # 定义门函数
        self.gate_activation = GatingFunction()

        # 更新decay的系数
    def set_decay(self, new_decay_value):
        self.gate_activation.decay = new_decay_value

    def forward(self, source, target):
        '''
        Parameters:
            source: Source image tensor.F (moving)
            target: Target image tensor.  (fixed)
            registration: Return transformed image and flow. Default is False.
        '''
        ori_alpha_1 = self.alpha_1
        ori_alpha_2 = self.alpha_2
        ori_alpha_3 = self.alpha_3
        ori_alpha_4 = self.alpha_4
        ori_alpha_5 = self.alpha_5
        ori_alpha_6_1 = self.alpha_6_1
        ori_alpha_6_2 = self.alpha_6_2
        ori_alpha_7_1 = self.alpha_7_1
        ori_alpha_7_2 = self.alpha_7_2
        ori_alpha_8_1 = self.alpha_8_1
        ori_alpha_8_2 = self.alpha_8_2
        
        # 输入门函数
        alpha_1 = self.gate_activation(ori_alpha_1)
        alpha_2 = self.gate_activation(ori_alpha_2)
        alpha_3 = self.gate_activation(ori_alpha_3)
        alpha_4 = self.gate_activation(ori_alpha_4)
        alpha_5 = self.gate_activation(ori_alpha_5)
        alpha_6_1 = self.gate_activation(ori_alpha_6_1)
        alpha_6_2 = self.gate_activation(ori_alpha_6_2)
        alpha_7_1 = self.gate_activation(ori_alpha_7_1)
        alpha_7_2 = self.gate_activation(ori_alpha_7_2)
        alpha_8_1 = self.gate_activation(ori_alpha_8_1)
        alpha_8_2 = self.gate_activation(ori_alpha_8_2)
        ori_alpha = [ori_alpha_1,ori_alpha_2,ori_alpha_3,ori_alpha_4,ori_alpha_5,ori_alpha_6_1,ori_alpha_6_2,ori_alpha_7_1,ori_alpha_7_2,ori_alpha_8_1,ori_alpha_8_2]
        gate_alpha = [alpha_1,alpha_2,alpha_3,alpha_4,alpha_5,alpha_6_1,alpha_6_2,alpha_7_1,alpha_7_2,alpha_8_1,alpha_8_2]

        # 维度扩展
        alpha_1 = alpha_1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  
        alpha_2 = alpha_2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        alpha_3 = alpha_3.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        alpha_4 = alpha_4.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        alpha_5 = alpha_5.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        alpha_6_1 = alpha_6_1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) 
        alpha_6_2 = alpha_6_2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        alpha_7_1 = alpha_7_1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) 
        alpha_7_2 = alpha_7_2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        alpha_8_1 = alpha_8_1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) 
        alpha_8_2 = alpha_8_2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)   

        xm = source
        xf = target
        xm_skip_1 = self.encoder1_m(xm) # (1,32,160,192,224)
        xm_skip_1 = xm_skip_1 * alpha_1
        xf_skip_1 = self.encoder1_f(xf) # (1,32,160,192,224)
        xf_skip_1 = xf_skip_1 * alpha_1

        # moving encoder
        #下采样到1/2,第二个encoder
        xm = self.pooling(xm_skip_1) # (1,32,80,96,112)
        xm_skip_2 = self.encoder2_m(xm) # (1,64,80,96,112)
        xm_skip_2 = xm_skip_2 * alpha_2
        # 下采样到1/4,第三个encoder
        xm = self.pooling(xm_skip_2) # (1,64,40,48,56)
        xm_skip_3 = self.encoder3_m(xm) # (1,64,40,48,56)
        xm_skip_3 = xm_skip_3 * alpha_3
        # 下采样到1/8,第四个encoder
        xm = self.pooling(xm_skip_3)  # (1,64,20,24,28)
        xm_skip_4 = self.encoder4_m(xm) # (1,64,20,24,28)
        xm_skip_4 = xm_skip_4 * alpha_4
        # 下采样到1/16
        xm = self.pooling(xm_skip_4) # (1,64,10,12,14)
        xm_skip_5 = self.encoder5_m(xm) # (1,64,10,12,14)
        xm_skip_5 = xm_skip_5 * alpha_5


        # fixed encoder
        #下采样到1/2,第二个encoder
        xf = self.pooling(xf_skip_1)
        xf_skip_2 = self.encoder2_f(xf)
        xf_skip_2 = xf_skip_2 * alpha_2
        # 下采样到1/4,第三个encoder
        xf = self.pooling(xf_skip_2)
        xf_skip_3 = self.encoder3_f(xf)
        xf_skip_3 = xf_skip_3 * alpha_3
        # 下采样到1/8,第四个encoder
        xf = self.pooling(xf_skip_3)  
        xf_skip_4 = self.encoder4_f(xf)
        xf_skip_4 = xf_skip_4 * alpha_4
        # 下采样到1/16
        xf = self.pooling(xf_skip_4)
        xf_skip_5 = self.encoder5_f(xf) # (1,64,10,12,14)
        xf_skip_5 = xf_skip_5 * alpha_5

        

        #-------------------------------Scale5--------------------------------
        # 第一个decoder
        x = torch.cat([xm_skip_5, xf_skip_5], dim=1) # (1,128,10,12,14)
        x = self.decoder1(x) # (1,64,10,12,14)
        x_decoder1_feature = x


        # ------------------------------Scale4--------------------------------------------
        # 融合多尺度特征：(20,24,28)
        xm_skip_1_fusion4 = self.pooling_8(xm_skip_1) # (1,32,20,24,28)
        # xm_skip_1_fusion4 = self.fusion_m1_scale4(xm_skip_1_fusion4) 
        xm_skip_2_fusion4 = self.pooling_4(xm_skip_2) # (1,64,20,24,28)
        # xm_skip_2_fusion4 = self.fusion_m2_scale4(xm_skip_2_fusion4)
        xm_skip_3_fusion4 = self.pooling(xm_skip_3) # (1,64,20,24,28)
        # xm_skip_3_fusion4 = self.fusion_m3_scale4(xm_skip_3_fusion4)
        xm_skip_4_fusion4 = xm_skip_4 # (1,64,20,24,28)

        xm_fusion4 = torch.cat([xm_skip_1_fusion4, xm_skip_2_fusion4], dim=1)
        xm_fusion4 = torch.cat([xm_fusion4, xm_skip_3_fusion4], dim=1)
        xm_fusion4 = torch.cat([xm_fusion4, xm_skip_4_fusion4], dim=1)

        xm_fusion4 = self.fusion_m_scale4(xm_fusion4) # (1,64,20,24,28)
        xm_fusion4 = xm_fusion4 * alpha_6_1
        # xm_fusion4_feature_1 = xm_fusion4

        xm_fusion4 = self.fusion_m_scale4_2(xm_fusion4) # (1,64,20,24,28)
        xm_fusion4 = xm_fusion4 * alpha_6_2
        # xm_fusion4_feature_2 = xm_fusion4
    

        xf_skip_1_fusion4 = self.pooling_8(xf_skip_1) # (1,32,20,24,28)
        # xf_skip_1_fusion4 = self.fusion_f1_scale4(xf_skip_1_fusion4)
        xf_skip_2_fusion4 = self.pooling_4(xf_skip_2) # (1,64,20,24,28)
        # xf_skip_2_fusion4 = self.fusion_f2_scale4(xf_skip_2_fusion4)
        xf_skip_3_fusion4 = self.pooling(xf_skip_3) # (1,64,20,24,28)
        # xf_skip_3_fusion4 = self.fusion_f3_scale4(xf_skip_3_fusion4)
        xf_skip_4_fusion4 = xf_skip_4 # (1,64,20,24,28)

        xf_fusion4 = torch.cat([xf_skip_1_fusion4, xf_skip_2_fusion4], dim=1) 
        xf_fusion4 = torch.cat([xf_fusion4, xf_skip_3_fusion4], dim=1)
        xf_fusion4 = torch.cat([xf_fusion4, xf_skip_4_fusion4], dim=1)
   
        xf_fusion4 = self.fusion_f_scale4(xf_fusion4)
        xf_fusion4 = xf_fusion4 * alpha_6_1
        # xf_fusion4_feature_1 = xf_fusion4

        xf_fusion4 = self.fusion_f_scale4_2(xf_fusion4) # (1,64,20,24,28) 
        xf_fusion4 = xf_fusion4 * alpha_6_2
        # xf_fusion4_feature_2 = xf_fusion4

        
        x_fusion4 = torch.cat([xm_fusion4, xf_fusion4], dim=1) # (1,128,20,24,28)

    
        # 第二个decoder
        x = self.decoder2(torch.cat([self.upsampling(x), x_fusion4], dim=1)) # (1,64,20,24,28)

        x_decoder2_feature = x
        # x = self.trans_3(x)

        # ----输出1/8分辨率的变形场-----
        flow_1 = self.output_block_1(x) # (1,3,20,24,28) 
        flow_1_up = nn.functional.interpolate(flow_1, scale_factor=2,mode="trilinear")*2 # (1,3,40,48,56)


        ## ------------------------------Scale3--------------------------------------------
        # 融合多尺度特征：(40,48,56)
        xm_skip_1_fusion3 = self.pooling_4(xm_skip_1) # (1,32,40,48,56)
        # xm_skip_1_fusion3 = self.fusion_m1_scale3(xm_skip_1_fusion3) 
        xm_skip_2_fusion3 = self.pooling(xm_skip_2) # (1,64,40,48,56)
        # xm_skip_2_fusion3 = self.fusion_m2_scale3(xm_skip_2_fusion3)
        xm_skip_3_fusion3 = xm_skip_3 # (1,64,40,48,56)
        # xm_skip_3_fusion3 = self.fusion_m3_scale3(xm_skip_3_fusion3)

        # xm_skip_4_fusion3 = self.fusion_m4_scale3(xm_skip_4) # (1,32,40,48,56)
        xm_skip_4_fusion3 = nn.functional.interpolate(xm_skip_4, scale_factor=2,mode="trilinear")
        

        xm_fusion3 = torch.cat([xm_skip_1_fusion3, xm_skip_2_fusion3], dim=1)
        xm_fusion3 = torch.cat([xm_fusion3, xm_skip_3_fusion3], dim=1)
        xm_fusion3 = torch.cat([xm_fusion3, xm_skip_4_fusion3], dim=1) # (1,224,40,48,56)

        xm_fusion3 = self.fusion_m_scale3(xm_fusion3) # (1,64,40,48,56)
        xm_fusion3 = xm_fusion3 * alpha_7_1
        # xm_fusion3_feature_1 = xm_fusion3

        xm_fusion3 = self.fusion_m_scale3_2(xm_fusion3) # (1,64,40,48,56)
        xm_fusion3 = xm_fusion3 * alpha_7_2
        # xm_fusion3_feature_2 = xm_fusion3

        xw_fusion3 = self.transformer2(xm_fusion3, flow_1_up)


        xf_skip_1_fusion3 = self.pooling_4(xf_skip_1) # (1,32,40,48,56)
        # xf_skip_1_fusion3 = self.fusion_f1_scale3(xf_skip_1_fusion3) 
        xf_skip_2_fusion3 = self.pooling(xf_skip_2) # (1,64,40,48,56)
        # xf_skip_2_fusion3 = self.fusion_f2_scale3(xf_skip_2_fusion3)
        xf_skip_3_fusion3 = xf_skip_3 # (1,64,40,48,56)
        # xf_skip_3_fusion3 = self.fusion_f3_scale3(xf_skip_3_fusion3)

        # xf_skip_4_fusion3 = self.fusion_f4_scale3(xf_skip_4) # (1,32,40,48,56)
        xf_skip_4_fusion3 = nn.functional.interpolate(xf_skip_4, scale_factor=2,mode="trilinear")
        

        xf_fusion3 = torch.cat([xf_skip_1_fusion3, xf_skip_2_fusion3], dim=1)
        xf_fusion3 = torch.cat([xf_fusion3, xf_skip_3_fusion3], dim=1)
        xf_fusion3 = torch.cat([xf_fusion3, xf_skip_4_fusion3], dim=1) # (1,224,40,48,56)

        xf_fusion3 = self.fusion_f_scale3(xf_fusion3) # (1,64,40,48,56)
        xf_fusion3 = xf_fusion3 * alpha_7_1
        # xf_fusion3_feature_1 = xf_fusion3

        xf_fusion3 = self.fusion_f_scale3_2(xf_fusion3) # (1,64,40,48,56)
        xf_fusion3 = xf_fusion3 * alpha_7_2
        # xf_fusion3_feature_2 = xf_fusion3

        # concat
        x_fusion3 = torch.cat([xw_fusion3, xf_fusion3], dim=1)  # (1,128,40,48,56)

        # 第三个decoder
        x = self.decoder3(torch.cat([self.upsampling(x), x_fusion3], dim=1)) # (1,64,40,48,56)
        x_decoder3_feature = x
        # x = self.trans_4(x)

        # ----输出1/4分辨率的变形场---
        delta_flow_2 = self.output_block_2(x) # (1,3,40,48,56)
        flow_2 = delta_flow_2 + flow_1_up # (1,3,40,48,56)
        flow_2_up = nn.functional.interpolate(flow_2, scale_factor=2,mode="trilinear")*2 # (1,3,80,96,112)

        ## ------------------------------Scale2--------------------------------------------
        # 融合多尺度特征：(80,96,112)
        xm_skip_1_fusion2 = self.pooling(xm_skip_1) # (1,32,80,96,112)
        # xm_skip_1_fusion2 = self.fusion_m1_scale2(xm_skip_1_fusion2) 
        xm_skip_2_fusion2 = xm_skip_2 # (1,64,80,96,112)
        # xm_skip_2_fusion2 = self.fusion_m2_scale2(xm_skip_2_fusion2)

        # xm_skip_3_fusion2 = self.fusion_m3_scale2(xm_skip_3)
        xm_skip_3_fusion2 = nn.functional.interpolate(xm_skip_3, scale_factor=2,mode="trilinear")  # (1,64,80,96,112)
        
        # xm_skip_4_fusion2 = self.fusion_m4_scale2(xm_skip_4) # (1,64,80,96,112)
        xm_skip_4_fusion2 = nn.functional.interpolate(xm_skip_4, scale_factor=4,mode="trilinear")
        

        xm_fusion2 = torch.cat([xm_skip_1_fusion2, xm_skip_2_fusion2], dim=1)
        xm_fusion2 = torch.cat([xm_fusion2, xm_skip_3_fusion2], dim=1)
        xm_fusion2 = torch.cat([xm_fusion2, xm_skip_4_fusion2], dim=1) # (1,224,80,96,112)


        xm_fusion2 = self.fusion_m_scale2(xm_fusion2) # (1,64,80,96,112)
        xm_fusion2 = xm_fusion2 * alpha_8_1
        # xm_fusion2_feature_1 = xm_fusion2

        xm_fusion2 = self.fusion_m_scale2_2(xm_fusion2) # (1,64,80,96,112)
        xm_fusion2 = xm_fusion2 * alpha_8_2
        # xm_fusion2_feature_2 = xm_fusion2


        xw_fusion2 = self.transformer3(xm_fusion2, flow_2_up)

        xf_skip_1_fusion2 = self.pooling(xf_skip_1) # (1,16,80,96,112)
        # xf_skip_1_fusion2 = self.fusion_f1_scale2(xf_skip_1_fusion2) 
        xf_skip_2_fusion2 = xf_skip_2 # (1,32,80,96,112)
        # xf_skip_2_fusion2 = self.fusion_f2_scale2(xf_skip_2_fusion2)

        # xf_skip_3_fusion2 = self.fusion_f3_scale2(xf_skip_3)
        xf_skip_3_fusion2 = nn.functional.interpolate(xf_skip_3, scale_factor=2,mode="trilinear")  # (1,32,80,96,112)
        
        # xf_skip_4_fusion2 = self.fusion_f4_scale2(xf_skip_4) # (1,32,80,96,112)
        xf_skip_4_fusion2 = nn.functional.interpolate(xf_skip_4, scale_factor=4,mode="trilinear")
        

        xf_fusion2 = torch.cat([xf_skip_1_fusion2, xf_skip_2_fusion2], dim=1)
        xf_fusion2 = torch.cat([xf_fusion2, xf_skip_3_fusion2], dim=1)
        xf_fusion2 = torch.cat([xf_fusion2, xf_skip_4_fusion2], dim=1) # (1,112,80,96,112)

        xf_fusion2 = self.fusion_f_scale2(xf_fusion2) # (1,64,80,96,112)
        xf_fusion2 = xf_fusion2 * alpha_8_1
        # xf_fusion2_feature_1 = xf_fusion2

        xf_fusion2 = self.fusion_f_scale2_2(xf_fusion2) # (1,64,80,96,112)
        xf_fusion2 = xf_fusion2 * alpha_8_2
        # xf_fusion2_feature_2 = xf_fusion2

        # concat
        x_fusion2 = torch.cat([xw_fusion2, xf_fusion2], dim=1)  # (1,32,80,96,112)


        # 第四个decoder
        x = self.decoder4(torch.cat([self.upsampling(x), x_fusion2], dim=1)) # (1,32,80,96,112)
        x_decoder4_feature = x
        # x = self.trans_5(x)
        # print(x.shape)

        # ----输出1/2分辨率的变形场-------
        delta_flow_3 = self.output_block_3(x) # (1,3,80,96,112)
        flow_3 = delta_flow_3 + flow_2_up # (1,3,80,96,112)
        flow_3_up = nn.functional.interpolate(flow_3, scale_factor=2,mode="trilinear")*2 # (1,3,160,112,224)

        # ------------------------------Scale1--------------------------------------------
        # # 对moving feature进行warp得到moved feature
        xw_skip_1 = self.transformer4(xm_skip_1, flow_3_up)
        # concat
        x_skip_1 = torch.cat([xw_skip_1, xf_skip_1], dim=1)  # (1,64,160,192,224)
        # print(x_skip_1.shape)


        # 第四个decoder
        x = self.decoder5(torch.cat([self.upsampling(x), x_skip_1], dim=1)) # (1,32,160,192,224)
        x_decoder5_feature = x

        # encoder_features = [xf_skip_1,xf_skip_2,xf_skip_3,xf_skip_4,xf_skip_5,xm_skip_1,xm_skip_2,xm_skip_3,xm_skip_4,xm_skip_5]
        # decoder_features = [x_decoder1_feature,x_decoder2_feature,x_decoder3_feature,x_decoder4_feature,x_decoder5_feature]
        # FFM_features = [xm_fusion4_feature_1,xm_fusion4_feature_2,xf_fusion4_feature_1,xf_fusion4_feature_2,xm_fusion3_feature_1,xm_fusion3_feature_2,xf_fusion3_feature_1,xf_fusion3_feature_2,xm_fusion2_feature_1,xm_fusion2_feature_2,xf_fusion2_feature_1,xf_fusion2_feature_2]
        # features = [encoder_features, decoder_features,FFM_features]
        # ----------------------输出最终的变形场------------------------------------------
        # output block
        x = self.output_block(x) # (1,16,160,192,224)
        # 生成flow场
        delta_flow_final = self.flow(x) # (1,3,160,192,224)
        flow_final = delta_flow_final + flow_3_up
            

        return flow_1,flow_2,flow_3,flow_final,delta_flow_2,delta_flow_3,delta_flow_final,ori_alpha,gate_alpha



class dual_pyramid_VxmDense_Trans_FFM_normal_GDP(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    自己写的, 两个权重共享的编码器来各自提取img_concat_wavelet的特征,
    然后特征融合, 再送入同一个解码器
    在multi-vxmdense的基础上改的

    配准网络本身配准的是输入model的source和target的尺寸, 而SPT则是new_shape的尺寸
    特别需要注意的就是输入的source和target的尺寸是否能完成四次下采样, 不能的话要注意调整网络下采样的次数
    最后生成的flow可以通过self.flow里面的卷积步长stride来改变尺寸(下采样等等)
    """

    @store_config_args
    def __init__(self,
                 inshape=(160,192,224)):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this 
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. 
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2. 
                Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)

        # # print(new_shape)
        # self.transformer2 = layers.SpatialTransformer((40,48,56))
        # self.transformer3 = layers.SpatialTransformer((80,96,112))
        # self.transformer4 = layers.SpatialTransformer((160,192,224))

        self.transformer2 = layers.SpatialTransformer((48,40,48))
        self.transformer3 = layers.SpatialTransformer((96,80,96))
        self.transformer4 = layers.SpatialTransformer((192,160,192))

        # self.transformer2 = layers.SpatialTransformer((48,40,40))
        # self.transformer3 = layers.SpatialTransformer((96,80,80))
        # self.transformer4 = layers.SpatialTransformer((192,160,160))

        # self.transformer2 = layers.SpatialTransformer((32,32,32))
        # self.transformer3 = layers.SpatialTransformer((64,64,64))
        # self.transformer4 = layers.SpatialTransformer((128,128,128))
        # self.input_model = input_model
        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = MaxPooling(2)
        self.pooling_4 =MaxPooling(4)
        self.pooling_8 =MaxPooling(8)

        self.upsampling = nn.Upsample(scale_factor=2, mode='trilinear') 


        self.encoder1_m = ConvBlock(3,1,16)  # 权重共享的编码器, 提取moving
        self.encoder2_m = ConvBlock(3,16,32)
        self.encoder3_m = ConvBlock(3,32,32)
        self.encoder4_m = ConvBlock(3,32,32)
        self.encoder5_m = ConvBlock(3,32,32)

        
        self.encoder1_f = self.encoder1_m  # 权重共享的编码器, 提取fixed
        self.encoder2_f = self.encoder2_m
        self.encoder3_f = self.encoder3_m
        self.encoder4_f = self.encoder4_m
        self.encoder5_f = self.encoder5_m

        
        self.decoder1 = ConvBlock(ndims,64,32)
        self.decoder2 = ConvBlock(ndims,96,32)
        self.decoder3 = ConvBlock(ndims,96,32)
        self.decoder4 = ConvBlock(ndims,96,16)
        self.decoder5 = ConvBlock(ndims,48,16)

        self.trans_2 = SwinTrans_stage_block(embed_dim=32,    # 16
                                             num_layers=2,
                                             num_heads=2,    # 1
                                             window_size=[5,5,5],
                                             use_checkpoint=False)
        self.trans_3 = SwinTrans_stage_block(embed_dim=32,    # 32
                                             num_layers=2,
                                             num_heads=2,   # 2
                                             window_size=[5,5,5],
                                             use_checkpoint=False)
        self.trans_4 = SwinTrans_stage_block(embed_dim=32,    # 64
                                             num_layers=2,
                                             num_heads=2,   #4
                                             window_size=[5,5,5],
                                             use_checkpoint=False)
        self.trans_5 = SwinTrans_stage_block(embed_dim=16, # 128
                                             num_layers=4,
                                             num_heads=1, # 8
                                             window_size=[5,5,5],
                                             use_checkpoint=False)

        self.output_block = nn.Sequential(ConvBlock(ndims,16,16),ConvBlock(ndims,16,16))
        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(16, ndims, kernel_size=3, padding=1)

        # # init flow layer with small weights and bias
        # self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        # self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        self.output_block_1 = nn.Sequential(ConvBlock(ndims,32,16),ConvBlock(ndims,16,3))
        self.output_block_2 = nn.Sequential(ConvBlock(ndims,32,16),ConvBlock(ndims,16,3))
        self.output_block_3 = nn.Sequential(ConvBlock(ndims,16,16),ConvBlock(ndims,16,3))

        # 整合特征
        # # ------------------------------Scale4--------------------------------------------
        # self.fusion_f1_scale4 = self.resblock_seq(16,16)
        # self.fusion_f2_scale4 = self.resblock_seq(32,32)
        # self.fusion_f3_scale4 = self.resblock_seq(32,32)
        # self.fusion_f4_scale4 = self.resblock_seq(32,32)
        self.fusion_f_scale4 = ConvBlock(ndims,112,32)
        self.fusion_f_scale4_2 = ConvBlock(ndims,32,32)

        # self.fusion_m1_scale4 = self.fusion_f1_scale4
        # self.fusion_m2_scale4 = self.fusion_f2_scale4
        # self.fusion_m3_scale4 = self.fusion_f3_scale4
        # self.fusion_m4_scale4 = self.fusion_f4_scale4
        self.fusion_m_scale4 = self.fusion_f_scale4
        self.fusion_m_scale4_2 = self.fusion_f_scale4_2

        # # ------------------------------Scale3--------------------------------------------
        # self.fusion_f1_scale3 = self.resblock_seq(16,16)
        # self.fusion_f2_scale3 = self.resblock_seq(32,32)
        # self.fusion_f3_scale3 = self.resblock_seq(32,32)
        # self.fusion_f4_scale3 = self.resblock_seq(32,32)
        self.fusion_f_scale3 = ConvBlock(ndims,112,32)
        self.fusion_f_scale3_2 = ConvBlock(ndims,32,32)

        # self.fusion_m1_scale3 = self.fusion_f1_scale3
        # self.fusion_m2_scale3 = self.fusion_f2_scale3
        # self.fusion_m3_scale3 = self.fusion_f3_scale3
        # self.fusion_m4_scale3 = self.fusion_f4_scale3
        self.fusion_m_scale3 = self.fusion_f_scale3
        self.fusion_m_scale3_2 = self.fusion_f_scale3_2

        # # ------------------------------Scale2--------------------------------------------
        # self.fusion_f1_scale2 = self.resblock_seq(16,16)
        # self.fusion_f2_scale2 = self.resblock_seq(32,32)
        # self.fusion_f3_scale2 = self.resblock_seq(32,32)
        # self.fusion_f4_scale2 = self.resblock_seq(32,32)
        self.fusion_f_scale2 = ConvBlock(ndims,112,32)
        self.fusion_f_scale2_2 = ConvBlock(ndims,32,32)

        # self.fusion_m1_scale2 = self.fusion_f1_scale2
        # self.fusion_m2_scale2 = self.fusion_f2_scale2
        # self.fusion_m3_scale2 = self.fusion_f3_scale2
        # self.fusion_m4_scale2 = self.fusion_f4_scale2
        self.fusion_m_scale2 = self.fusion_f_scale2
        self.fusion_m_scale2_2 = self.fusion_f_scale2_2

        # 定义筛选特征的向量
        self.alpha_1 = nn.Parameter(torch.ones(1, 16))
        self.alpha_2 = nn.Parameter(torch.ones(1, 32))
        self.alpha_3 = nn.Parameter(torch.ones(1, 32))
        self.alpha_4 = nn.Parameter(torch.ones(1, 32))
        self.alpha_5 = nn.Parameter(torch.ones(1, 32))
        self.alpha_6_1 = nn.Parameter(torch.ones(1, 32))
        self.alpha_6_2 = nn.Parameter(torch.ones(1, 32))
        self.alpha_7_1 = nn.Parameter(torch.ones(1, 32))
        self.alpha_7_2 = nn.Parameter(torch.ones(1, 32))
        self.alpha_8_1 = nn.Parameter(torch.ones(1, 32))
        self.alpha_8_2 = nn.Parameter(torch.ones(1, 32))

        # 定义门函数
        self.gate_activation = GatingFunction()

        # 更新decay的系数
    def set_decay(self, new_decay_value):
        self.gate_activation.decay = new_decay_value

    
    def forward(self, source, target):
        '''
        Parameters:
            source: Source image tensor.F (moving)
            target: Target image tensor.  (fixed)
            registration: Return transformed image and flow. Default is False.
        '''

        ori_alpha_1 = self.alpha_1
        ori_alpha_2 = self.alpha_2
        ori_alpha_3 = self.alpha_3
        ori_alpha_4 = self.alpha_4
        ori_alpha_5 = self.alpha_5
        ori_alpha_6_1 = self.alpha_6_1
        ori_alpha_6_2 = self.alpha_6_2
        ori_alpha_7_1 = self.alpha_7_1
        ori_alpha_7_2 = self.alpha_7_2
        ori_alpha_8_1 = self.alpha_8_1
        ori_alpha_8_2 = self.alpha_8_2
        
        # 输入门函数
        alpha_1 = self.gate_activation(ori_alpha_1)
        alpha_2 = self.gate_activation(ori_alpha_2)
        alpha_3 = self.gate_activation(ori_alpha_3)
        alpha_4 = self.gate_activation(ori_alpha_4)
        alpha_5 = self.gate_activation(ori_alpha_5)
        alpha_6_1 = self.gate_activation(ori_alpha_6_1)
        alpha_6_2 = self.gate_activation(ori_alpha_6_2)
        alpha_7_1 = self.gate_activation(ori_alpha_7_1)
        alpha_7_2 = self.gate_activation(ori_alpha_7_2)
        alpha_8_1 = self.gate_activation(ori_alpha_8_1)
        alpha_8_2 = self.gate_activation(ori_alpha_8_2)
        ori_alpha = [ori_alpha_1,ori_alpha_2,ori_alpha_3,ori_alpha_4,ori_alpha_5,ori_alpha_6_1,ori_alpha_6_2,ori_alpha_7_1,ori_alpha_7_2,ori_alpha_8_1,ori_alpha_8_2]
        gate_alpha = [alpha_1,alpha_2,alpha_3,alpha_4,alpha_5,alpha_6_1,alpha_6_2,alpha_7_1,alpha_7_2,alpha_8_1,alpha_8_2]

        # 维度扩展
        alpha_1 = alpha_1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  
        alpha_2 = alpha_2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        alpha_3 = alpha_3.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        alpha_4 = alpha_4.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        alpha_5 = alpha_5.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        alpha_6_1 = alpha_6_1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) 
        alpha_6_2 = alpha_6_2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        alpha_7_1 = alpha_7_1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) 
        alpha_7_2 = alpha_7_2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        alpha_8_1 = alpha_8_1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) 
        alpha_8_2 = alpha_8_2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  


        xm = source
        xf = target
        xm_skip_1 = self.encoder1_m(xm) # (1,16,160,192,224)
        xm_skip_1 = xm_skip_1 * alpha_1
        xf_skip_1 = self.encoder1_f(xf) # (1,16,160,192,224)
        xf_skip_1 = xf_skip_1 * alpha_1

        # moving encoder
        #下采样到1/2,第二个encoder
        xm = self.pooling(xm_skip_1) # (1,16,80,96,112)
        xm_skip_2 = self.encoder2_m(xm) # (1,16,80,96,112)
        xm_skip_2 = xm_skip_2 * alpha_2
        # 下采样到1/4,第三个encoder
        xm = self.pooling(xm_skip_2) # (1,16,40,48,56)
        xm_skip_3 = self.encoder3_m(xm) # (1,32,40,48,56)
        xm_skip_3 = xm_skip_3 * alpha_3
        # 下采样到1/8,第四个encoder
        xm = self.pooling(xm_skip_3)  # (1,32,20,24,28)
        xm_skip_4 = self.encoder4_m(xm) # (1,32,20,24,28)
        xm_skip_4 = xm_skip_4 * alpha_4
        # 下采样到1/16
        xm = self.pooling(xm_skip_4) # (1,32,10,12,14)
        xm_skip_5 = self.encoder5_m(xm)
        xm_skip_5 = xm_skip_5 * alpha_5


        # fixed encoder
        #下采样到1/2,第二个encoder
        xf = self.pooling(xf_skip_1)
        xf_skip_2 = self.encoder2_f(xf)
        xf_skip_2 = xf_skip_2 * alpha_2
        # 下采样到1/4,第三个encoder
        xf = self.pooling(xf_skip_2)
        xf_skip_3 = self.encoder3_f(xf)
        xf_skip_3 = xf_skip_3 * alpha_3
        # 下采样到1/8,第四个encoder
        xf = self.pooling(xf_skip_3)  
        xf_skip_4 = self.encoder4_f(xf)
        xf_skip_4 = xf_skip_4 * alpha_4
        # 下采样到1/16
        xf = self.pooling(xf_skip_4)
        xf_skip_5 = self.encoder5_f(xf)
        xf_skip_5 = xf_skip_5 * alpha_5

        # 第一个decoder
        x = torch.cat([xm_skip_5, xf_skip_5], dim=1) # (1,64,10,12,14)
        x = self.decoder1(x) # (1,32,10,12,14)
        x = self.trans_2(x)

        # ------------------------------Scale4--------------------------------------------
        # 融合多尺度特征：(20,24,28)
        xm_skip_1_fusion4 = self.pooling_8(xm_skip_1) # (1,16,20,24,28)
        # xm_skip_1_fusion4 = self.fusion_m1_scale4(xm_skip_1_fusion4) 
        xm_skip_2_fusion4 = self.pooling_4(xm_skip_2) # (1,16,20,24,28)
        # xm_skip_2_fusion4 = self.fusion_m2_scale4(xm_skip_2_fusion4)
        xm_skip_3_fusion4 = self.pooling(xm_skip_3) # (1,32,20,24,28)
        # xm_skip_3_fusion4 = self.fusion_m3_scale4(xm_skip_3_fusion4)
        xm_skip_4_fusion4 = xm_skip_4 # (1,32,20,24,28)

        xm_fusion4 = torch.cat([xm_skip_1_fusion4, xm_skip_2_fusion4], dim=1)
        xm_fusion4 = torch.cat([xm_fusion4, xm_skip_3_fusion4], dim=1)
        xm_fusion4 = torch.cat([xm_fusion4, xm_skip_4_fusion4], dim=1)

        xm_fusion4 = self.fusion_m_scale4(xm_fusion4) # (1,32,20,24,28)
        xm_fusion4 = xm_fusion4 * alpha_6_1
        xm_fusion4 = self.fusion_m_scale4_2(xm_fusion4) # (1,32,20,24,28)
        xm_fusion4 = xm_fusion4 * alpha_6_2

        xf_skip_1_fusion4 = self.pooling_8(xf_skip_1) # (1,16,20,24,28)
        # xf_skip_1_fusion4 = self.fusion_f1_scale4(xf_skip_1_fusion4)
        xf_skip_2_fusion4 = self.pooling_4(xf_skip_2) # (1,16,20,24,28)
        # xf_skip_2_fusion4 = self.fusion_f2_scale4(xf_skip_2_fusion4)
        xf_skip_3_fusion4 = self.pooling(xf_skip_3) # (1,32,20,24,28)
        # xf_skip_3_fusion4 = self.fusion_f3_scale4(xf_skip_3_fusion4)
        xf_skip_4_fusion4 = xf_skip_4 # (1,32,20,24,28)

        xf_fusion4 = torch.cat([xf_skip_1_fusion4, xf_skip_2_fusion4], dim=1) 
        xf_fusion4 = torch.cat([xf_fusion4, xf_skip_3_fusion4], dim=1)
        xf_fusion4 = torch.cat([xf_fusion4, xf_skip_4_fusion4], dim=1)
   
        xf_fusion4 = self.fusion_f_scale4(xf_fusion4) # (1,32,20,24,28)
        xf_fusion4 = xf_fusion4 * alpha_6_1
        xf_fusion4 = self.fusion_f_scale4_2(xf_fusion4)
        xf_fusion4 = xf_fusion4 * alpha_6_2

        x_fusion4 = torch.cat([xm_fusion4, xf_fusion4], dim=1) # (1,64,20,24,28)

    
        # 第二个decoder
        x = self.decoder2(torch.cat([self.upsampling(x), x_fusion4], dim=1)) # (1,32,20,24,28)
        x = self.trans_3(x)

        # ----输出1/8分辨率的变形场-----
        flow_1 = self.output_block_1(x) # (1,3,20,24,28) 
        flow_1_up = nn.functional.interpolate(flow_1, scale_factor=2,mode="trilinear")*2 # (1,3,40,48,56)



        ## ------------------------------Scale3--------------------------------------------
        # 融合多尺度特征：(40,48,56)
        xm_skip_1_fusion3 = self.pooling_4(xm_skip_1) # (1,16,40,48,56)
        # xm_skip_1_fusion3 = self.fusion_m1_scale3(xm_skip_1_fusion3) 
        xm_skip_2_fusion3 = self.pooling(xm_skip_2) # (1,16,40,48,56)
        # xm_skip_2_fusion3 = self.fusion_m2_scale3(xm_skip_2_fusion3)
        xm_skip_3_fusion3 = xm_skip_3 # (1,32,40,48,56)
        # xm_skip_3_fusion3 = self.fusion_m3_scale3(xm_skip_3_fusion3)

        # xm_skip_4_fusion3 = self.fusion_m4_scale3(xm_skip_4) # (1,32,40,48,56)
        xm_skip_4_fusion3 = nn.functional.interpolate(xm_skip_4, scale_factor=2,mode="trilinear")
        

        xm_fusion3 = torch.cat([xm_skip_1_fusion3, xm_skip_2_fusion3], dim=1)
        xm_fusion3 = torch.cat([xm_fusion3, xm_skip_3_fusion3], dim=1)
        xm_fusion3 = torch.cat([xm_fusion3, xm_skip_4_fusion3], dim=1) # (1,88,40,48,56)

        xw_fusion3 = self.transformer2(xm_fusion3, flow_1_up)

        xw_fusion3 = self.fusion_m_scale3(xw_fusion3) # (1,32,40,48,56)
        xw_fusion3 = xw_fusion3 * alpha_7_1
        xw_fusion3 = self.fusion_m_scale3_2(xw_fusion3) # (1,32,40,48,56)
        xw_fusion3 = xw_fusion3 * alpha_7_2


        xf_skip_1_fusion3 = self.pooling_4(xf_skip_1) # (1,16,40,48,56)
        # xf_skip_1_fusion3 = self.fusion_f1_scale3(xf_skip_1_fusion3) 
        xf_skip_2_fusion3 = self.pooling(xf_skip_2) # (1,16,40,48,56)
        # xf_skip_2_fusion3 = self.fusion_f2_scale3(xf_skip_2_fusion3)
        xf_skip_3_fusion3 = xf_skip_3 # (1,32,40,48,56)
        # xf_skip_3_fusion3 = self.fusion_f3_scale3(xf_skip_3_fusion3)

        # xf_skip_4_fusion3 = self.fusion_f4_scale3(xf_skip_4) # (1,32,40,48,56)
        xf_skip_4_fusion3 = nn.functional.interpolate(xf_skip_4, scale_factor=2,mode="trilinear")
        

        xf_fusion3 = torch.cat([xf_skip_1_fusion3, xf_skip_2_fusion3], dim=1)
        xf_fusion3 = torch.cat([xf_fusion3, xf_skip_3_fusion3], dim=1)
        xf_fusion3 = torch.cat([xf_fusion3, xf_skip_4_fusion3], dim=1) # (1,88,40,48,56)

        xf_fusion3 = self.fusion_f_scale3(xf_fusion3) # (1,32,40,48,56)
        xf_fusion3 = xf_fusion3 * alpha_7_1
        xf_fusion3 = self.fusion_f_scale3_2(xf_fusion3) # (1,32,40,48,56)
        xf_fusion3 = xf_fusion3 * alpha_7_2

        # concat
        x_fusion3 = torch.cat([xw_fusion3, xf_fusion3], dim=1)  # (1,64,40,48,56)

        # 第三个decoder
        x = self.decoder3(torch.cat([self.upsampling(x), x_fusion3], dim=1)) # (1,32,40,48,56)
        x = self.trans_4(x)

        # ----输出1/4分辨率的变形场---
        delta_flow_2 = self.output_block_2(x) # (1,3,40,48,56)
        flow_2 = delta_flow_2 + flow_1_up # (1,3,40,48,56)
        flow_2_up = nn.functional.interpolate(flow_2, scale_factor=2,mode="trilinear")*2 # (1,3,80,96,112)

        ## ------------------------------Scale2--------------------------------------------
        # 融合多尺度特征：(80,96,112)
        xm_skip_1_fusion2 = self.pooling(xm_skip_1) # (1,16,80,96,112)
        # xm_skip_1_fusion2 = self.fusion_m1_scale2(xm_skip_1_fusion2) 
        xm_skip_2_fusion2 = xm_skip_2 # (1,16,80,96,112)
        # xm_skip_2_fusion2 = self.fusion_m2_scale2(xm_skip_2_fusion2)

        # xm_skip_3_fusion2 = self.fusion_m3_scale2(xm_skip_3)
        xm_skip_3_fusion2 = nn.functional.interpolate(xm_skip_3, scale_factor=2,mode="trilinear")  # (1,32,80,96,112)
        
        # xm_skip_4_fusion2 = self.fusion_m4_scale2(xm_skip_4) # (1,32,80,96,112)
        xm_skip_4_fusion2 = nn.functional.interpolate(xm_skip_4, scale_factor=4,mode="trilinear")
        

        xm_fusion2 = torch.cat([xm_skip_1_fusion2, xm_skip_2_fusion2], dim=1)
        xm_fusion2 = torch.cat([xm_fusion2, xm_skip_3_fusion2], dim=1)
        xm_fusion2 = torch.cat([xm_fusion2, xm_skip_4_fusion2], dim=1) # (1,96,80,96,112)

        xw_fusion2 = self.transformer3(xm_fusion2, flow_2_up)

        xw_fusion2 = self.fusion_m_scale2(xw_fusion2) # (1,32,80,96,112)
        xw_fusion2 = xw_fusion2 * alpha_8_1
        xw_fusion2 = self.fusion_m_scale2_2(xw_fusion2) # (1,32,80,96,112)
        xw_fusion2 = xw_fusion2 * alpha_8_2

        xf_skip_1_fusion2 = self.pooling(xf_skip_1) # (1,16,80,96,112)
        # xf_skip_1_fusion2 = self.fusion_f1_scale2(xf_skip_1_fusion2) 
        xf_skip_2_fusion2 = xf_skip_2 # (1,16,80,96,112)
        # xf_skip_2_fusion2 = self.fusion_f2_scale2(xf_skip_2_fusion2)

        # xf_skip_3_fusion2 = self.fusion_f3_scale2(xf_skip_3)
        xf_skip_3_fusion2 = nn.functional.interpolate(xf_skip_3, scale_factor=2,mode="trilinear")  # (1,32,80,96,112)
        
        # xf_skip_4_fusion2 = self.fusion_f4_scale2(xf_skip_4) # (1,32,80,96,112)
        xf_skip_4_fusion2 = nn.functional.interpolate(xf_skip_4, scale_factor=4,mode="trilinear")
        

        xf_fusion2 = torch.cat([xf_skip_1_fusion2, xf_skip_2_fusion2], dim=1)
        xf_fusion2 = torch.cat([xf_fusion2, xf_skip_3_fusion2], dim=1)
        xf_fusion2 = torch.cat([xf_fusion2, xf_skip_4_fusion2], dim=1) # (1,112,80,96,112)

        xf_fusion2 = self.fusion_f_scale2(xf_fusion2) # (1,16,80,96,112)
        xf_fusion2 = xf_fusion2 * alpha_8_1
        xf_fusion2 = self.fusion_f_scale2_2(xf_fusion2) # (1,16,80,96,112)
        xf_fusion2 = xf_fusion2 * alpha_8_2

        # concat
        x_fusion2 = torch.cat([xw_fusion2, xf_fusion2], dim=1)  # (1,32,80,96,112)


        # 第四个decoder
        x = self.decoder4(torch.cat([self.upsampling(x), x_fusion2], dim=1)) # (1,32,80,96,112)
        x = self.trans_5(x)
        # print(x.shape)

        # ----输出1/2分辨率的变形场-------
        delta_flow_3 = self.output_block_3(x) # (1,3,80,96,112)
        flow_3 = delta_flow_3 + flow_2_up # (1,3,80,96,112)
        flow_3_up = nn.functional.interpolate(flow_3, scale_factor=2,mode="trilinear")*2 # (1,3,160,112,224)

        # ------------------------------Scale1--------------------------------------------
        # # 对moving feature进行warp得到moved feature
        xw_skip_1 = self.transformer4(xm_skip_1, flow_3_up)
        # concat
        x_skip_1 = torch.cat([xw_skip_1, xf_skip_1], dim=1)  # (1,32,160,192,224)
        # print(x_skip_1.shape)


        # 第四个decoder
        x = self.decoder5(torch.cat([self.upsampling(x), x_skip_1], dim=1)) # (1,32,160,192,224)

        # ----------------------输出最终的变形场------------------------------------------
        # output block
        x = self.output_block(x) # (1,16,160,192,224)
        # 生成flow场
        delta_flow_final = self.flow(x) # (1,3,160,192,224)
        flow_final = delta_flow_final + flow_3_up
            

        return flow_1,flow_2,flow_3,flow_final,delta_flow_2,delta_flow_3,delta_flow_final,ori_alpha,gate_alpha




#(with FFM) 直接融合 + 俩卷积 (剪枝后的网络) encoder和FFM 都×3
class dual_pyramid_VxmDense_FFM_huge_adaptive_val(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    自己写的, 两个权重共享的编码器来各自提取img_concat_wavelet的特征,
    然后特征融合, 再送入同一个解码器
    在multi-vxmdense的基础上改的

    配准网络本身配准的是输入model的source和target的尺寸, 而SPT则是new_shape的尺寸
    特别需要注意的就是输入的source和target的尺寸是否能完成四次下采样, 不能的话要注意调整网络下采样的次数
    最后生成的flow可以通过self.flow里面的卷积步长stride来改变尺寸(下采样等等)
    """

    @store_config_args
    def __init__(self,
                 inshape=(160,192,224),
                 list_num=[8,15,4,2,96,96,96,64,64,48,48]):
        """ 
        Parameters:list_num=[15,8,2,2,96,96,96,64,64,48,48][15,8,2,2,96,96,96,63,58,48,47] Ct:[8,15,4,2,96,96,96,64,64,48,48]
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this 
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. 
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2. 
                Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)

        # # print(new_shape)
        # self.transformer1 = layers.SpatialTransformer((20,24,28))
        # self.transformer2 = layers.SpatialTransformer((40,48,56))
        # self.transformer3 = layers.SpatialTransformer((80,96,112))
        # self.transformer4 = layers.SpatialTransformer((160,192,224))

        # self.transformer1 = layers.SpatialTransformer((16,16,16))
        # self.transformer2 = layers.SpatialTransformer((32,32,32))
        # self.transformer3 = layers.SpatialTransformer((64,64,64))
        # self.transformer4 = layers.SpatialTransformer((128,128,128))

        
        self.transformer1 = layers.SpatialTransformer(tuple(d // 8 for d in inshape))
        self.transformer2 = layers.SpatialTransformer(tuple(d // 4 for d in inshape))
        self.transformer3 = layers.SpatialTransformer(tuple(d // 2 for d in inshape))
        self.transformer4 = layers.SpatialTransformer(inshape)

        # self.transformer1 = layers.SpatialTransformer((24,20,20))
        # self.transformer2 = layers.SpatialTransformer((48,40,40))
        # self.transformer3 = layers.SpatialTransformer((96,80,80))
        # self.transformer4 = layers.SpatialTransformer((192,160,160))



        # self.input_model = input_model
        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = MaxPooling(2)
        self.pooling_4 =MaxPooling(4)
        self.pooling_8 =MaxPooling(8)

        self.upsampling = nn.Upsample(scale_factor=2, mode='trilinear') 


        self.encoder1_m = ConvBlock(3,1,list_num[0])  # 权重共享的编码器, 提取moving
        self.encoder2_m = ConvBlock(3,list_num[0],list_num[1])
        self.encoder3_m = ConvBlock(3,list_num[1],list_num[2])
        self.encoder4_m = ConvBlock(3,list_num[2],list_num[3])
        self.encoder5_m = ConvBlock(3,list_num[3],list_num[4])

        
        self.encoder1_f = self.encoder1_m  # 权重共享的编码器, 提取fixed
        self.encoder2_f = self.encoder2_m
        self.encoder3_f = self.encoder3_m
        self.encoder4_f = self.encoder4_m
        self.encoder5_f = self.encoder5_m

        
        self.decoder1 = ConvBlock(ndims,list_num[4]+list_num[4],32)
        self.decoder2 = ConvBlock(ndims,list_num[6]+list_num[6]+32,32)
        self.decoder3 = ConvBlock(ndims,list_num[8]+list_num[8]+32,32)
        self.decoder4 = ConvBlock(ndims,list_num[10]+list_num[10]+32,16)
        self.decoder5 = ConvBlock(ndims,list_num[0]+list_num[0]+16,16)

        self.output_block = nn.Sequential(ConvBlock(ndims,16,16),ConvBlock(ndims,16,16))
        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(16, ndims, kernel_size=3, padding=1)

        # # init flow layer with small weights and bias
        # self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        # self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        self.output_block_0 = nn.Sequential(ConvBlock(ndims,32,16),ConvBlock(ndims,16,3))
        self.output_block_1 = nn.Sequential(ConvBlock(ndims,32,16),ConvBlock(ndims,16,3))
        self.output_block_2 = nn.Sequential(ConvBlock(ndims,32,16),ConvBlock(ndims,16,3))
        self.output_block_3 = nn.Sequential(ConvBlock(ndims,16,16),ConvBlock(ndims,16,3))

        # 整合特征
        # # ------------------------------Scale4--------------------------------------------
        self.fusion_f_scale4 = ConvBlock(ndims,list_num[0]+list_num[1]+list_num[2]+list_num[3],list_num[5])
        self.fusion_f_scale4_2 = ConvBlock(ndims,list_num[5],list_num[6])


        self.fusion_m_scale4 = self.fusion_f_scale4
        self.fusion_m_scale4_2 = self.fusion_f_scale4_2

        # # ------------------------------Scale3--------------------------------------------
        self.fusion_f_scale3 = ConvBlock(ndims,list_num[0]+list_num[1]+list_num[2]+list_num[3],list_num[7])
        self.fusion_f_scale3_2 = ConvBlock(ndims,list_num[7],list_num[8])


        self.fusion_m_scale3 = self.fusion_f_scale3
        self.fusion_m_scale3_2 = self.fusion_f_scale3_2

        # # ------------------------------Scale2--------------------------------------------
        self.fusion_f_scale2 = ConvBlock(ndims,list_num[0]+list_num[1]+list_num[2]+list_num[3],list_num[9])
        self.fusion_f_scale2_2 = ConvBlock(ndims,list_num[9],list_num[10])


        self.fusion_m_scale2 = self.fusion_f_scale2
        self.fusion_m_scale2_2 = self.fusion_f_scale2_2


    
    def forward(self, source, target):
        '''
        Parameters:
            source: Source image tensor.F (moving)
            target: Target image tensor.  (fixed)
            registration: Return transformed image and flow. Default is False.

        ''' 
        xm = source
        xf = target
        xm_skip_1 = self.encoder1_m(xm) # (1,48,160,192,224)
        # xm_skip_1 = xm_skip_1
        xf_skip_1 = self.encoder1_f(xf) # (1,48,160,192,224)
        # xf_skip_1 = xf_skip_1

        # moving encoder
        #下采样到1/2,第二个encoder
        xm = self.pooling(xm_skip_1) # (1,16,80,96,112)
        xm_skip_2 = self.encoder2_m(xm) # (1,32,80,96,112)
        # xm_skip_2 = xm_skip_2
        # 下采样到1/4,第三个encoder
        xm = self.pooling(xm_skip_2) # (1,32,40,48,56)
        xm_skip_3 = self.encoder3_m(xm) # (1,32,40,48,56)
        # xm_skip_3 = xm_skip_3
        # 下采样到1/8,第四个encoder
        xm = self.pooling(xm_skip_3)  # (1,32,20,24,28)
        xm_skip_4 = self.encoder4_m(xm) # (1,32,20,24,28)
        # xm_skip_4 = xm_skip_4
        # 下采样到1/16
        xm = self.pooling(xm_skip_4) # (1,32,10,12,14)
        xm_skip_5 = self.encoder5_m(xm) # (1,32,10,12,14)
        # xm_skip_5 = xm_skip_5

        features_xm = [xm_skip_1,xm_skip_2,xm_skip_3,xm_skip_4,xm_skip_5]


        # fixed encoder
        #下采样到1/2,第二个encoder
        xf = self.pooling(xf_skip_1)
        xf_skip_2 = self.encoder2_f(xf)
        # xf_skip_2 = xf_skip_2
        # 下采样到1/4,第三个encoder
        xf = self.pooling(xf_skip_2)
        xf_skip_3 = self.encoder3_f(xf)
        # xf_skip_3 = xf_skip_3
        # 下采样到1/8,第四个encoder
        xf = self.pooling(xf_skip_3)  
        xf_skip_4 = self.encoder4_f(xf)
        # xf_skip_4 = xf_skip_4
        # 下采样到1/16
        xf = self.pooling(xf_skip_4)
        xf_skip_5 = self.encoder5_f(xf) # (1,32,10,12,14)
        # xf_skip_5 = xf_skip_5

        features_xf = [xf_skip_1,xf_skip_2,xf_skip_3,xf_skip_4,xf_skip_5]

        #-------------------------------Scale5--------------------------------
        # 第一个decoder
        x = torch.cat([xm_skip_5, xf_skip_5], dim=1) # (1,64,10,12,14)
        x = self.decoder1(x) # (1,32,10,12,14)


        # ------------------------------Scale4--------------------------------------------
        # 融合多尺度特征：(20,24,28)
        xm_skip_1_fusion4 = self.pooling_8(xm_skip_1) # (1,16,20,24,28)
        xm_skip_2_fusion4 = self.pooling_4(xm_skip_2) # (1,32,20,24,28)
        xm_skip_3_fusion4 = self.pooling(xm_skip_3) # (1,32,20,24,28)
        xm_skip_4_fusion4 = xm_skip_4 # (1,32,20,24,28)

        xm_fusion4 = torch.cat([xm_skip_1_fusion4, xm_skip_2_fusion4], dim=1)
        xm_fusion4 = torch.cat([xm_fusion4, xm_skip_3_fusion4], dim=1)
        xm_fusion4 = torch.cat([xm_fusion4, xm_skip_4_fusion4], dim=1)

        xm_fusion4 = self.fusion_m_scale4(xm_fusion4) # (1,32,20,24,28)
        xm_fusion4 = xm_fusion4
        xm_fusion4 = self.fusion_m_scale4_2(xm_fusion4) # (1,32,20,24,28)
        xm_fusion4 = xm_fusion4
    

        xf_skip_1_fusion4 = self.pooling_8(xf_skip_1) # (1,16,20,24,28)
        xf_skip_2_fusion4 = self.pooling_4(xf_skip_2) # (1,32,20,24,28)
        xf_skip_3_fusion4 = self.pooling(xf_skip_3) # (1,32,20,24,28)
        xf_skip_4_fusion4 = xf_skip_4 # (1,32,20,24,28)

        xf_fusion4 = torch.cat([xf_skip_1_fusion4, xf_skip_2_fusion4], dim=1) 
        xf_fusion4 = torch.cat([xf_fusion4, xf_skip_3_fusion4], dim=1)
        xf_fusion4 = torch.cat([xf_fusion4, xf_skip_4_fusion4], dim=1)
   
        xf_fusion4 = self.fusion_f_scale4(xf_fusion4)
        xf_fusion4 = xf_fusion4
        xf_fusion4 = self.fusion_f_scale4_2(xf_fusion4) # (1,32,20,24,28) 
        xf_fusion4 = xf_fusion4

        
        x_fusion4 = torch.cat([xm_fusion4, xf_fusion4], dim=1) # (1,64,20,24,28)

    
        # 第二个decoder
        x = self.decoder2(torch.cat([self.upsampling(x), x_fusion4], dim=1)) # (1,32,20,24,28)
        # x = self.trans_3(x)

        # ----输出1/8分辨率的变形场-----
        flow_1 = self.output_block_1(x) # (1,3,20,24,28) 
        flow_1_up = nn.functional.interpolate(flow_1, scale_factor=2,mode="trilinear")*2 # (1,3,40,48,56)


        ## ------------------------------Scale3--------------------------------------------
        # 融合多尺度特征：(40,48,56)
        xm_skip_1_fusion3 = self.pooling_4(xm_skip_1) # (1,16,40,48,56)
        xm_skip_2_fusion3 = self.pooling(xm_skip_2) # (1,32,40,48,56)
        xm_skip_3_fusion3 = xm_skip_3 # (1,32,40,48,56)
        xm_skip_4_fusion3 = nn.functional.interpolate(xm_skip_4, scale_factor=2,mode="trilinear")
        

        xm_fusion3 = torch.cat([xm_skip_1_fusion3, xm_skip_2_fusion3], dim=1)
        xm_fusion3 = torch.cat([xm_fusion3, xm_skip_3_fusion3], dim=1)
        xm_fusion3 = torch.cat([xm_fusion3, xm_skip_4_fusion3], dim=1) # (1,112,40,48,56)

        xm_fusion3 = self.fusion_m_scale3(xm_fusion3) # (1,32,40,48,56)
        xm_fusion3 = xm_fusion3
        xm_fusion3 = self.fusion_m_scale3_2(xm_fusion3) # (1,32,40,48,56)
        xm_fusion3 = xm_fusion3

        xw_fusion3 = self.transformer2(xm_fusion3, flow_1_up)


        xf_skip_1_fusion3 = self.pooling_4(xf_skip_1) # (1,16,40,48,56)
        xf_skip_2_fusion3 = self.pooling(xf_skip_2) # (1,32,40,48,56)
        xf_skip_3_fusion3 = xf_skip_3 # (1,32,40,48,56)

        xf_skip_4_fusion3 = nn.functional.interpolate(xf_skip_4, scale_factor=2,mode="trilinear")
        

        xf_fusion3 = torch.cat([xf_skip_1_fusion3, xf_skip_2_fusion3], dim=1)
        xf_fusion3 = torch.cat([xf_fusion3, xf_skip_3_fusion3], dim=1)
        xf_fusion3 = torch.cat([xf_fusion3, xf_skip_4_fusion3], dim=1) # (1,112,40,48,56)

        xf_fusion3 = self.fusion_f_scale3(xf_fusion3) # (1,32,40,48,56)
        xf_fusion3 = xf_fusion3
        xf_fusion3 = self.fusion_f_scale3_2(xf_fusion3) # (1,32,40,48,56)
        xf_fusion3 = xf_fusion3

        # concat
        x_fusion3 = torch.cat([xw_fusion3, xf_fusion3], dim=1)  # (1,64,40,48,56)

        # 第三个decoder
        x = self.decoder3(torch.cat([self.upsampling(x), x_fusion3], dim=1)) # (1,32,40,48,56)
        # x = self.trans_4(x)

        # ----输出1/4分辨率的变形场---
        delta_flow_2 = self.output_block_2(x) # (1,3,40,48,56)
        flow_2 = delta_flow_2 + flow_1_up # (1,3,40,48,56)
        flow_2_up = nn.functional.interpolate(flow_2, scale_factor=2,mode="trilinear")*2 # (1,3,80,96,112)

        ## ------------------------------Scale2--------------------------------------------
        # 融合多尺度特征：(80,96,112)
        xm_skip_1_fusion2 = self.pooling(xm_skip_1) # (1,16,80,96,112)
        xm_skip_2_fusion2 = xm_skip_2 # (1,32,80,96,112)

        xm_skip_3_fusion2 = nn.functional.interpolate(xm_skip_3, scale_factor=2,mode="trilinear")  # (1,32,80,96,112)
        
        xm_skip_4_fusion2 = nn.functional.interpolate(xm_skip_4, scale_factor=4,mode="trilinear")
        

        xm_fusion2 = torch.cat([xm_skip_1_fusion2, xm_skip_2_fusion2], dim=1)
        xm_fusion2 = torch.cat([xm_fusion2, xm_skip_3_fusion2], dim=1)
        xm_fusion2 = torch.cat([xm_fusion2, xm_skip_4_fusion2], dim=1) # (1,112,80,96,112)


        xm_fusion2 = self.fusion_m_scale2(xm_fusion2) # (1,32,80,96,112)
        xm_fusion2 = xm_fusion2
        xm_fusion2 = self.fusion_m_scale2_2(xm_fusion2) # (1,32,80,96,112)
        xm_fusion2 = xm_fusion2


        xw_fusion2 = self.transformer3(xm_fusion2, flow_2_up)

        xf_skip_1_fusion2 = self.pooling(xf_skip_1) # (1,16,80,96,112)
        xf_skip_2_fusion2 = xf_skip_2 # (1,32,80,96,112)

        xf_skip_3_fusion2 = nn.functional.interpolate(xf_skip_3, scale_factor=2,mode="trilinear")  # (1,32,80,96,112)
        
        xf_skip_4_fusion2 = nn.functional.interpolate(xf_skip_4, scale_factor=4,mode="trilinear")
        

        xf_fusion2 = torch.cat([xf_skip_1_fusion2, xf_skip_2_fusion2], dim=1)
        xf_fusion2 = torch.cat([xf_fusion2, xf_skip_3_fusion2], dim=1)
        xf_fusion2 = torch.cat([xf_fusion2, xf_skip_4_fusion2], dim=1) # (1,112,80,96,112)

        xf_fusion2 = self.fusion_f_scale2(xf_fusion2) # (1,32,80,96,112)
        xf_fusion2 = xf_fusion2
        xf_fusion2 = self.fusion_f_scale2_2(xf_fusion2) # (1,32,80,96,112)
        xf_fusion2 = xf_fusion2

        # concat
        x_fusion2 = torch.cat([xw_fusion2, xf_fusion2], dim=1)  # (1,32,80,96,112)


        # 第四个decoder
        x = self.decoder4(torch.cat([self.upsampling(x), x_fusion2], dim=1)) # (1,32,80,96,112)
        # x = self.trans_5(x)
        # print(x.shape)

        # ----输出1/2分辨率的变形场-------
        delta_flow_3 = self.output_block_3(x) # (1,3,80,96,112)
        flow_3 = delta_flow_3 + flow_2_up # (1,3,80,96,112)
        flow_3_up = nn.functional.interpolate(flow_3, scale_factor=2,mode="trilinear")*2 # (1,3,160,112,224)

        # ------------------------------Scale1--------------------------------------------
        # # 对moving feature进行warp得到moved feature
        xw_skip_1 = self.transformer4(xm_skip_1, flow_3_up)
        # concat
        x_skip_1 = torch.cat([xw_skip_1, xf_skip_1], dim=1)  # (1,32,160,192,224)
        # print(x_skip_1.shape)


        # 第四个decoder
        x = self.decoder5(torch.cat([self.upsampling(x), x_skip_1], dim=1)) # (1,32,160,192,224)

        # ----------------------输出最终的变形场------------------------------------------
        # output block
        x = self.output_block(x) # (1,16,160,192,224)
        # 生成flow场
        delta_flow_final = self.flow(x) # (1,3,160,192,224)
        flow_final = delta_flow_final + flow_3_up
            

        return flow_1,flow_2,flow_3,flow_final,delta_flow_2,delta_flow_3,delta_flow_final,features_xf,features_xm

# #(with FFM) 直接融合 + 俩卷积 (剪枝后的网络) encoder和FFM 都×3
class dual_pyramid_VxmDense_FFM_large_adaptive_val(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    自己写的, 两个权重共享的编码器来各自提取img_concat_wavelet的特征,
    然后特征融合, 再送入同一个解码器
    在multi-vxmdense的基础上改的

    配准网络本身配准的是输入model的source和target的尺寸, 而SPT则是new_shape的尺寸
    特别需要注意的就是输入的source和target的尺寸是否能完成四次下采样, 不能的话要注意调整网络下采样的次数
    最后生成的flow可以通过self.flow里面的卷积步长stride来改变尺寸(下采样等等)
    """

    @store_config_args
    def __init__(self,
                 inshape=(160,192,224),
                 list_num=[9,13,2,3,64,64,64,32,45,32,37]):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)  CT:list_num=[9,13,2,3,64,64,64,32,32,32,32]  CT:list_num=[9,13,2,3,64,64,64,32,45,32,37]
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this 
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. 
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2. 
                Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)

        # # print(new_shape)
        # self.transformer1 = layers.SpatialTransformer((16,16,16))
        # self.transformer2 = layers.SpatialTransformer((32,32,32))
        # self.transformer3 = layers.SpatialTransformer((64,64,64))
        # self.transformer4 = layers.SpatialTransformer((128,128,128))

        # self.transformer1 = layers.SpatialTransformer((20,24,28))
        # self.transformer2 = layers.SpatialTransformer((40,48,56))
        # self.transformer3 = layers.SpatialTransformer((80,96,112))
        # self.transformer4 = layers.SpatialTransformer((160,192,224))

        self.transformer1 = layers.SpatialTransformer(tuple(d // 8 for d in inshape))
        self.transformer2 = layers.SpatialTransformer(tuple(d // 4 for d in inshape))
        self.transformer3 = layers.SpatialTransformer(tuple(d // 2 for d in inshape))
        self.transformer4 = layers.SpatialTransformer(inshape)
        
        # self.transformer1 = layers.SpatialTransformer((24,20,20))
        # self.transformer2 = layers.SpatialTransformer((48,40,40))
        # self.transformer3 = layers.SpatialTransformer((96,80,80))
        # self.transformer4 = layers.SpatialTransformer((192,160,160))


        # self.input_model = input_model
        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = MaxPooling(2)
        self.pooling_4 =MaxPooling(4)
        self.pooling_8 =MaxPooling(8)

        self.upsampling = nn.Upsample(scale_factor=2, mode='trilinear') 


        self.encoder1_m = ConvBlock(3,1,list_num[0])  # 权重共享的编码器, 提取moving
        self.encoder2_m = ConvBlock(3,list_num[0],list_num[1])
        self.encoder3_m = ConvBlock(3,list_num[1],list_num[2])
        self.encoder4_m = ConvBlock(3,list_num[2],list_num[3])
        self.encoder5_m = ConvBlock(3,list_num[3],list_num[4])

        
        self.encoder1_f = self.encoder1_m  # 权重共享的编码器, 提取fixed
        self.encoder2_f = self.encoder2_m
        self.encoder3_f = self.encoder3_m
        self.encoder4_f = self.encoder4_m
        self.encoder5_f = self.encoder5_m

        
        self.decoder1 = ConvBlock(ndims,list_num[4]+list_num[4],32)
        self.decoder2 = ConvBlock(ndims,list_num[6]+list_num[6]+32,32)
        self.decoder3 = ConvBlock(ndims,list_num[8]+list_num[8]+32,32)
        self.decoder4 = ConvBlock(ndims,list_num[10]+list_num[10]+32,16)
        self.decoder5 = ConvBlock(ndims,list_num[0]+list_num[0]+16,16)

        self.output_block = nn.Sequential(ConvBlock(ndims,16,16),ConvBlock(ndims,16,16))
        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(16, ndims, kernel_size=3, padding=1)

        # # init flow layer with small weights and bias
        # self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        # self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        self.output_block_0 = nn.Sequential(ConvBlock(ndims,32,16),ConvBlock(ndims,16,3))
        self.output_block_1 = nn.Sequential(ConvBlock(ndims,32,16),ConvBlock(ndims,16,3))
        self.output_block_2 = nn.Sequential(ConvBlock(ndims,32,16),ConvBlock(ndims,16,3))
        self.output_block_3 = nn.Sequential(ConvBlock(ndims,16,16),ConvBlock(ndims,16,3))

        # 整合特征
        # # ------------------------------Scale4--------------------------------------------
        self.fusion_f_scale4 = ConvBlock(ndims,list_num[0]+list_num[1]+list_num[2]+list_num[3],list_num[5])
        self.fusion_f_scale4_2 = ConvBlock(ndims,list_num[5],list_num[6])


        self.fusion_m_scale4 = self.fusion_f_scale4
        self.fusion_m_scale4_2 = self.fusion_f_scale4_2

        # # ------------------------------Scale3--------------------------------------------
        self.fusion_f_scale3 = ConvBlock(ndims,list_num[0]+list_num[1]+list_num[2]+list_num[3],list_num[7])
        self.fusion_f_scale3_2 = ConvBlock(ndims,list_num[7],list_num[8])


        self.fusion_m_scale3 = self.fusion_f_scale3
        self.fusion_m_scale3_2 = self.fusion_f_scale3_2

        # # ------------------------------Scale2--------------------------------------------
        self.fusion_f_scale2 = ConvBlock(ndims,list_num[0]+list_num[1]+list_num[2]+list_num[3],list_num[9])
        self.fusion_f_scale2_2 = ConvBlock(ndims,list_num[9],list_num[10])


        self.fusion_m_scale2 = self.fusion_f_scale2
        self.fusion_m_scale2_2 = self.fusion_f_scale2_2


    
    def forward(self, source, target):
        '''
        Parameters:
            source: Source image tensor.F (moving)
            target: Target image tensor.  (fixed)
            registration: Return transformed image and flow. Default is False.

        ''' 
        xm = source
        xf = target
        xm_skip_1 = self.encoder1_m(xm) # (1,48,160,192,224)
        # xm_skip_1 = xm_skip_1
        xf_skip_1 = self.encoder1_f(xf) # (1,48,160,192,224)
        # xf_skip_1 = xf_skip_1

        # moving encoder
        #下采样到1/2,第二个encoder
        xm = self.pooling(xm_skip_1) # (1,16,80,96,112)
        xm_skip_2 = self.encoder2_m(xm) # (1,32,80,96,112)
        # xm_skip_2 = xm_skip_2
        # 下采样到1/4,第三个encoder
        xm = self.pooling(xm_skip_2) # (1,32,40,48,56)
        xm_skip_3 = self.encoder3_m(xm) # (1,32,40,48,56)
        # xm_skip_3 = xm_skip_3
        # 下采样到1/8,第四个encoder
        xm = self.pooling(xm_skip_3)  # (1,32,20,24,28)
        xm_skip_4 = self.encoder4_m(xm) # (1,32,20,24,28)
        # xm_skip_4 = xm_skip_4
        # 下采样到1/16
        xm = self.pooling(xm_skip_4) # (1,32,10,12,14)
        xm_skip_5 = self.encoder5_m(xm) # (1,32,10,12,14)
        # xm_skip_5 = xm_skip_5

        features_xm = [xm_skip_1,xm_skip_2,xm_skip_3,xm_skip_4,xm_skip_5]


        # fixed encoder
        #下采样到1/2,第二个encoder
        xf = self.pooling(xf_skip_1)
        xf_skip_2 = self.encoder2_f(xf)
        # xf_skip_2 = xf_skip_2
        # 下采样到1/4,第三个encoder
        xf = self.pooling(xf_skip_2)
        xf_skip_3 = self.encoder3_f(xf)
        # xf_skip_3 = xf_skip_3
        # 下采样到1/8,第四个encoder
        xf = self.pooling(xf_skip_3)  
        xf_skip_4 = self.encoder4_f(xf)
        # xf_skip_4 = xf_skip_4
        # 下采样到1/16
        xf = self.pooling(xf_skip_4)
        xf_skip_5 = self.encoder5_f(xf) # (1,32,10,12,14)
        # xf_skip_5 = xf_skip_5

        features_xf = [xf_skip_1,xf_skip_2,xf_skip_3,xf_skip_4,xf_skip_5]

        #-------------------------------Scale5--------------------------------
        # 第一个decoder
        x = torch.cat([xm_skip_5, xf_skip_5], dim=1) # (1,64,10,12,14)
        x = self.decoder1(x) # (1,32,10,12,14)


        # ------------------------------Scale4--------------------------------------------
        # 融合多尺度特征：(20,24,28)
        xm_skip_1_fusion4 = self.pooling_8(xm_skip_1) # (1,16,20,24,28)
        xm_skip_2_fusion4 = self.pooling_4(xm_skip_2) # (1,32,20,24,28)
        xm_skip_3_fusion4 = self.pooling(xm_skip_3) # (1,32,20,24,28)
        xm_skip_4_fusion4 = xm_skip_4 # (1,32,20,24,28)

        xm_fusion4 = torch.cat([xm_skip_1_fusion4, xm_skip_2_fusion4], dim=1)
        xm_fusion4 = torch.cat([xm_fusion4, xm_skip_3_fusion4], dim=1)
        xm_fusion4 = torch.cat([xm_fusion4, xm_skip_4_fusion4], dim=1)

        xm_fusion4 = self.fusion_m_scale4(xm_fusion4) # (1,32,20,24,28)
        xm_fusion4 = xm_fusion4
        xm_fusion4 = self.fusion_m_scale4_2(xm_fusion4) # (1,32,20,24,28)
        xm_fusion4 = xm_fusion4
    

        xf_skip_1_fusion4 = self.pooling_8(xf_skip_1) # (1,16,20,24,28)
        xf_skip_2_fusion4 = self.pooling_4(xf_skip_2) # (1,32,20,24,28)
        xf_skip_3_fusion4 = self.pooling(xf_skip_3) # (1,32,20,24,28)
        xf_skip_4_fusion4 = xf_skip_4 # (1,32,20,24,28)

        xf_fusion4 = torch.cat([xf_skip_1_fusion4, xf_skip_2_fusion4], dim=1) 
        xf_fusion4 = torch.cat([xf_fusion4, xf_skip_3_fusion4], dim=1)
        xf_fusion4 = torch.cat([xf_fusion4, xf_skip_4_fusion4], dim=1)
   
        xf_fusion4 = self.fusion_f_scale4(xf_fusion4)
        xf_fusion4 = xf_fusion4
        xf_fusion4 = self.fusion_f_scale4_2(xf_fusion4) # (1,32,20,24,28) 
        xf_fusion4 = xf_fusion4

        
        x_fusion4 = torch.cat([xm_fusion4, xf_fusion4], dim=1) # (1,64,20,24,28)

    
        # 第二个decoder
        x = self.decoder2(torch.cat([self.upsampling(x), x_fusion4], dim=1)) # (1,32,20,24,28)
        # x = self.trans_3(x)

        # ----输出1/8分辨率的变形场-----
        flow_1 = self.output_block_1(x) # (1,3,20,24,28) 
        flow_1_up = nn.functional.interpolate(flow_1, scale_factor=2,mode="trilinear")*2 # (1,3,40,48,56)


        ## ------------------------------Scale3--------------------------------------------
        # 融合多尺度特征：(40,48,56)
        xm_skip_1_fusion3 = self.pooling_4(xm_skip_1) # (1,16,40,48,56)
        xm_skip_2_fusion3 = self.pooling(xm_skip_2) # (1,32,40,48,56)
        xm_skip_3_fusion3 = xm_skip_3 # (1,32,40,48,56)
        xm_skip_4_fusion3 = nn.functional.interpolate(xm_skip_4, scale_factor=2,mode="trilinear")
        

        xm_fusion3 = torch.cat([xm_skip_1_fusion3, xm_skip_2_fusion3], dim=1)
        xm_fusion3 = torch.cat([xm_fusion3, xm_skip_3_fusion3], dim=1)
        xm_fusion3 = torch.cat([xm_fusion3, xm_skip_4_fusion3], dim=1) # (1,112,40,48,56)

        xm_fusion3 = self.fusion_m_scale3(xm_fusion3) # (1,32,40,48,56)
        xm_fusion3 = xm_fusion3
        xm_fusion3 = self.fusion_m_scale3_2(xm_fusion3) # (1,32,40,48,56)
        xm_fusion3 = xm_fusion3

        xw_fusion3 = self.transformer2(xm_fusion3, flow_1_up)


        xf_skip_1_fusion3 = self.pooling_4(xf_skip_1) # (1,16,40,48,56)
        xf_skip_2_fusion3 = self.pooling(xf_skip_2) # (1,32,40,48,56)
        xf_skip_3_fusion3 = xf_skip_3 # (1,32,40,48,56)

        xf_skip_4_fusion3 = nn.functional.interpolate(xf_skip_4, scale_factor=2,mode="trilinear")
        

        xf_fusion3 = torch.cat([xf_skip_1_fusion3, xf_skip_2_fusion3], dim=1)
        xf_fusion3 = torch.cat([xf_fusion3, xf_skip_3_fusion3], dim=1)
        xf_fusion3 = torch.cat([xf_fusion3, xf_skip_4_fusion3], dim=1) # (1,112,40,48,56)

        xf_fusion3 = self.fusion_f_scale3(xf_fusion3) # (1,32,40,48,56)
        xf_fusion3 = xf_fusion3
        xf_fusion3 = self.fusion_f_scale3_2(xf_fusion3) # (1,32,40,48,56)
        xf_fusion3 = xf_fusion3

        # concat
        x_fusion3 = torch.cat([xw_fusion3, xf_fusion3], dim=1)  # (1,64,40,48,56)

        # 第三个decoder
        x = self.decoder3(torch.cat([self.upsampling(x), x_fusion3], dim=1)) # (1,32,40,48,56)
        # x = self.trans_4(x)

        # ----输出1/4分辨率的变形场---
        delta_flow_2 = self.output_block_2(x) # (1,3,40,48,56)
        flow_2 = delta_flow_2 + flow_1_up # (1,3,40,48,56)
        flow_2_up = nn.functional.interpolate(flow_2, scale_factor=2,mode="trilinear")*2 # (1,3,80,96,112)

        ## ------------------------------Scale2--------------------------------------------
        # 融合多尺度特征：(80,96,112)
        xm_skip_1_fusion2 = self.pooling(xm_skip_1) # (1,16,80,96,112)
        xm_skip_2_fusion2 = xm_skip_2 # (1,32,80,96,112)

        xm_skip_3_fusion2 = nn.functional.interpolate(xm_skip_3, scale_factor=2,mode="trilinear")  # (1,32,80,96,112)
        
        xm_skip_4_fusion2 = nn.functional.interpolate(xm_skip_4, scale_factor=4,mode="trilinear")
        

        xm_fusion2 = torch.cat([xm_skip_1_fusion2, xm_skip_2_fusion2], dim=1)
        xm_fusion2 = torch.cat([xm_fusion2, xm_skip_3_fusion2], dim=1)
        xm_fusion2 = torch.cat([xm_fusion2, xm_skip_4_fusion2], dim=1) # (1,112,80,96,112)


        xm_fusion2 = self.fusion_m_scale2(xm_fusion2) # (1,32,80,96,112)
        xm_fusion2 = xm_fusion2
        xm_fusion2 = self.fusion_m_scale2_2(xm_fusion2) # (1,32,80,96,112)
        xm_fusion2 = xm_fusion2


        xw_fusion2 = self.transformer3(xm_fusion2, flow_2_up)

        xf_skip_1_fusion2 = self.pooling(xf_skip_1) # (1,16,80,96,112)
        xf_skip_2_fusion2 = xf_skip_2 # (1,32,80,96,112)

        xf_skip_3_fusion2 = nn.functional.interpolate(xf_skip_3, scale_factor=2,mode="trilinear")  # (1,32,80,96,112)
        
        xf_skip_4_fusion2 = nn.functional.interpolate(xf_skip_4, scale_factor=4,mode="trilinear")
        

        xf_fusion2 = torch.cat([xf_skip_1_fusion2, xf_skip_2_fusion2], dim=1)
        xf_fusion2 = torch.cat([xf_fusion2, xf_skip_3_fusion2], dim=1)
        xf_fusion2 = torch.cat([xf_fusion2, xf_skip_4_fusion2], dim=1) # (1,112,80,96,112)

        xf_fusion2 = self.fusion_f_scale2(xf_fusion2) # (1,32,80,96,112)
        xf_fusion2 = xf_fusion2
        xf_fusion2 = self.fusion_f_scale2_2(xf_fusion2) # (1,32,80,96,112)
        xf_fusion2 = xf_fusion2

        # concat
        x_fusion2 = torch.cat([xw_fusion2, xf_fusion2], dim=1)  # (1,32,80,96,112)


        # 第四个decoder
        x = self.decoder4(torch.cat([self.upsampling(x), x_fusion2], dim=1)) # (1,32,80,96,112)
        # x = self.trans_5(x)
        # print(x.shape)

        # ----输出1/2分辨率的变形场-------
        delta_flow_3 = self.output_block_3(x) # (1,3,80,96,112)
        flow_3 = delta_flow_3 + flow_2_up # (1,3,80,96,112)
        flow_3_up = nn.functional.interpolate(flow_3, scale_factor=2,mode="trilinear")*2 # (1,3,160,112,224)

        # ------------------------------Scale1--------------------------------------------
        # # 对moving feature进行warp得到moved feature
        xw_skip_1 = self.transformer4(xm_skip_1, flow_3_up)
        # concat
        x_skip_1 = torch.cat([xw_skip_1, xf_skip_1], dim=1)  # (1,32,160,192,224)
        # print(x_skip_1.shape)


        # 第四个decoder
        x = self.decoder5(torch.cat([self.upsampling(x), x_skip_1], dim=1)) # (1,32,160,192,224)

        # ----------------------输出最终的变形场------------------------------------------
        # output block
        x = self.output_block(x) # (1,16,160,192,224)
        # 生成flow场
        delta_flow_final = self.flow(x) # (1,3,160,192,224)
        flow_final = delta_flow_final + flow_3_up
            

        return flow_1,flow_2,flow_3,flow_final,delta_flow_2,delta_flow_3,delta_flow_final,features_xf,features_xm

# #(with FFM) 直接融合 + 俩卷积 (剪枝后的网络) encoder和FFM 都×3
class dual_pyramid_VxmDense_FFM_normal_adaptive_val(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    自己写的, 两个权重共享的编码器来各自提取img_concat_wavelet的特征,
    然后特征融合, 再送入同一个解码器
    在multi-vxmdense的基础上改的

    配准网络本身配准的是输入model的source和target的尺寸, 而SPT则是new_shape的尺寸
    特别需要注意的就是输入的source和target的尺寸是否能完成四次下采样, 不能的话要注意调整网络下采样的次数
    最后生成的flow可以通过self.flow里面的卷积步长stride来改变尺寸(下采样等等)
    """

    @store_config_args
    def __init__(self,
                 inshape=(128,128,128),
                 list_num=[4,6,3,2,32,32,32,22,17,16,16]):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)[4,9,2,2,32,32,32,20,16,16,16]  CT:list_num=[4,6,3,2,32,32,32,22,17,16,16]
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this 
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. 
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2. 
                Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)

        # # print(new_shape)
        # self.transformer1 = layers.SpatialTransformer((16,16,16))
        # self.transformer2 = layers.SpatialTransformer((32,32,32))
        # self.transformer3 = layers.SpatialTransformer((64,64,64))
        # self.transformer4 = layers.SpatialTransformer((128,128,128))

        # self.transformer1 = layers.SpatialTransformer((20,24,28))
        # self.transformer2 = layers.SpatialTransformer((40,48,56))
        # self.transformer3 = layers.SpatialTransformer((80,96,112))
        # self.transformer4 = layers.SpatialTransformer((160,192,224))

        self.transformer1 = layers.SpatialTransformer(tuple(d // 8 for d in inshape))
        self.transformer2 = layers.SpatialTransformer(tuple(d // 4 for d in inshape))
        self.transformer3 = layers.SpatialTransformer(tuple(d // 2 for d in inshape))
        self.transformer4 = layers.SpatialTransformer(inshape)

        # self.transformer1 = layers.SpatialTransformer((24,20,20))
        # self.transformer2 = layers.SpatialTransformer((48,40,40))
        # self.transformer3 = layers.SpatialTransformer((96,80,80))
        # self.transformer4 = layers.SpatialTransformer((192,160,160))


        # self.input_model = input_model
        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = MaxPooling(2)
        self.pooling_4 =MaxPooling(4)
        self.pooling_8 =MaxPooling(8)

        self.upsampling = nn.Upsample(scale_factor=2, mode='trilinear') 


        self.encoder1_m = ConvBlock(3,1,list_num[0])  # 权重共享的编码器, 提取moving
        self.encoder2_m = ConvBlock(3,list_num[0],list_num[1])
        self.encoder3_m = ConvBlock(3,list_num[1],list_num[2])
        self.encoder4_m = ConvBlock(3,list_num[2],list_num[3])
        self.encoder5_m = ConvBlock(3,list_num[3],list_num[4])

        
        self.encoder1_f = self.encoder1_m  # 权重共享的编码器, 提取fixed
        self.encoder2_f = self.encoder2_m
        self.encoder3_f = self.encoder3_m
        self.encoder4_f = self.encoder4_m
        self.encoder5_f = self.encoder5_m

        
        self.decoder1 = ConvBlock(ndims,list_num[4]+list_num[4],32)
        self.decoder2 = ConvBlock(ndims,list_num[6]+list_num[6]+32,32)
        self.decoder3 = ConvBlock(ndims,list_num[8]+list_num[8]+32,32)
        self.decoder4 = ConvBlock(ndims,list_num[10]+list_num[10]+32,16)
        self.decoder5 = ConvBlock(ndims,list_num[0]+list_num[0]+16,16)

        self.output_block = nn.Sequential(ConvBlock(ndims,16,16),ConvBlock(ndims,16,16))
        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(16, ndims, kernel_size=3, padding=1)

        # # init flow layer with small weights and bias
        # self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        # self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        self.output_block_0 = nn.Sequential(ConvBlock(ndims,32,16),ConvBlock(ndims,16,3))
        self.output_block_1 = nn.Sequential(ConvBlock(ndims,32,16),ConvBlock(ndims,16,3))
        self.output_block_2 = nn.Sequential(ConvBlock(ndims,32,16),ConvBlock(ndims,16,3))
        self.output_block_3 = nn.Sequential(ConvBlock(ndims,16,16),ConvBlock(ndims,16,3))

        # 整合特征
        # # ------------------------------Scale4--------------------------------------------
        self.fusion_f_scale4 = ConvBlock(ndims,list_num[0]+list_num[1]+list_num[2]+list_num[3],list_num[5])
        self.fusion_f_scale4_2 = ConvBlock(ndims,list_num[5],list_num[6])


        self.fusion_m_scale4 = self.fusion_f_scale4
        self.fusion_m_scale4_2 = self.fusion_f_scale4_2

        # # ------------------------------Scale3--------------------------------------------
        self.fusion_f_scale3 = ConvBlock(ndims,list_num[0]+list_num[1]+list_num[2]+list_num[3],list_num[7])
        self.fusion_f_scale3_2 = ConvBlock(ndims,list_num[7],list_num[8])


        self.fusion_m_scale3 = self.fusion_f_scale3
        self.fusion_m_scale3_2 = self.fusion_f_scale3_2

        # # ------------------------------Scale2--------------------------------------------
        self.fusion_f_scale2 = ConvBlock(ndims,list_num[0]+list_num[1]+list_num[2]+list_num[3],list_num[9])
        self.fusion_f_scale2_2 = ConvBlock(ndims,list_num[9],list_num[10])


        self.fusion_m_scale2 = self.fusion_f_scale2
        self.fusion_m_scale2_2 = self.fusion_f_scale2_2


    
    def forward(self, source, target):
        '''
        Parameters:
            source: Source image tensor.F (moving)
            target: Target image tensor.  (fixed)
            registration: Return transformed image and flow. Default is False.

        ''' 
        xm = source
        xf = target
        xm_skip_1 = self.encoder1_m(xm) # (1,48,160,192,224)
        # xm_skip_1 = xm_skip_1
        xf_skip_1 = self.encoder1_f(xf) # (1,48,160,192,224)
        # xf_skip_1 = xf_skip_1

        # moving encoder
        #下采样到1/2,第二个encoder
        xm = self.pooling(xm_skip_1) # (1,16,80,96,112)
        xm_skip_2 = self.encoder2_m(xm) # (1,32,80,96,112)
        # xm_skip_2 = xm_skip_2
        # 下采样到1/4,第三个encoder
        xm = self.pooling(xm_skip_2) # (1,32,40,48,56)
        xm_skip_3 = self.encoder3_m(xm) # (1,32,40,48,56)
        # xm_skip_3 = xm_skip_3
        # 下采样到1/8,第四个encoder
        xm = self.pooling(xm_skip_3)  # (1,32,20,24,28)
        xm_skip_4 = self.encoder4_m(xm) # (1,32,20,24,28)
        # xm_skip_4 = xm_skip_4
        # 下采样到1/16
        xm = self.pooling(xm_skip_4) # (1,32,10,12,14)
        xm_skip_5 = self.encoder5_m(xm) # (1,32,10,12,14)
        # xm_skip_5 = xm_skip_5

        features = [xm_skip_1,xm_skip_2,xm_skip_3,xm_skip_4,xm_skip_5]


        # fixed encoder
        #下采样到1/2,第二个encoder
        xf = self.pooling(xf_skip_1)
        xf_skip_2 = self.encoder2_f(xf)
        # xf_skip_2 = xf_skip_2
        # 下采样到1/4,第三个encoder
        xf = self.pooling(xf_skip_2)
        xf_skip_3 = self.encoder3_f(xf)
        # xf_skip_3 = xf_skip_3
        # 下采样到1/8,第四个encoder
        xf = self.pooling(xf_skip_3)  
        xf_skip_4 = self.encoder4_f(xf)
        # xf_skip_4 = xf_skip_4
        # 下采样到1/16
        xf = self.pooling(xf_skip_4)
        xf_skip_5 = self.encoder5_f(xf) # (1,32,10,12,14)
        # xf_skip_5 = xf_skip_5

        #-------------------------------Scale5--------------------------------
        # 第一个decoder
        x = torch.cat([xm_skip_5, xf_skip_5], dim=1) # (1,64,10,12,14)
        x = self.decoder1(x) # (1,32,10,12,14)


        # ------------------------------Scale4--------------------------------------------
        # 融合多尺度特征：(20,24,28)
        xm_skip_1_fusion4 = self.pooling_8(xm_skip_1) # (1,16,20,24,28)
        xm_skip_2_fusion4 = self.pooling_4(xm_skip_2) # (1,32,20,24,28)
        xm_skip_3_fusion4 = self.pooling(xm_skip_3) # (1,32,20,24,28)
        xm_skip_4_fusion4 = xm_skip_4 # (1,32,20,24,28)

        xm_fusion4 = torch.cat([xm_skip_1_fusion4, xm_skip_2_fusion4], dim=1)
        xm_fusion4 = torch.cat([xm_fusion4, xm_skip_3_fusion4], dim=1)
        xm_fusion4 = torch.cat([xm_fusion4, xm_skip_4_fusion4], dim=1)

        xm_fusion4 = self.fusion_m_scale4(xm_fusion4) # (1,32,20,24,28)
        xm_fusion4 = xm_fusion4
        xm_fusion4 = self.fusion_m_scale4_2(xm_fusion4) # (1,32,20,24,28)
        xm_fusion4 = xm_fusion4
    

        xf_skip_1_fusion4 = self.pooling_8(xf_skip_1) # (1,16,20,24,28)
        xf_skip_2_fusion4 = self.pooling_4(xf_skip_2) # (1,32,20,24,28)
        xf_skip_3_fusion4 = self.pooling(xf_skip_3) # (1,32,20,24,28)
        xf_skip_4_fusion4 = xf_skip_4 # (1,32,20,24,28)

        xf_fusion4 = torch.cat([xf_skip_1_fusion4, xf_skip_2_fusion4], dim=1) 
        xf_fusion4 = torch.cat([xf_fusion4, xf_skip_3_fusion4], dim=1)
        xf_fusion4 = torch.cat([xf_fusion4, xf_skip_4_fusion4], dim=1)
   
        xf_fusion4 = self.fusion_f_scale4(xf_fusion4)
        xf_fusion4 = xf_fusion4
        xf_fusion4 = self.fusion_f_scale4_2(xf_fusion4) # (1,32,20,24,28) 
        xf_fusion4 = xf_fusion4

        
        x_fusion4 = torch.cat([xm_fusion4, xf_fusion4], dim=1) # (1,64,20,24,28)

    
        # 第二个decoder
        x = self.decoder2(torch.cat([self.upsampling(x), x_fusion4], dim=1)) # (1,32,20,24,28)
        # x = self.trans_3(x)

        # ----输出1/8分辨率的变形场-----
        flow_1 = self.output_block_1(x) # (1,3,20,24,28) 
        flow_1_up = nn.functional.interpolate(flow_1, scale_factor=2,mode="trilinear")*2 # (1,3,40,48,56)


        ## ------------------------------Scale3--------------------------------------------
        # 融合多尺度特征：(40,48,56)
        xm_skip_1_fusion3 = self.pooling_4(xm_skip_1) # (1,16,40,48,56)
        xm_skip_2_fusion3 = self.pooling(xm_skip_2) # (1,32,40,48,56)
        xm_skip_3_fusion3 = xm_skip_3 # (1,32,40,48,56)
        xm_skip_4_fusion3 = nn.functional.interpolate(xm_skip_4, scale_factor=2,mode="trilinear")
        

        xm_fusion3 = torch.cat([xm_skip_1_fusion3, xm_skip_2_fusion3], dim=1)
        xm_fusion3 = torch.cat([xm_fusion3, xm_skip_3_fusion3], dim=1)
        xm_fusion3 = torch.cat([xm_fusion3, xm_skip_4_fusion3], dim=1) # (1,112,40,48,56)

        xm_fusion3 = self.fusion_m_scale3(xm_fusion3) # (1,32,40,48,56)
        xm_fusion3 = xm_fusion3
        xm_fusion3 = self.fusion_m_scale3_2(xm_fusion3) # (1,32,40,48,56)
        xm_fusion3 = xm_fusion3

        xw_fusion3 = self.transformer2(xm_fusion3, flow_1_up)


        xf_skip_1_fusion3 = self.pooling_4(xf_skip_1) # (1,16,40,48,56)
        xf_skip_2_fusion3 = self.pooling(xf_skip_2) # (1,32,40,48,56)
        xf_skip_3_fusion3 = xf_skip_3 # (1,32,40,48,56)

        xf_skip_4_fusion3 = nn.functional.interpolate(xf_skip_4, scale_factor=2,mode="trilinear")
        

        xf_fusion3 = torch.cat([xf_skip_1_fusion3, xf_skip_2_fusion3], dim=1)
        xf_fusion3 = torch.cat([xf_fusion3, xf_skip_3_fusion3], dim=1)
        xf_fusion3 = torch.cat([xf_fusion3, xf_skip_4_fusion3], dim=1) # (1,112,40,48,56)

        xf_fusion3 = self.fusion_f_scale3(xf_fusion3) # (1,32,40,48,56)
        xf_fusion3 = xf_fusion3
        xf_fusion3 = self.fusion_f_scale3_2(xf_fusion3) # (1,32,40,48,56)
        xf_fusion3 = xf_fusion3

        # concat
        x_fusion3 = torch.cat([xw_fusion3, xf_fusion3], dim=1)  # (1,64,40,48,56)

        # 第三个decoder
        x = self.decoder3(torch.cat([self.upsampling(x), x_fusion3], dim=1)) # (1,32,40,48,56)
        # x = self.trans_4(x)

        # ----输出1/4分辨率的变形场---
        delta_flow_2 = self.output_block_2(x) # (1,3,40,48,56)
        flow_2 = delta_flow_2 + flow_1_up # (1,3,40,48,56)
        flow_2_up = nn.functional.interpolate(flow_2, scale_factor=2,mode="trilinear")*2 # (1,3,80,96,112)

        ## ------------------------------Scale2--------------------------------------------
        # 融合多尺度特征：(80,96,112)
        xm_skip_1_fusion2 = self.pooling(xm_skip_1) # (1,16,80,96,112)
        xm_skip_2_fusion2 = xm_skip_2 # (1,32,80,96,112)

        xm_skip_3_fusion2 = nn.functional.interpolate(xm_skip_3, scale_factor=2,mode="trilinear")  # (1,32,80,96,112)
        
        xm_skip_4_fusion2 = nn.functional.interpolate(xm_skip_4, scale_factor=4,mode="trilinear")
        

        xm_fusion2 = torch.cat([xm_skip_1_fusion2, xm_skip_2_fusion2], dim=1)
        xm_fusion2 = torch.cat([xm_fusion2, xm_skip_3_fusion2], dim=1)
        xm_fusion2 = torch.cat([xm_fusion2, xm_skip_4_fusion2], dim=1) # (1,112,80,96,112)


        xm_fusion2 = self.fusion_m_scale2(xm_fusion2) # (1,32,80,96,112)
        xm_fusion2 = xm_fusion2
        xm_fusion2 = self.fusion_m_scale2_2(xm_fusion2) # (1,32,80,96,112)
        xm_fusion2 = xm_fusion2


        xw_fusion2 = self.transformer3(xm_fusion2, flow_2_up)

        xf_skip_1_fusion2 = self.pooling(xf_skip_1) # (1,16,80,96,112)
        xf_skip_2_fusion2 = xf_skip_2 # (1,32,80,96,112)

        xf_skip_3_fusion2 = nn.functional.interpolate(xf_skip_3, scale_factor=2,mode="trilinear")  # (1,32,80,96,112)
        
        xf_skip_4_fusion2 = nn.functional.interpolate(xf_skip_4, scale_factor=4,mode="trilinear")
        

        xf_fusion2 = torch.cat([xf_skip_1_fusion2, xf_skip_2_fusion2], dim=1)
        xf_fusion2 = torch.cat([xf_fusion2, xf_skip_3_fusion2], dim=1)
        xf_fusion2 = torch.cat([xf_fusion2, xf_skip_4_fusion2], dim=1) # (1,112,80,96,112)

        xf_fusion2 = self.fusion_f_scale2(xf_fusion2) # (1,32,80,96,112)
        xf_fusion2 = xf_fusion2
        xf_fusion2 = self.fusion_f_scale2_2(xf_fusion2) # (1,32,80,96,112)
        xf_fusion2 = xf_fusion2

        # concat
        x_fusion2 = torch.cat([xw_fusion2, xf_fusion2], dim=1)  # (1,32,80,96,112)


        # 第四个decoder
        x = self.decoder4(torch.cat([self.upsampling(x), x_fusion2], dim=1)) # (1,32,80,96,112)
        # x = self.trans_5(x)
        # print(x.shape)

        # ----输出1/2分辨率的变形场-------
        delta_flow_3 = self.output_block_3(x) # (1,3,80,96,112)
        flow_3 = delta_flow_3 + flow_2_up # (1,3,80,96,112)
        flow_3_up = nn.functional.interpolate(flow_3, scale_factor=2,mode="trilinear")*2 # (1,3,160,112,224)

        # ------------------------------Scale1--------------------------------------------
        # # 对moving feature进行warp得到moved feature
        xw_skip_1 = self.transformer4(xm_skip_1, flow_3_up)
        # concat
        x_skip_1 = torch.cat([xw_skip_1, xf_skip_1], dim=1)  # (1,32,160,192,224)
        # print(x_skip_1.shape)


        # 第四个decoder
        x = self.decoder5(torch.cat([self.upsampling(x), x_skip_1], dim=1)) # (1,32,160,192,224)

        # ----------------------输出最终的变形场------------------------------------------
        # output block
        x = self.output_block(x) # (1,16,160,192,224)
        # 生成flow场
        delta_flow_final = self.flow(x) # (1,3,160,192,224)
        flow_final = delta_flow_final + flow_3_up
            

        return flow_1,flow_2,flow_3,flow_final,delta_flow_2,delta_flow_3,delta_flow_final

# #(with FFM) 直接融合 + 俩卷积 (剪枝后的网络) encoder和FFM 都×3
class dual_pyramid_VxmDense_Trans_FFM_normal_adaptive_val(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    自己写的, 两个权重共享的编码器来各自提取img_concat_wavelet的特征,
    然后特征融合, 再送入同一个解码器
    在multi-vxmdense的基础上改的

    配准网络本身配准的是输入model的source和target的尺寸, 而SPT则是new_shape的尺寸
    特别需要注意的就是输入的source和target的尺寸是否能完成四次下采样, 不能的话要注意调整网络下采样的次数
    最后生成的flow可以通过self.flow里面的卷积步长stride来改变尺寸(下采样等等)
    """

    @store_config_args
    def __init__(self,
                 inshape=(160,192,224),
                 list_num=[5,8,2,4,32,32,32,32,16,16,16]):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this 
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. 
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2. 
                Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)

        # # print(new_shape)
        # self.transformer1 = layers.SpatialTransformer((24,20,24))
        # self.transformer2 = layers.SpatialTransformer((48,40,48))
        # self.transformer3 = layers.SpatialTransformer((96,80,96))
        # self.transformer4 = layers.SpatialTransformer((192,160,192))

        self.transformer2 = layers.SpatialTransformer((48,40,40))
        self.transformer3 = layers.SpatialTransformer((96,80,80))
        self.transformer4 = layers.SpatialTransformer((192,160,160))

        # self.transformer2 = layers.SpatialTransformer((32,32,32))
        # self.transformer3 = layers.SpatialTransformer((64,64,64))
        # self.transformer4 = layers.SpatialTransformer((128,128,128))

        # self.transformer2 = layers.SpatialTransformer((40,48,56))
        # self.transformer3 = layers.SpatialTransformer((80,96,112))
        # self.transformer4 = layers.SpatialTransformer((160,192,224))
        # self.input_model = input_model
        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = MaxPooling(2)
        self.pooling_4 =MaxPooling(4)
        self.pooling_8 =MaxPooling(8)

        self.upsampling = nn.Upsample(scale_factor=2, mode='trilinear') 


        self.encoder1_m = ConvBlock(3,1,list_num[0])  # 权重共享的编码器, 提取moving
        self.encoder2_m = ConvBlock(3,list_num[0],list_num[1])
        self.encoder3_m = ConvBlock(3,list_num[1],list_num[2])
        self.encoder4_m = ConvBlock(3,list_num[2],list_num[3])
        self.encoder5_m = ConvBlock(3,list_num[3],list_num[4])

        
        self.encoder1_f = self.encoder1_m  # 权重共享的编码器, 提取fixed
        self.encoder2_f = self.encoder2_m
        self.encoder3_f = self.encoder3_m
        self.encoder4_f = self.encoder4_m
        self.encoder5_f = self.encoder5_m


        self.trans_2 = SwinTrans_stage_block(embed_dim=32,    # 16
                                             num_layers=2,
                                             num_heads=2,    # 1
                                             window_size=[5,5,5],
                                             use_checkpoint=False)
        self.trans_3 = SwinTrans_stage_block(embed_dim=32,    # 32
                                             num_layers=2,
                                             num_heads=2,   # 2
                                             window_size=[5,5,5],
                                             use_checkpoint=False)
        self.trans_4 = SwinTrans_stage_block(embed_dim=32,    # 64
                                             num_layers=2,
                                             num_heads=2,   #4
                                             window_size=[5,5,5],
                                             use_checkpoint=False)
        self.trans_5 = SwinTrans_stage_block(embed_dim=16, # 128
                                             num_layers=4,
                                             num_heads=1, # 8
                                             window_size=[5,5,5],
                                             use_checkpoint=False)

        
        self.decoder1 = ConvBlock(ndims,list_num[4]+list_num[4],32)
        self.decoder2 = ConvBlock(ndims,list_num[6]+list_num[6]+32,32)
        self.decoder3 = ConvBlock(ndims,list_num[8]+list_num[8]+32,32)
        self.decoder4 = ConvBlock(ndims,list_num[10]+list_num[10]+32,16)
        self.decoder5 = ConvBlock(ndims,list_num[0]+list_num[0]+16,16)

        self.output_block = nn.Sequential(ConvBlock(ndims,16,16),ConvBlock(ndims,16,16))
        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(16, ndims, kernel_size=3, padding=1)

        # # init flow layer with small weights and bias
        # self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        # self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        self.output_block_0 = nn.Sequential(ConvBlock(ndims,32,16),ConvBlock(ndims,16,3))
        self.output_block_1 = nn.Sequential(ConvBlock(ndims,32,16),ConvBlock(ndims,16,3))
        self.output_block_2 = nn.Sequential(ConvBlock(ndims,32,16),ConvBlock(ndims,16,3))
        self.output_block_3 = nn.Sequential(ConvBlock(ndims,16,16),ConvBlock(ndims,16,3))

        # 整合特征
        # # ------------------------------Scale4--------------------------------------------
        self.fusion_f_scale4 = ConvBlock(ndims,list_num[0]+list_num[1]+list_num[2]+list_num[3],list_num[5])
        self.fusion_f_scale4_2 = ConvBlock(ndims,list_num[5],list_num[6])


        self.fusion_m_scale4 = self.fusion_f_scale4
        self.fusion_m_scale4_2 = self.fusion_f_scale4_2

        # # ------------------------------Scale3--------------------------------------------
        self.fusion_f_scale3 = ConvBlock(ndims,list_num[0]+list_num[1]+list_num[2]+list_num[3],list_num[7])
        self.fusion_f_scale3_2 = ConvBlock(ndims,list_num[7],list_num[8])


        self.fusion_m_scale3 = self.fusion_f_scale3
        self.fusion_m_scale3_2 = self.fusion_f_scale3_2

        # # ------------------------------Scale2--------------------------------------------
        self.fusion_f_scale2 = ConvBlock(ndims,list_num[0]+list_num[1]+list_num[2]+list_num[3],list_num[9])
        self.fusion_f_scale2_2 = ConvBlock(ndims,list_num[9],list_num[10])


        self.fusion_m_scale2 = self.fusion_f_scale2
        self.fusion_m_scale2_2 = self.fusion_f_scale2_2


    
    def forward(self, source, target):
        '''
        Parameters:
            source: Source image tensor.F (moving)
            target: Target image tensor.  (fixed)
            registration: Return transformed image and flow. Default is False.

        ''' 
        xm = source
        xf = target
        xm_skip_1 = self.encoder1_m(xm) # (1,48,160,192,224)
        xm_skip_1 = xm_skip_1
        xf_skip_1 = self.encoder1_f(xf) # (1,48,160,192,224)
        xf_skip_1 = xf_skip_1

        # moving encoder
        #下采样到1/2,第二个encoder
        xm = self.pooling(xm_skip_1) # (1,16,80,96,112)
        xm_skip_2 = self.encoder2_m(xm) # (1,32,80,96,112)
        xm_skip_2 = xm_skip_2
        # 下采样到1/4,第三个encoder
        xm = self.pooling(xm_skip_2) # (1,32,40,48,56)
        xm_skip_3 = self.encoder3_m(xm) # (1,32,40,48,56)
        xm_skip_3 = xm_skip_3
        # 下采样到1/8,第四个encoder
        xm = self.pooling(xm_skip_3)  # (1,32,20,24,28)
        xm_skip_4 = self.encoder4_m(xm) # (1,32,20,24,28)
        xm_skip_4 = xm_skip_4
        # 下采样到1/16
        xm = self.pooling(xm_skip_4) # (1,32,10,12,14)
        xm_skip_5 = self.encoder5_m(xm) # (1,32,10,12,14)
        xm_skip_5 = xm_skip_5


        # fixed encoder
        #下采样到1/2,第二个encoder
        xf = self.pooling(xf_skip_1)
        xf_skip_2 = self.encoder2_f(xf)
        xf_skip_2 = xf_skip_2
        # 下采样到1/4,第三个encoder
        xf = self.pooling(xf_skip_2)
        xf_skip_3 = self.encoder3_f(xf)
        xf_skip_3 = xf_skip_3
        # 下采样到1/8,第四个encoder
        xf = self.pooling(xf_skip_3)  
        xf_skip_4 = self.encoder4_f(xf)
        xf_skip_4 = xf_skip_4
        # 下采样到1/16
        xf = self.pooling(xf_skip_4)
        xf_skip_5 = self.encoder5_f(xf) # (1,32,10,12,14)
        xf_skip_5 = xf_skip_5

        #-------------------------------Scale5--------------------------------
        # 第一个decoder
        x = torch.cat([xm_skip_5, xf_skip_5], dim=1) # (1,64,10,12,14)
        x = self.decoder1(x) # (1,32,10,12,14)
        x = self.trans_2(x)


        # ------------------------------Scale4--------------------------------------------
        # 融合多尺度特征：(20,24,28)
        xm_skip_1_fusion4 = self.pooling_8(xm_skip_1) # (1,16,20,24,28)
        xm_skip_2_fusion4 = self.pooling_4(xm_skip_2) # (1,32,20,24,28)
        xm_skip_3_fusion4 = self.pooling(xm_skip_3) # (1,32,20,24,28)
        xm_skip_4_fusion4 = xm_skip_4 # (1,32,20,24,28)

        xm_fusion4 = torch.cat([xm_skip_1_fusion4, xm_skip_2_fusion4], dim=1)
        xm_fusion4 = torch.cat([xm_fusion4, xm_skip_3_fusion4], dim=1)
        xm_fusion4 = torch.cat([xm_fusion4, xm_skip_4_fusion4], dim=1)

        xm_fusion4 = self.fusion_m_scale4(xm_fusion4) # (1,32,20,24,28)
        xm_fusion4 = xm_fusion4
        xm_fusion4 = self.fusion_m_scale4_2(xm_fusion4) # (1,32,20,24,28)
        xm_fusion4 = xm_fusion4
    

        xf_skip_1_fusion4 = self.pooling_8(xf_skip_1) # (1,16,20,24,28)
        xf_skip_2_fusion4 = self.pooling_4(xf_skip_2) # (1,32,20,24,28)
        xf_skip_3_fusion4 = self.pooling(xf_skip_3) # (1,32,20,24,28)
        xf_skip_4_fusion4 = xf_skip_4 # (1,32,20,24,28)

        xf_fusion4 = torch.cat([xf_skip_1_fusion4, xf_skip_2_fusion4], dim=1) 
        xf_fusion4 = torch.cat([xf_fusion4, xf_skip_3_fusion4], dim=1)
        xf_fusion4 = torch.cat([xf_fusion4, xf_skip_4_fusion4], dim=1)
   
        xf_fusion4 = self.fusion_f_scale4(xf_fusion4)
        xf_fusion4 = xf_fusion4
        xf_fusion4 = self.fusion_f_scale4_2(xf_fusion4) # (1,32,20,24,28) 
        xf_fusion4 = xf_fusion4

        
        x_fusion4 = torch.cat([xm_fusion4, xf_fusion4], dim=1) # (1,64,20,24,28)

    
        # 第二个decoder
        x = self.decoder2(torch.cat([self.upsampling(x), x_fusion4], dim=1)) # (1,32,20,24,28)
        x = self.trans_3(x)

        # ----输出1/8分辨率的变形场-----
        flow_1 = self.output_block_1(x) # (1,3,20,24,28) 
        flow_1_up = nn.functional.interpolate(flow_1, scale_factor=2,mode="trilinear")*2 # (1,3,40,48,56)


        ## ------------------------------Scale3--------------------------------------------
        # 融合多尺度特征：(40,48,56)
        xm_skip_1_fusion3 = self.pooling_4(xm_skip_1) # (1,16,40,48,56)
        xm_skip_2_fusion3 = self.pooling(xm_skip_2) # (1,32,40,48,56)
        xm_skip_3_fusion3 = xm_skip_3 # (1,32,40,48,56)
        xm_skip_4_fusion3 = nn.functional.interpolate(xm_skip_4, scale_factor=2,mode="trilinear")
        

        xm_fusion3 = torch.cat([xm_skip_1_fusion3, xm_skip_2_fusion3], dim=1)
        xm_fusion3 = torch.cat([xm_fusion3, xm_skip_3_fusion3], dim=1)
        xm_fusion3 = torch.cat([xm_fusion3, xm_skip_4_fusion3], dim=1) # (1,112,40,48,56)

        # xm_fusion3 = self.fusion_m_scale3(xm_fusion3) # (1,32,40,48,56)
        # xm_fusion3 = xm_fusion3
        # xm_fusion3 = self.fusion_m_scale3_2(xm_fusion3) # (1,32,40,48,56)
        # xm_fusion3 = xm_fusion3

        # xw_fusion3 = self.transformer2(xm_fusion3, flow_1_up)
        xw_fusion3 = self.transformer2(xm_fusion3, flow_1_up)
        # 2. 再过融合卷积
        xw_fusion3 = self.fusion_m_scale3(xw_fusion3)
        xw_fusion3 = self.fusion_m_scale3_2(xw_fusion3)


        xf_skip_1_fusion3 = self.pooling_4(xf_skip_1) # (1,16,40,48,56)
        xf_skip_2_fusion3 = self.pooling(xf_skip_2) # (1,32,40,48,56)
        xf_skip_3_fusion3 = xf_skip_3 # (1,32,40,48,56)

        xf_skip_4_fusion3 = nn.functional.interpolate(xf_skip_4, scale_factor=2,mode="trilinear")
        

        xf_fusion3 = torch.cat([xf_skip_1_fusion3, xf_skip_2_fusion3], dim=1)
        xf_fusion3 = torch.cat([xf_fusion3, xf_skip_3_fusion3], dim=1)
        xf_fusion3 = torch.cat([xf_fusion3, xf_skip_4_fusion3], dim=1) # (1,112,40,48,56)

        xf_fusion3 = self.fusion_f_scale3(xf_fusion3) # (1,32,40,48,56)
        xf_fusion3 = xf_fusion3
        xf_fusion3 = self.fusion_f_scale3_2(xf_fusion3) # (1,32,40,48,56)
        xf_fusion3 = xf_fusion3

        # concat
        x_fusion3 = torch.cat([xw_fusion3, xf_fusion3], dim=1)  # (1,64,40,48,56)

        # 第三个decoder
        x = self.decoder3(torch.cat([self.upsampling(x), x_fusion3], dim=1)) # (1,32,40,48,56)
        x = self.trans_4(x)

        # ----输出1/4分辨率的变形场---
        delta_flow_2 = self.output_block_2(x) # (1,3,40,48,56)
        flow_2 = delta_flow_2 + flow_1_up # (1,3,40,48,56)
        flow_2_up = nn.functional.interpolate(flow_2, scale_factor=2,mode="trilinear")*2 # (1,3,80,96,112)

        ## ------------------------------Scale2--------------------------------------------
        # 融合多尺度特征：(80,96,112)
        xm_skip_1_fusion2 = self.pooling(xm_skip_1) # (1,16,80,96,112)
        xm_skip_2_fusion2 = xm_skip_2 # (1,32,80,96,112)

        xm_skip_3_fusion2 = nn.functional.interpolate(xm_skip_3, scale_factor=2,mode="trilinear")  # (1,32,80,96,112)
        
        xm_skip_4_fusion2 = nn.functional.interpolate(xm_skip_4, scale_factor=4,mode="trilinear")
        

        xm_fusion2 = torch.cat([xm_skip_1_fusion2, xm_skip_2_fusion2], dim=1)
        xm_fusion2 = torch.cat([xm_fusion2, xm_skip_3_fusion2], dim=1)
        xm_fusion2 = torch.cat([xm_fusion2, xm_skip_4_fusion2], dim=1) # (1,112,80,96,112)


        # xm_fusion2 = self.fusion_m_scale2(xm_fusion2) # (1,32,80,96,112)
        # xm_fusion2 = xm_fusion2
        # xm_fusion2 = self.fusion_m_scale2_2(xm_fusion2) # (1,32,80,96,112)
        # xm_fusion2 = xm_fusion2


        # xw_fusion2 = self.transformer3(xm_fusion2, flow_2_up)
        xw_fusion2 = self.transformer3(xm_fusion2, flow_2_up)
        # 2. 再过融合卷积
        xw_fusion2 = self.fusion_m_scale2(xw_fusion2)
        xw_fusion2 = self.fusion_m_scale2_2(xw_fusion2)

        xf_skip_1_fusion2 = self.pooling(xf_skip_1) # (1,16,80,96,112)
        xf_skip_2_fusion2 = xf_skip_2 # (1,32,80,96,112)

        xf_skip_3_fusion2 = nn.functional.interpolate(xf_skip_3, scale_factor=2,mode="trilinear")  # (1,32,80,96,112)
        
        xf_skip_4_fusion2 = nn.functional.interpolate(xf_skip_4, scale_factor=4,mode="trilinear")
        

        xf_fusion2 = torch.cat([xf_skip_1_fusion2, xf_skip_2_fusion2], dim=1)
        xf_fusion2 = torch.cat([xf_fusion2, xf_skip_3_fusion2], dim=1)
        xf_fusion2 = torch.cat([xf_fusion2, xf_skip_4_fusion2], dim=1) # (1,112,80,96,112)

        xf_fusion2 = self.fusion_f_scale2(xf_fusion2) # (1,32,80,96,112)
        xf_fusion2 = xf_fusion2
        xf_fusion2 = self.fusion_f_scale2_2(xf_fusion2) # (1,32,80,96,112)
        xf_fusion2 = xf_fusion2

        # concat
        x_fusion2 = torch.cat([xw_fusion2, xf_fusion2], dim=1)  # (1,32,80,96,112)


        # 第四个decoder
        x = self.decoder4(torch.cat([self.upsampling(x), x_fusion2], dim=1)) # (1,32,80,96,112)
        x = self.trans_5(x)
        # print(x.shape)

        # ----输出1/2分辨率的变形场-------
        delta_flow_3 = self.output_block_3(x) # (1,3,80,96,112)
        flow_3 = delta_flow_3 + flow_2_up # (1,3,80,96,112)
        flow_3_up = nn.functional.interpolate(flow_3, scale_factor=2,mode="trilinear")*2 # (1,3,160,112,224)

        # ------------------------------Scale1--------------------------------------------
        # # 对moving feature进行warp得到moved feature
        xw_skip_1 = self.transformer4(xm_skip_1, flow_3_up)
        # concat
        x_skip_1 = torch.cat([xw_skip_1, xf_skip_1], dim=1)  # (1,32,160,192,224)
        # print(x_skip_1.shape)


        # 第四个decoder
        x = self.decoder5(torch.cat([self.upsampling(x), x_skip_1], dim=1)) # (1,32,160,192,224)

        # ----------------------输出最终的变形场------------------------------------------
        # output block
        x = self.output_block(x) # (1,16,160,192,224)
        # 生成flow场
        delta_flow_final = self.flow(x) # (1,3,160,192,224)
        flow_final = delta_flow_final + flow_3_up
            

        # return flow_1,flow_2,flow_3,flow_final,delta_flow_2,delta_flow_3,delta_flow_final
        return flow_1,flow_2,flow_3,flow_final,delta_flow_2,delta_flow_3,delta_flow_final



#---------------------PDFNet的微分同胚版本------------------------------------------
# #(with FFM) 直接融合 + 俩卷积 (剪枝后的网络) 
class dual_pyramid_VxmDense_FFM_normal_diff_new_GDP(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    自己写的, 两个权重共享的编码器来各自提取img_concat_wavelet的特征,
    然后特征融合, 再送入同一个解码器
    在multi-vxmdense的基础上改的

    配准网络本身配准的是输入model的source和target的尺寸, 而SPT则是new_shape的尺寸
    特别需要注意的就是输入的source和target的尺寸是否能完成四次下采样, 不能的话要注意调整网络下采样的次数
    最后生成的flow可以通过self.flow里面的卷积步长stride来改变尺寸(下采样等等)
    """

    @store_config_args
    def __init__(self,
                 inshape=(128,128,128)):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this 
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. 
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2. 
                Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)

        self.warp = nn.ModuleList()
        self.diff = nn.ModuleList()
        for i in range(4):
            self.warp.append(layers.SpatialTransformer([s // 2**i for s in inshape]))
            self.diff.append(layers.VecInt([s // 2**i for s in inshape]))

        # self.upsample_trilin = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        # # print(new_shape)
        # self.transformer1 = layers.SpatialTransformer((20,24,28))
        # self.transformer2 = layers.SpatialTransformer((40,48,56))
        # self.transformer3 = layers.SpatialTransformer((80,96,112))
        # self.transformer4 = layers.SpatialTransformer((160,192,224))

        # self.transformer1 = layers.SpatialTransformer((16,16,16))
        # self.transformer2 = layers.SpatialTransformer((32,32,32))
        # self.transformer3 = layers.SpatialTransformer((64,64,64))
        # self.transformer4 = layers.SpatialTransformer((128,128,128))


        self.transformer1 = layers.SpatialTransformer((24,20,24))
        self.transformer2 = layers.SpatialTransformer((48,40,48))
        self.transformer3 = layers.SpatialTransformer((96,80,96))
        self.transformer4 = layers.SpatialTransformer((192,160,192))

        # self.transformer1 = layers.SpatialTransformer((24,20,20))
        # self.transformer2 = layers.SpatialTransformer((48,40,40))
        # self.transformer3 = layers.SpatialTransformer((96,80,80))
        # self.transformer4 = layers.SpatialTransformer((192,160,160))
        # self.input_model = input_model
        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = MaxPooling(2)
        self.pooling_4 =MaxPooling(4)
        self.pooling_8 =MaxPooling(8)

        self.upsampling = nn.Upsample(scale_factor=2, mode='trilinear') 


        self.encoder1_m = ConvBlock(3,1,16)  # 权重共享的编码器, 提取moving
        self.encoder2_m = ConvBlock(3,16,32)
        self.encoder3_m = ConvBlock(3,32,32)
        self.encoder4_m = ConvBlock(3,32,32)
        self.encoder5_m = ConvBlock(3,32,32)

        
        self.encoder1_f = self.encoder1_m  # 权重共享的编码器, 提取fixed
        self.encoder2_f = self.encoder2_m
        self.encoder3_f = self.encoder3_m
        self.encoder4_f = self.encoder4_m
        self.encoder5_f = self.encoder5_m

        
        self.decoder1 = ConvBlock(ndims,64,32)
        self.decoder2 = ConvBlock(ndims,96,32)
        self.decoder3 = ConvBlock(ndims,96,32)
        self.decoder4 = ConvBlock(ndims,96,16)
        self.decoder5 = ConvBlock(ndims,48,16)

        self.output_block = nn.Sequential(ConvBlock(ndims,16,16),ConvBlock(ndims,16,16))
        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(16, ndims, kernel_size=3, padding=1)

        # # init flow layer with small weights and bias
        # self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        # self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        self.output_block_0 = nn.Sequential(ConvBlock(ndims,32,16),ConvBlock(ndims,16,3))
        self.output_block_1 = nn.Sequential(ConvBlock(ndims,32,16),ConvBlock(ndims,16,3))
        self.output_block_2 = nn.Sequential(ConvBlock(ndims,32,16),ConvBlock(ndims,16,3))
        self.output_block_3 = nn.Sequential(ConvBlock(ndims,16,16),ConvBlock(ndims,16,3))

        # 整合特征
        # # ------------------------------Scale4--------------------------------------------
        self.fusion_f_scale4 = ConvBlock(ndims,112,32)
        self.fusion_f_scale4_2 = ConvBlock(ndims,32,32)

        self.fusion_m_scale4 = self.fusion_f_scale4
        self.fusion_m_scale4_2 = self.fusion_f_scale4_2

        # # ------------------------------Scale3--------------------------------------------
        self.fusion_f_scale3 = ConvBlock(ndims,112,32)
        self.fusion_f_scale3_2 = ConvBlock(ndims,32,32)

        self.fusion_m_scale3 = self.fusion_f_scale3
        self.fusion_m_scale3_2 = self.fusion_f_scale3_2

        # # ------------------------------Scale2--------------------------------------------
        self.fusion_f_scale2 = ConvBlock(ndims,112,32)
        self.fusion_f_scale2_2 = ConvBlock(ndims,32,32)

        self.fusion_m_scale2 = self.fusion_f_scale2
        self.fusion_m_scale2_2 = self.fusion_f_scale2_2

        
        # self.integrate0 = layers.VecInt((12,10,10), 7)
        # self.integrate1 = layers.VecInt((24,20,20), 7)
        # self.integrate2 = layers.VecInt((48,40,40), 7)
        # self.integrate3 = layers.VecInt((96,80,80), 7)

        self.integrate0 = layers.VecInt((12,10,12), 7)
        self.integrate1 = layers.VecInt((24,20,24), 7)
        self.integrate2 = layers.VecInt((48,40,48), 7)
        self.integrate3 = layers.VecInt((96,80,96), 7)

        # self.integrate0 = layers.VecInt((10,12,14), 7)
        # self.integrate1 = layers.VecInt((20,24,28), 7)
        # self.integrate2 = layers.VecInt((40,48,56), 7)
        # self.integrate3 = layers.VecInt((80,96,112), 7)
        self.fullsize = layers.ResizeTransform(1 / 2, 3)
        self.resize = layers.ResizeTransform(2, 3)

        # 定义筛选特征的向量
        self.alpha_1 = nn.Parameter(torch.ones(1, 16))
        self.alpha_2 = nn.Parameter(torch.ones(1, 32))
        self.alpha_3 = nn.Parameter(torch.ones(1, 32))
        self.alpha_4 = nn.Parameter(torch.ones(1, 32))
        self.alpha_5 = nn.Parameter(torch.ones(1, 32))
        self.alpha_6_1 = nn.Parameter(torch.ones(1, 32))
        self.alpha_6_2 = nn.Parameter(torch.ones(1, 32))
        self.alpha_7_1 = nn.Parameter(torch.ones(1, 32))
        self.alpha_7_2 = nn.Parameter(torch.ones(1, 32))
        self.alpha_8_1 = nn.Parameter(torch.ones(1, 32))
        self.alpha_8_2 = nn.Parameter(torch.ones(1, 32))

        # 定义门函数
        self.gate_activation = GatingFunction()

        # 更新decay的系数
    def set_decay(self, new_decay_value):
        self.gate_activation.decay = new_decay_value
    
    
    
    def forward(self, source, target):
        '''
        Parameters:
            source: Source image tensor.F (moving)
            target: Target image tensor.  (fixed)
            registration: Return transformed image and flow. Default is False.
        '''
        ori_alpha_1 = self.alpha_1
        ori_alpha_2 = self.alpha_2
        ori_alpha_3 = self.alpha_3
        ori_alpha_4 = self.alpha_4
        ori_alpha_5 = self.alpha_5
        ori_alpha_6_1 = self.alpha_6_1
        ori_alpha_6_2 = self.alpha_6_2
        ori_alpha_7_1 = self.alpha_7_1
        ori_alpha_7_2 = self.alpha_7_2
        ori_alpha_8_1 = self.alpha_8_1
        ori_alpha_8_2 = self.alpha_8_2
        
        # 输入门函数
        alpha_1 = self.gate_activation(ori_alpha_1)
        alpha_2 = self.gate_activation(ori_alpha_2)
        alpha_3 = self.gate_activation(ori_alpha_3)
        alpha_4 = self.gate_activation(ori_alpha_4)
        alpha_5 = self.gate_activation(ori_alpha_5)
        alpha_6_1 = self.gate_activation(ori_alpha_6_1)
        alpha_6_2 = self.gate_activation(ori_alpha_6_2)
        alpha_7_1 = self.gate_activation(ori_alpha_7_1)
        alpha_7_2 = self.gate_activation(ori_alpha_7_2)
        alpha_8_1 = self.gate_activation(ori_alpha_8_1)
        alpha_8_2 = self.gate_activation(ori_alpha_8_2)
        ori_alpha = [ori_alpha_1,ori_alpha_2,ori_alpha_3,ori_alpha_4,ori_alpha_5,ori_alpha_6_1,ori_alpha_6_2,ori_alpha_7_1,ori_alpha_7_2,ori_alpha_8_1,ori_alpha_8_2]
        gate_alpha = [alpha_1,alpha_2,alpha_3,alpha_4,alpha_5,alpha_6_1,alpha_6_2,alpha_7_1,alpha_7_2,alpha_8_1,alpha_8_2]

        # 维度扩展
        alpha_1 = alpha_1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  
        alpha_2 = alpha_2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        alpha_3 = alpha_3.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        alpha_4 = alpha_4.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        alpha_5 = alpha_5.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        alpha_6_1 = alpha_6_1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) 
        alpha_6_2 = alpha_6_2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        alpha_7_1 = alpha_7_1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) 
        alpha_7_2 = alpha_7_2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        alpha_8_1 = alpha_8_1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) 
        alpha_8_2 = alpha_8_2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  

        xm = source
        xf = target
        xm_skip_1 = self.encoder1_m(xm) # (1,16,160,192,224)
        xm_skip_1 = xm_skip_1 * alpha_1
        xf_skip_1 = self.encoder1_f(xf) # (1,16,160,192,224)
        xf_skip_1 = xf_skip_1 * alpha_1

        # moving encoder
        #下采样到1/2,第二个encoder
        xm = self.pooling(xm_skip_1) # (1,16,80,96,112)
        xm_skip_2 = self.encoder2_m(xm) # (1,32,80,96,112)
        xm_skip_2 = xm_skip_2 * alpha_2
        # 下采样到1/4,第三个encoder
        xm = self.pooling(xm_skip_2) # (1,32,40,48,56)
        xm_skip_3 = self.encoder3_m(xm) # (1,32,40,48,56)
        xm_skip_3 = xm_skip_3 * alpha_3
        # 下采样到1/8,第四个encoder
        xm = self.pooling(xm_skip_3)  # (1,32,20,24,28)
        xm_skip_4 = self.encoder4_m(xm) # (1,32,20,24,28)
        xm_skip_4 = xm_skip_4 * alpha_4
        # 下采样到1/16
        xm = self.pooling(xm_skip_4) # (1,32,10,12,14)
        xm = self.encoder5_m(xm) # (1,32,10,12,14)
        xm = xm * alpha_5


        # fixed encoder
        #下采样到1/2,第二个encoder
        xf = self.pooling(xf_skip_1)
        xf_skip_2 = self.encoder2_f(xf)
        xf_skip_2 = xf_skip_2 * alpha_2
        # 下采样到1/4,第三个encoder
        xf = self.pooling(xf_skip_2)
        xf_skip_3 = self.encoder3_f(xf)
        xf_skip_3 = xf_skip_3 * alpha_3
        # 下采样到1/8,第四个encoder
        xf = self.pooling(xf_skip_3)  
        xf_skip_4 = self.encoder4_f(xf)
        xf_skip_4 = xf_skip_4 * alpha_4
        # 下采样到1/16
        xf = self.pooling(xf_skip_4)
        xf = self.encoder5_f(xf) # (1,32,10,12,14)
        xf = xf * alpha_5

        #-------------------------------Scale5--------------------------------
        # 第一个decoder
        x = torch.cat([xm, xf], dim=1) # (1,64,10,12,14)
        x = self.decoder1(x) # (1,32,10,12,14)


        # ------------------------------Scale4--------------------------------------------
        # 融合多尺度特征：(20,24,28)
        xm_skip_1_fusion4 = self.pooling_8(xm_skip_1) # (1,16,20,24,28)
        # xm_skip_1_fusion4 = self.fusion_m1_scale4(xm_skip_1_fusion4) 
        xm_skip_2_fusion4 = self.pooling_4(xm_skip_2) # (1,32,20,24,28)
        # xm_skip_2_fusion4 = self.fusion_m2_scale4(xm_skip_2_fusion4)
        xm_skip_3_fusion4 = self.pooling(xm_skip_3) # (1,32,20,24,28)
        # xm_skip_3_fusion4 = self.fusion_m3_scale4(xm_skip_3_fusion4)
        xm_skip_4_fusion4 = xm_skip_4 # (1,32,20,24,28)

        xm_fusion4 = torch.cat([xm_skip_1_fusion4, xm_skip_2_fusion4], dim=1)
        xm_fusion4 = torch.cat([xm_fusion4, xm_skip_3_fusion4], dim=1)
        xm_fusion4 = torch.cat([xm_fusion4, xm_skip_4_fusion4], dim=1)

        xm_fusion4 = self.fusion_m_scale4(xm_fusion4) # (1,32,20,24,28)
        xm_fusion4 = xm_fusion4 * alpha_6_1
        xm_fusion4 = self.fusion_m_scale4_2(xm_fusion4) # (1,32,20,24,28)
        xm_fusion4 = xm_fusion4 * alpha_6_2
    

        xf_skip_1_fusion4 = self.pooling_8(xf_skip_1) # (1,16,20,24,28)
        # xf_skip_1_fusion4 = self.fusion_f1_scale4(xf_skip_1_fusion4)
        xf_skip_2_fusion4 = self.pooling_4(xf_skip_2) # (1,32,20,24,28)
        # xf_skip_2_fusion4 = self.fusion_f2_scale4(xf_skip_2_fusion4)
        xf_skip_3_fusion4 = self.pooling(xf_skip_3) # (1,32,20,24,28)
        # xf_skip_3_fusion4 = self.fusion_f3_scale4(xf_skip_3_fusion4)
        xf_skip_4_fusion4 = xf_skip_4 # (1,32,20,24,28)

        xf_fusion4 = torch.cat([xf_skip_1_fusion4, xf_skip_2_fusion4], dim=1) 
        xf_fusion4 = torch.cat([xf_fusion4, xf_skip_3_fusion4], dim=1)
        xf_fusion4 = torch.cat([xf_fusion4, xf_skip_4_fusion4], dim=1)
   
        xf_fusion4 = self.fusion_f_scale4(xf_fusion4)
        xf_fusion4 = xf_fusion4 * alpha_6_1
        xf_fusion4 = self.fusion_f_scale4_2(xf_fusion4) # (1,32,20,24,28) 
        xf_fusion4 = xf_fusion4 * alpha_6_2

        
        x_fusion4 = torch.cat([xm_fusion4, xf_fusion4], dim=1) # (1,64,20,24,28)

    
        # 第二个decoder
        x = self.decoder2(torch.cat([self.upsampling(x), x_fusion4], dim=1)) # (1,32,20,24,28)
        # x = self.trans_3(x)

        # ----输出1/8分辨率的变形场-----
        vec_flow_1 = self.output_block_1(x) # (1,3,20,24,28) 

        # 速度场生成变形场
        # vec_flow_1 = self.resize(vec_flow_1) # (1,3,10,12,14)
        # pos_flow = self.integrate0(vec_flow_1)
        # flow_1 = self.fullsize(pos_flow) # (1,3,20,24,28) 
        flow_1 = self.diff[3](vec_flow_1) # (1,3,20,24,28) 

        flow_1_up = nn.functional.interpolate(flow_1, scale_factor=2,mode="trilinear")*2 # (1,3,40,48,56)


        ## ------------------------------Scale3--------------------------------------------
        # 融合多尺度特征：(40,48,56)
        xm_skip_1_fusion3 = self.pooling_4(xm_skip_1) # (1,16,40,48,56)
        # xm_skip_1_fusion3 = self.fusion_m1_scale3(xm_skip_1_fusion3) 
        xm_skip_2_fusion3 = self.pooling(xm_skip_2) # (1,32,40,48,56)
        # xm_skip_2_fusion3 = self.fusion_m2_scale3(xm_skip_2_fusion3)
        xm_skip_3_fusion3 = xm_skip_3 # (1,32,40,48,56)
        # xm_skip_3_fusion3 = self.fusion_m3_scale3(xm_skip_3_fusion3)

        # xm_skip_4_fusion3 = self.fusion_m4_scale3(xm_skip_4) # (1,32,40,48,56)
        xm_skip_4_fusion3 = nn.functional.interpolate(xm_skip_4, scale_factor=2,mode="trilinear")
        

        xm_fusion3 = torch.cat([xm_skip_1_fusion3, xm_skip_2_fusion3], dim=1)
        xm_fusion3 = torch.cat([xm_fusion3, xm_skip_3_fusion3], dim=1)
        xm_fusion3 = torch.cat([xm_fusion3, xm_skip_4_fusion3], dim=1) # (1,112,40,48,56)

        xm_fusion3 = self.fusion_m_scale3(xm_fusion3) # (1,32,40,48,56)
        xm_fusion3 = xm_fusion3 * alpha_7_1
        xm_fusion3 = self.fusion_m_scale3_2(xm_fusion3) # (1,32,40,48,56)
        xm_fusion3 = xm_fusion3 * alpha_7_2

        xw_fusion3 = self.transformer2(xm_fusion3, flow_1_up)


        xf_skip_1_fusion3 = self.pooling_4(xf_skip_1) # (1,16,40,48,56)
        # xf_skip_1_fusion3 = self.fusion_f1_scale3(xf_skip_1_fusion3) 
        xf_skip_2_fusion3 = self.pooling(xf_skip_2) # (1,32,40,48,56)
        # xf_skip_2_fusion3 = self.fusion_f2_scale3(xf_skip_2_fusion3)
        xf_skip_3_fusion3 = xf_skip_3 # (1,32,40,48,56)
        # xf_skip_3_fusion3 = self.fusion_f3_scale3(xf_skip_3_fusion3)

        # xf_skip_4_fusion3 = self.fusion_f4_scale3(xf_skip_4) # (1,32,40,48,56)
        xf_skip_4_fusion3 = nn.functional.interpolate(xf_skip_4, scale_factor=2,mode="trilinear")
        

        xf_fusion3 = torch.cat([xf_skip_1_fusion3, xf_skip_2_fusion3], dim=1)
        xf_fusion3 = torch.cat([xf_fusion3, xf_skip_3_fusion3], dim=1)
        xf_fusion3 = torch.cat([xf_fusion3, xf_skip_4_fusion3], dim=1) # (1,112,40,48,56)

        xf_fusion3 = self.fusion_f_scale3(xf_fusion3) # (1,32,40,48,56)
        xf_fusion3 = xf_fusion3 * alpha_7_1
        xf_fusion3 = self.fusion_f_scale3_2(xf_fusion3) # (1,32,40,48,56)
        xf_fusion3 = xf_fusion3 * alpha_7_2

        # concat
        x_fusion3 = torch.cat([xw_fusion3, xf_fusion3], dim=1)  # (1,64,40,48,56)

        # 第三个decoder
        x = self.decoder3(torch.cat([self.upsampling(x), x_fusion3], dim=1)) # (1,32,40,48,56)
        # x = self.trans_4(x)

        # ----输出1/4分辨率的变形场---
        delta_vec_flow_2 = self.output_block_2(x) # (1,3,40,48,56)

        # down_delta_vec_flow_2 = self.resize(delta_vec_flow_2) # (1,3,20,24,28)
        # pos_flow = self.integrate1(down_delta_vec_flow_2)
        # delta_flow_2 = self.fullsize(pos_flow) # (1,3,40,48,56)
        delta_flow_2 = self.diff[2](delta_vec_flow_2) # (1,3,40,48,56)

        flow_2 = self.warp[2](flow_1_up, delta_flow_2)+delta_flow_2



        # total_vec_flow_2 = down_delta_vec_flow_2 + nn.functional.interpolate(vec_flow_1,scale_factor=2,mode='trilinear')*2  # (1,3,20,24,28)

        # # 速度场生成变形场
        # pos_flow = self.integrate1(total_vec_flow_2)
        # flow_2 = self.fullsize(pos_flow) # (1,3,40,48,56)

        flow_2_up = nn.functional.interpolate(flow_2, scale_factor=2,mode="trilinear")*2 # (1,3,80,96,112)

        ## ------------------------------Scale2--------------------------------------------
        # 融合多尺度特征：(80,96,112)
        xm_skip_1_fusion2 = self.pooling(xm_skip_1) # (1,16,80,96,112)
        # xm_skip_1_fusion2 = self.fusion_m1_scale2(xm_skip_1_fusion2) 
        xm_skip_2_fusion2 = xm_skip_2 # (1,32,80,96,112)
        # xm_skip_2_fusion2 = self.fusion_m2_scale2(xm_skip_2_fusion2)

        # xm_skip_3_fusion2 = self.fusion_m3_scale2(xm_skip_3)
        xm_skip_3_fusion2 = nn.functional.interpolate(xm_skip_3, scale_factor=2,mode="trilinear")  # (1,32,80,96,112)
        
        # xm_skip_4_fusion2 = self.fusion_m4_scale2(xm_skip_4) # (1,32,80,96,112)
        xm_skip_4_fusion2 = nn.functional.interpolate(xm_skip_4, scale_factor=4,mode="trilinear")
        

        xm_fusion2 = torch.cat([xm_skip_1_fusion2, xm_skip_2_fusion2], dim=1)
        xm_fusion2 = torch.cat([xm_fusion2, xm_skip_3_fusion2], dim=1)
        xm_fusion2 = torch.cat([xm_fusion2, xm_skip_4_fusion2], dim=1) # (1,112,80,96,112)


        xm_fusion2 = self.fusion_m_scale2(xm_fusion2) # (1,32,80,96,112)
        xm_fusion2 = xm_fusion2 * alpha_8_1
        xm_fusion2 = self.fusion_m_scale2_2(xm_fusion2) # (1,32,80,96,112)
        xm_fusion2 = xm_fusion2 * alpha_8_2


        xw_fusion2 = self.transformer3(xm_fusion2, flow_2_up)

        xf_skip_1_fusion2 = self.pooling(xf_skip_1) # (1,16,80,96,112)
        # xf_skip_1_fusion2 = self.fusion_f1_scale2(xf_skip_1_fusion2) 
        xf_skip_2_fusion2 = xf_skip_2 # (1,32,80,96,112)
        # xf_skip_2_fusion2 = self.fusion_f2_scale2(xf_skip_2_fusion2)

        # xf_skip_3_fusion2 = self.fusion_f3_scale2(xf_skip_3)
        xf_skip_3_fusion2 = nn.functional.interpolate(xf_skip_3, scale_factor=2,mode="trilinear")  # (1,32,80,96,112)
        
        # xf_skip_4_fusion2 = self.fusion_f4_scale2(xf_skip_4) # (1,32,80,96,112)
        xf_skip_4_fusion2 = nn.functional.interpolate(xf_skip_4, scale_factor=4,mode="trilinear")
        

        xf_fusion2 = torch.cat([xf_skip_1_fusion2, xf_skip_2_fusion2], dim=1)
        xf_fusion2 = torch.cat([xf_fusion2, xf_skip_3_fusion2], dim=1)
        xf_fusion2 = torch.cat([xf_fusion2, xf_skip_4_fusion2], dim=1) # (1,112,80,96,112)

        xf_fusion2 = self.fusion_f_scale2(xf_fusion2) # (1,32,80,96,112)
        xf_fusion2 = xf_fusion2 * alpha_8_1
        xf_fusion2 = self.fusion_f_scale2_2(xf_fusion2) # (1,32,80,96,112)
        xf_fusion2 = xf_fusion2 * alpha_8_2

        # concat
        x_fusion2 = torch.cat([xw_fusion2, xf_fusion2], dim=1)  # (1,32,80,96,112)


        # 第四个decoder
        x = self.decoder4(torch.cat([self.upsampling(x), x_fusion2], dim=1)) # (1,32,80,96,112)
        # x = self.trans_5(x)
        # print(x.shape)

        # ----输出1/2分辨率的变形场-------
        delta_vec_flow_3 = self.output_block_3(x) # (1,3,80,96,112)
        # down_delta_vec_flow_3 = self.resize(delta_vec_flow_3) # (1,3,40,48,56)
        # pos_flow = self.integrate2(down_delta_vec_flow_3)
        # delta_flow_3 = self.fullsize(pos_flow) # (1,3,80,96,112)
        delta_flow_3 = self.diff[1](delta_vec_flow_3) # (1,3,80,96,112)

        flow_3 = self.warp[1](flow_2_up, delta_flow_3)+delta_flow_3


        # total_vec_flow_3 = down_delta_vec_flow_3 + nn.functional.interpolate(total_vec_flow_2,scale_factor=2,mode='trilinear')*2  # (1,3,40,48,56)

        # # 速度场生成变形场
        # pos_flow = self.integrate2(total_vec_flow_3)
        # flow_3 = self.fullsize(pos_flow) # (1,3,80,96,112)

        flow_3_up = nn.functional.interpolate(flow_3, scale_factor=2,mode="trilinear")*2 # (1,3,160,112,224)

        # ------------------------------Scale1--------------------------------------------
        # # 对moving feature进行warp得到moved feature
        xw_skip_1 = self.transformer4(xm_skip_1, flow_3_up)
        # concat
        x_skip_1 = torch.cat([xw_skip_1, xf_skip_1], dim=1)  # (1,32,160,192,224)
        # print(x_skip_1.shape)


        # 第四个decoder
        x = self.decoder5(torch.cat([self.upsampling(x), x_skip_1], dim=1)) # (1,32,160,192,224)

        # ----------------------输出最终的变形场------------------------------------------
        # output block
        x = self.output_block(x) # (1,16,160,192,224)
        # 生成flow场
        delta_vec_flow_final = self.flow(x) # (1,3,160,192,224)
        # down_delta_vec_flow_final = self.resize(delta_vec_flow_final) # (1,3,80,96,112)
        # pos_flow = self.integrate3(down_delta_vec_flow_final)
        # delta_flow_final = self.fullsize(pos_flow) # (1,3,80,96,112)
        delta_flow_final = self.diff[0](delta_vec_flow_final) # (1,3,160,192,224)

        # total_vec_flow_final = down_delta_vec_flow_final + nn.functional.interpolate(total_vec_flow_3,scale_factor=2,mode='trilinear')*2  # (1,3,80,96,112)

        # # 速度场生成变形场
        # pos_flow = self.integrate3(total_vec_flow_final)
        # flow_final = self.fullsize(pos_flow) # (1,3,80,96,112)
        flow_final = self.warp[0](flow_3_up, delta_flow_final)+delta_flow_final
       

        return flow_1,flow_2,flow_3,flow_final,delta_flow_2,delta_flow_3,delta_flow_final,ori_alpha,gate_alpha

#(with FFM) 直接融合 + 俩卷积 (剪枝后的网络) 
class dual_pyramid_VxmDense_FFM_normal_diff_new_adaptive_val(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    自己写的, 两个权重共享的编码器来各自提取img_concat_wavelet的特征,
    然后特征融合, 再送入同一个解码器
    在multi-vxmdense的基础上改的

    配准网络本身配准的是输入model的source和target的尺寸, 而SPT则是new_shape的尺寸
    特别需要注意的就是输入的source和target的尺寸是否能完成四次下采样, 不能的话要注意调整网络下采样的次数
    最后生成的flow可以通过self.flow里面的卷积步长stride来改变尺寸(下采样等等)
    """

    @store_config_args
    def __init__(self,
                 inshape=(128,128,128),
                 list_num=[5,3,2,4,1,1,1,19,12,16,14]):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this 
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. 
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2. 
                Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)

        self.warp = nn.ModuleList()
        self.diff = nn.ModuleList()
        for i in range(4):
            self.warp.append(SpatialTransformer([s // 2**i for s in inshape]))
            self.diff.append(VecInt([s // 2**i for s in inshape]))

        # self.upsample_trilin = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        # # print(new_shape)
        self.transformer1 = layers.SpatialTransformer((16,16,16))
        self.transformer2 = layers.SpatialTransformer((32,32,32))
        self.transformer3 = layers.SpatialTransformer((64,64,64))
        self.transformer4 = layers.SpatialTransformer((128,128,128))

        # self.transformer1 = layers.SpatialTransformer((20,24,28))
        # self.transformer2 = layers.SpatialTransformer((40,48,56))
        # self.transformer3 = layers.SpatialTransformer((80,96,112))
        # self.transformer4 = layers.SpatialTransformer((160,192,224))

        # self.transformer1 = layers.SpatialTransformer((24,20,20))
        # self.transformer2 = layers.SpatialTransformer((48,40,40))
        # self.transformer3 = layers.SpatialTransformer((96,80,80))
        # self.transformer4 = layers.SpatialTransformer((192,160,160))
        # self.input_model = input_model
        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = MaxPooling(2)
        self.pooling_4 =MaxPooling(4)
        self.pooling_8 =MaxPooling(8)

        self.upsampling = nn.Upsample(scale_factor=2, mode='trilinear') 


        self.encoder1_m = ConvBlock(3,1,list_num[0])  # 权重共享的编码器, 提取moving
        self.encoder2_m = ConvBlock(3,list_num[0],list_num[1])
        self.encoder3_m = ConvBlock(3,list_num[1],list_num[2])
        self.encoder4_m = ConvBlock(3,list_num[2],list_num[3])
        self.encoder5_m = ConvBlock(3,list_num[3],list_num[4])

        
        self.encoder1_f = self.encoder1_m  # 权重共享的编码器, 提取fixed
        self.encoder2_f = self.encoder2_m
        self.encoder3_f = self.encoder3_m
        self.encoder4_f = self.encoder4_m
        self.encoder5_f = self.encoder5_m

        # self.encoder1_f = ConvBlock(3,1,list_num[0])  # 权重共享的编码器, 提取moving
        # self.encoder2_f = ConvBlock(3,list_num[0],list_num[1])
        # self.encoder3_f = ConvBlock(3,list_num[1],list_num[2])
        # self.encoder4_f = ConvBlock(3,list_num[2],list_num[3])
        # self.encoder5_f = ConvBlock(3,list_num[3],list_num[4])

        


        self.decoder1 = ConvBlock(ndims,list_num[4]+list_num[4],32)
        self.decoder2 = ConvBlock(ndims,list_num[6]+list_num[6]+32,32)
        self.decoder3 = ConvBlock(ndims,list_num[8]+list_num[8]+32,32)
        self.decoder4 = ConvBlock(ndims,list_num[10]+list_num[10]+32,16)
        self.decoder5 = ConvBlock(ndims,list_num[0]+list_num[0]+16,16)

        self.output_block = nn.Sequential(ConvBlock(ndims,16,16),ConvBlock(ndims,16,16))
        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(16, ndims, kernel_size=3, padding=1)

        # # init flow layer with small weights and bias
        # self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        # self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        self.output_block_0 = nn.Sequential(ConvBlock(ndims,32,16),ConvBlock(ndims,16,3))
        self.output_block_1 = nn.Sequential(ConvBlock(ndims,32,16),ConvBlock(ndims,16,3))
        self.output_block_2 = nn.Sequential(ConvBlock(ndims,32,16),ConvBlock(ndims,16,3))
        self.output_block_3 = nn.Sequential(ConvBlock(ndims,16,16),ConvBlock(ndims,16,3))

        # 整合特征
        # # ------------------------------Scale4--------------------------------------------
        self.fusion_f_scale4 = ConvBlock(ndims,list_num[0]+list_num[1]+list_num[2]+list_num[3],list_num[5])
        self.fusion_f_scale4_2 = ConvBlock(ndims,list_num[5],list_num[6])


        self.fusion_m_scale4 = self.fusion_f_scale4
        self.fusion_m_scale4_2 = self.fusion_f_scale4_2

        # # ------------------------------Scale3--------------------------------------------
        self.fusion_f_scale3 = ConvBlock(ndims,list_num[0]+list_num[1]+list_num[2]+list_num[3],list_num[7])
        self.fusion_f_scale3_2 = ConvBlock(ndims,list_num[7],list_num[8])


        self.fusion_m_scale3 = self.fusion_f_scale3
        self.fusion_m_scale3_2 = self.fusion_f_scale3_2

        # # ------------------------------Scale2--------------------------------------------
        self.fusion_f_scale2 = ConvBlock(ndims,list_num[0]+list_num[1]+list_num[2]+list_num[3],list_num[9])
        self.fusion_f_scale2_2 = ConvBlock(ndims,list_num[9],list_num[10])

     
        self.fusion_m_scale2 = self.fusion_f_scale2
        self.fusion_m_scale2_2 = self.fusion_f_scale2_2

        
        # self.integrate0 = layers.VecInt((10,12,14), 7)
        # self.integrate1 = layers.VecInt((20,24,28), 7)
        # self.integrate2 = layers.VecInt((40,48,56), 7)
        # self.integrate3 = layers.VecInt((80,96,112), 7)

        # self.integrate0 = layers.VecInt((12,10,12), 7)
        # self.integrate1 = layers.VecInt((24,20,24), 7)
        # self.integrate2 = layers.VecInt((48,40,48), 7)
        # self.integrate3 = layers.VecInt((96,80,96), 7)

        # self.integrate0 = layers.VecInt((12,10,10), 7)
        # self.integrate1 = layers.VecInt((24,20,20), 7)
        # self.integrate2 = layers.VecInt((48,40,40), 7)
        # self.integrate3 = layers.VecInt((96,80,80), 7)

        self.integrate0 = layers.VecInt((8,8,8), 7)
        self.integrate1 = layers.VecInt((16,16,16), 7)
        self.integrate2 = layers.VecInt((32,32,32), 7)
        self.integrate3 = layers.VecInt((64,64,64), 7)
        self.fullsize = layers.ResizeTransform(1 / 2, 3)
        self.resize = layers.ResizeTransform(2, 3)

        
    def forward(self, source, target):
        '''
        Parameters:
            source: Source image tensor.F (moving)
            target: Target image tensor.  (fixed)
            registration: Return transformed image and flow. Default is False.
        '''
        xm = source
        xf = target
        xm_skip_1 = self.encoder1_m(xm) # (1,16,160,192,224)
        xf_skip_1 = self.encoder1_f(xf) # (1,16,160,192,224)

        # moving encoder
        #下采样到1/2,第二个encoder
        xm = self.pooling(xm_skip_1) # (1,16,80,96,112)
        xm_skip_2 = self.encoder2_m(xm) # (1,32,80,96,112)
        # 下采样到1/4,第三个encoder
        xm = self.pooling(xm_skip_2) # (1,32,40,48,56)
        xm_skip_3 = self.encoder3_m(xm) # (1,32,40,48,56)
        # 下采样到1/8,第四个encoder
        xm = self.pooling(xm_skip_3)  # (1,32,20,24,28)
        xm_skip_4 = self.encoder4_m(xm) # (1,32,20,24,28)
        # 下采样到1/16
        xm = self.pooling(xm_skip_4) # (1,32,10,12,14)
        xm = self.encoder5_m(xm) # (1,32,10,12,14)


        # fixed encoder
        #下采样到1/2,第二个encoder
        xf = self.pooling(xf_skip_1)
        xf_skip_2 = self.encoder2_f(xf)
        # 下采样到1/4,第三个encoder
        xf = self.pooling(xf_skip_2)
        xf_skip_3 = self.encoder3_f(xf)
        # 下采样到1/8,第四个encoder
        xf = self.pooling(xf_skip_3)  
        xf_skip_4 = self.encoder4_f(xf)
        # 下采样到1/16
        xf = self.pooling(xf_skip_4)
        xf = self.encoder5_f(xf) # (1,32,10,12,14)

        #-------------------------------Scale5--------------------------------
        # 第一个decoder
        x = torch.cat([xm, xf], dim=1) # (1,64,10,12,14)
        x = self.decoder1(x) # (1,32,10,12,14)


        # ------------------------------Scale4--------------------------------------------
        # 融合多尺度特征：(20,24,28)
        xm_skip_1_fusion4 = self.pooling_8(xm_skip_1) # (1,16,20,24,28)
        # xm_skip_1_fusion4 = self.fusion_m1_scale4(xm_skip_1_fusion4) 
        xm_skip_2_fusion4 = self.pooling_4(xm_skip_2) # (1,32,20,24,28)
        # xm_skip_2_fusion4 = self.fusion_m2_scale4(xm_skip_2_fusion4)
        xm_skip_3_fusion4 = self.pooling(xm_skip_3) # (1,32,20,24,28)
        # xm_skip_3_fusion4 = self.fusion_m3_scale4(xm_skip_3_fusion4)
        xm_skip_4_fusion4 = xm_skip_4 # (1,32,20,24,28)

        xm_fusion4 = torch.cat([xm_skip_1_fusion4, xm_skip_2_fusion4], dim=1)
        xm_fusion4 = torch.cat([xm_fusion4, xm_skip_3_fusion4], dim=1)
        xm_fusion4 = torch.cat([xm_fusion4, xm_skip_4_fusion4], dim=1)

        xm_fusion4 = self.fusion_m_scale4(xm_fusion4) # (1,32,20,24,28)
        xm_fusion4 = self.fusion_m_scale4_2(xm_fusion4) # (1,32,20,24,28)
    

        xf_skip_1_fusion4 = self.pooling_8(xf_skip_1) # (1,16,20,24,28)
        # xf_skip_1_fusion4 = self.fusion_f1_scale4(xf_skip_1_fusion4)
        xf_skip_2_fusion4 = self.pooling_4(xf_skip_2) # (1,32,20,24,28)
        # xf_skip_2_fusion4 = self.fusion_f2_scale4(xf_skip_2_fusion4)
        xf_skip_3_fusion4 = self.pooling(xf_skip_3) # (1,32,20,24,28)
        # xf_skip_3_fusion4 = self.fusion_f3_scale4(xf_skip_3_fusion4)
        xf_skip_4_fusion4 = xf_skip_4 # (1,32,20,24,28)

        xf_fusion4 = torch.cat([xf_skip_1_fusion4, xf_skip_2_fusion4], dim=1) 
        xf_fusion4 = torch.cat([xf_fusion4, xf_skip_3_fusion4], dim=1)
        xf_fusion4 = torch.cat([xf_fusion4, xf_skip_4_fusion4], dim=1)
   
        xf_fusion4 = self.fusion_f_scale4(xf_fusion4)
        xf_fusion4 = self.fusion_f_scale4_2(xf_fusion4) # (1,32,20,24,28) 

        
        x_fusion4 = torch.cat([xm_fusion4, xf_fusion4], dim=1) # (1,64,20,24,28)

    
        # 第二个decoder
        x = self.decoder2(torch.cat([self.upsampling(x), x_fusion4], dim=1)) # (1,32,20,24,28)
        # x = self.trans_3(x)

        # ----输出1/8分辨率的变形场-----
        vec_flow_1 = self.output_block_1(x) # (1,3,20,24,28) 

        # 速度场生成变形场
        # vec_flow_1 = self.resize(vec_flow_1) # (1,3,10,12,14)
        # pos_flow = self.integrate0(vec_flow_1)
        # flow_1 = self.fullsize(pos_flow) # (1,3,20,24,28) 
        flow_1 = self.diff[3](vec_flow_1) # (1,3,20,24,28) 

        flow_1_up = nn.functional.interpolate(flow_1, scale_factor=2,mode="trilinear")*2 # (1,3,40,48,56)


        ## ------------------------------Scale3--------------------------------------------
        # 融合多尺度特征：(40,48,56)
        xm_skip_1_fusion3 = self.pooling_4(xm_skip_1) # (1,16,40,48,56)
        # xm_skip_1_fusion3 = self.fusion_m1_scale3(xm_skip_1_fusion3) 
        xm_skip_2_fusion3 = self.pooling(xm_skip_2) # (1,32,40,48,56)
        # xm_skip_2_fusion3 = self.fusion_m2_scale3(xm_skip_2_fusion3)
        xm_skip_3_fusion3 = xm_skip_3 # (1,32,40,48,56)
        # xm_skip_3_fusion3 = self.fusion_m3_scale3(xm_skip_3_fusion3)

        # xm_skip_4_fusion3 = self.fusion_m4_scale3(xm_skip_4) # (1,32,40,48,56)
        xm_skip_4_fusion3 = nn.functional.interpolate(xm_skip_4, scale_factor=2,mode="trilinear")
        

        xm_fusion3 = torch.cat([xm_skip_1_fusion3, xm_skip_2_fusion3], dim=1)
        xm_fusion3 = torch.cat([xm_fusion3, xm_skip_3_fusion3], dim=1)
        xm_fusion3 = torch.cat([xm_fusion3, xm_skip_4_fusion3], dim=1) # (1,112,40,48,56)

        xm_fusion3 = self.fusion_m_scale3(xm_fusion3) # (1,32,40,48,56)
        xm_fusion3 = self.fusion_m_scale3_2(xm_fusion3) # (1,32,40,48,56)

        xw_fusion3 = self.transformer2(xm_fusion3, flow_1_up)


        xf_skip_1_fusion3 = self.pooling_4(xf_skip_1) # (1,16,40,48,56)
        # xf_skip_1_fusion3 = self.fusion_f1_scale3(xf_skip_1_fusion3) 
        xf_skip_2_fusion3 = self.pooling(xf_skip_2) # (1,32,40,48,56)
        # xf_skip_2_fusion3 = self.fusion_f2_scale3(xf_skip_2_fusion3)
        xf_skip_3_fusion3 = xf_skip_3 # (1,32,40,48,56)
        # xf_skip_3_fusion3 = self.fusion_f3_scale3(xf_skip_3_fusion3)

        # xf_skip_4_fusion3 = self.fusion_f4_scale3(xf_skip_4) # (1,32,40,48,56)
        xf_skip_4_fusion3 = nn.functional.interpolate(xf_skip_4, scale_factor=2,mode="trilinear")
        

        xf_fusion3 = torch.cat([xf_skip_1_fusion3, xf_skip_2_fusion3], dim=1)
        xf_fusion3 = torch.cat([xf_fusion3, xf_skip_3_fusion3], dim=1)
        xf_fusion3 = torch.cat([xf_fusion3, xf_skip_4_fusion3], dim=1) # (1,112,40,48,56)

        xf_fusion3 = self.fusion_f_scale3(xf_fusion3) # (1,32,40,48,56)
        xf_fusion3 = self.fusion_f_scale3_2(xf_fusion3) # (1,32,40,48,56)

        # concat
        x_fusion3 = torch.cat([xw_fusion3, xf_fusion3], dim=1)  # (1,64,40,48,56)

        # 第三个decoder
        x = self.decoder3(torch.cat([self.upsampling(x), x_fusion3], dim=1)) # (1,32,40,48,56)
        # x = self.trans_4(x)

        # ----输出1/4分辨率的变形场---
        delta_vec_flow_2 = self.output_block_2(x) # (1,3,40,48,56)

        # down_delta_vec_flow_2 = self.resize(delta_vec_flow_2) # (1,3,20,24,28)
        # pos_flow = self.integrate1(down_delta_vec_flow_2)
        # delta_flow_2 = self.fullsize(pos_flow) # (1,3,40,48,56)
        delta_flow_2 = self.diff[2](delta_vec_flow_2) # (1,3,40,48,56)

        flow_2 = self.warp[2](flow_1_up, delta_flow_2)+delta_flow_2



        # total_vec_flow_2 = down_delta_vec_flow_2 + nn.functional.interpolate(vec_flow_1,scale_factor=2,mode='trilinear')*2  # (1,3,20,24,28)

        # # 速度场生成变形场
        # pos_flow = self.integrate1(total_vec_flow_2)
        # flow_2 = self.fullsize(pos_flow) # (1,3,40,48,56)

        flow_2_up = nn.functional.interpolate(flow_2, scale_factor=2,mode="trilinear")*2 # (1,3,80,96,112)

        ## ------------------------------Scale2--------------------------------------------
        # 融合多尺度特征：(80,96,112)
        xm_skip_1_fusion2 = self.pooling(xm_skip_1) # (1,16,80,96,112)
        # xm_skip_1_fusion2 = self.fusion_m1_scale2(xm_skip_1_fusion2) 
        xm_skip_2_fusion2 = xm_skip_2 # (1,32,80,96,112)
        # xm_skip_2_fusion2 = self.fusion_m2_scale2(xm_skip_2_fusion2)

        # xm_skip_3_fusion2 = self.fusion_m3_scale2(xm_skip_3)
        xm_skip_3_fusion2 = nn.functional.interpolate(xm_skip_3, scale_factor=2,mode="trilinear")  # (1,32,80,96,112)
        
        # xm_skip_4_fusion2 = self.fusion_m4_scale2(xm_skip_4) # (1,32,80,96,112)
        xm_skip_4_fusion2 = nn.functional.interpolate(xm_skip_4, scale_factor=4,mode="trilinear")
        

        xm_fusion2 = torch.cat([xm_skip_1_fusion2, xm_skip_2_fusion2], dim=1)
        xm_fusion2 = torch.cat([xm_fusion2, xm_skip_3_fusion2], dim=1)
        xm_fusion2 = torch.cat([xm_fusion2, xm_skip_4_fusion2], dim=1) # (1,112,80,96,112)


        xm_fusion2 = self.fusion_m_scale2(xm_fusion2) # (1,32,80,96,112)
        xm_fusion2 = self.fusion_m_scale2_2(xm_fusion2) # (1,32,80,96,112)


        xw_fusion2 = self.transformer3(xm_fusion2, flow_2_up)

        xf_skip_1_fusion2 = self.pooling(xf_skip_1) # (1,16,80,96,112)
        # xf_skip_1_fusion2 = self.fusion_f1_scale2(xf_skip_1_fusion2) 
        xf_skip_2_fusion2 = xf_skip_2 # (1,32,80,96,112)
        # xf_skip_2_fusion2 = self.fusion_f2_scale2(xf_skip_2_fusion2)

        # xf_skip_3_fusion2 = self.fusion_f3_scale2(xf_skip_3)
        xf_skip_3_fusion2 = nn.functional.interpolate(xf_skip_3, scale_factor=2,mode="trilinear")  # (1,32,80,96,112)
        
        # xf_skip_4_fusion2 = self.fusion_f4_scale2(xf_skip_4) # (1,32,80,96,112)
        xf_skip_4_fusion2 = nn.functional.interpolate(xf_skip_4, scale_factor=4,mode="trilinear")
        

        xf_fusion2 = torch.cat([xf_skip_1_fusion2, xf_skip_2_fusion2], dim=1)
        xf_fusion2 = torch.cat([xf_fusion2, xf_skip_3_fusion2], dim=1)
        xf_fusion2 = torch.cat([xf_fusion2, xf_skip_4_fusion2], dim=1) # (1,112,80,96,112)

        xf_fusion2 = self.fusion_f_scale2(xf_fusion2) # (1,32,80,96,112)
        xf_fusion2 = self.fusion_f_scale2_2(xf_fusion2) # (1,32,80,96,112)


        # concat
        x_fusion2 = torch.cat([xw_fusion2, xf_fusion2], dim=1)  # (1,32,80,96,112)


        # 第四个decoder
        x = self.decoder4(torch.cat([self.upsampling(x), x_fusion2], dim=1)) # (1,32,80,96,112)
        # x = self.trans_5(x)
        # print(x.shape)

        # ----输出1/2分辨率的变形场-------
        delta_vec_flow_3 = self.output_block_3(x) # (1,3,80,96,112)
        # down_delta_vec_flow_3 = self.resize(delta_vec_flow_3) # (1,3,40,48,56)
        # pos_flow = self.integrate2(down_delta_vec_flow_3)
        # delta_flow_3 = self.fullsize(pos_flow) # (1,3,80,96,112)
        delta_flow_3 = self.diff[1](delta_vec_flow_3) # (1,3,80,96,112)

        flow_3 = self.warp[1](flow_2_up, delta_flow_3)+delta_flow_3


        # total_vec_flow_3 = down_delta_vec_flow_3 + nn.functional.interpolate(total_vec_flow_2,scale_factor=2,mode='trilinear')*2  # (1,3,40,48,56)

        # # 速度场生成变形场
        # pos_flow = self.integrate2(total_vec_flow_3)
        # flow_3 = self.fullsize(pos_flow) # (1,3,80,96,112)

        flow_3_up = nn.functional.interpolate(flow_3, scale_factor=2,mode="trilinear")*2 # (1,3,160,112,224)

        # ------------------------------Scale1--------------------------------------------
        # # 对moving feature进行warp得到moved feature
        xw_skip_1 = self.transformer4(xm_skip_1, flow_3_up)
        # concat
        x_skip_1 = torch.cat([xw_skip_1, xf_skip_1], dim=1)  # (1,32,160,192,224)
        # print(x_skip_1.shape)


        # 第四个decoder
        x = self.decoder5(torch.cat([self.upsampling(x), x_skip_1], dim=1)) # (1,32,160,192,224)

        # ----------------------输出最终的变形场------------------------------------------
        # output block
        x = self.output_block(x) # (1,16,160,192,224)
        # 生成flow场
        delta_vec_flow_final = self.flow(x) # (1,3,160,192,224)
        # down_delta_vec_flow_final = self.resize(delta_vec_flow_final) # (1,3,80,96,112)
        # pos_flow = self.integrate3(down_delta_vec_flow_final)
        # delta_flow_final = self.fullsize(pos_flow) # (1,3,80,96,112)
        delta_flow_final = self.diff[0](delta_vec_flow_final) # (1,3,160,192,224)

        # total_vec_flow_final = down_delta_vec_flow_final + nn.functional.interpolate(total_vec_flow_3,scale_factor=2,mode='trilinear')*2  # (1,3,80,96,112)

        # # 速度场生成变形场
        # pos_flow = self.integrate3(total_vec_flow_final)
        # flow_final = self.fullsize(pos_flow) # (1,3,80,96,112)
        flow_final = self.warp[0](flow_3_up, delta_flow_final)+delta_flow_final
       

        return flow_1,flow_2,flow_3,flow_final,delta_flow_2,delta_flow_3,delta_flow_final

