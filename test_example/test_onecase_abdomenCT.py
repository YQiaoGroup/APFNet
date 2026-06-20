#!/usr/bin/env python

"""
Example script to train a VoxelMorph model.

You will likely have to customize this script slightly to accommodate your own data. All images
should be appropriately cropped and scaled to values between 0 and 1.

If an atlas file is provided with the --atlas flag, then scan-to-atlas training is performed.
Otherwise, registration will be scan-to-scan.

If you use this code, please cite the following, and read function docs for further info/citations.

    VoxelMorph: A Learning Framework for Deformable Medical Image Registration G. Balakrishnan, A.
    Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. IEEE TMI: Transactions on Medical Imaging. 38(8). pp
    1788-1800. 2019. 

    or

    Unsupervised Learning for Probabilistic Diffeomorphic Registration for Images and Surfaces
    A.V. Dalca, G. Balakrishnan, J. Guttag, M.R. Sabuncu. 
    MedIA: Medical Image Analysis. (57). pp 226-236, 2019 

Copyright 2020 Adrian V. Dalca

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""

import os
import random
import argparse
import time
import math
import numpy as np
import torch
import gc
import torch.nn as nn
import swanlab
import sys
import pandas as pd
from torchvision import transforms
import glob
from torch.utils.data import DataLoader
import ml_collections
from thop import profile
import statistics
import copy
import inspect
import psutil
import functools
import torch.nn.functional as F
import einops

# sys.path.append('/home/boys/project/XMorpher')
# from data import datasets, trans

# project_root = '/home/boys/project/voxelmorph/voxelmorph_code'
# if project_root not in sys.path:
#     sys.path.append(project_root)

# import voxelmorph as vxm

# sys.path.append('/home/boys/project')
# from LapIRN import miccai2020_model_stage


# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters
# parser.add_argument('--img-list', default="/home/boys/project/voxelmorph/voxelmorph_code/sub_abdomenMRCT_list.txt",help='line-seperated list of training files')
parser.add_argument('--seg-atlas', default="./AbdomenMRCT_1041_0001_image.nii.gz",
                    help='seg-atlas filename (default: data/atlas_norm.npz)')
parser.add_argument('--val-t1-list', default="./sub_abdomenMRCT_test_onecase_list.txt",help='line-seperated list of val files')
parser.add_argument('--val-seg-list', default="./sub_abdomenMRCT_test_seg_onecase_list.txt",help='line-seperated list of val files')

# parser.add_argument('--train-seg-list', default="/home/boys/project/voxelmorph/voxelmorph_code/sub_abdomenMRCT_list.txt",help='line-seperated list of val files')
parser.add_argument('--img-prefix', help='optional input image file prefix')
parser.add_argument('--img-suffix', help='optional input image file suffix')
parser.add_argument('--atlas', help='atlas filename (default: data/atlas_norm.npz)')


parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')
parser.add_argument('--exp-name', default='dualVxm',  #
                                    help='Experiment name')  # 保存log和模型对应的实验设置的名字
parser.add_argument('--notes', help='notes for wandb')  # 保存log和模型对应的实验设置的名字

# training parameters
parser.add_argument('--gpu', default='3', help='GPU ID number(s), comma-separated (default: 0)')
parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=1000,
                    help='number of training epochs (default: 1500)')
parser.add_argument('--steps-per-epoch', type=int, default=100,
                    help='frequency of model saves (default: 100)')
parser.add_argument('--load-model', help='optional model file to initialize with')
parser.add_argument('--initial-epoch', type=int, default=0,
                    help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
parser.add_argument('--cudnn-nondet', action='store_true',
                    help='disable cudnn determinism - might slow down training')

# multi_resolution parameters
parser.add_argument('--layer1-epochs', type=int, default=200,
                    help='number of training epochs (default: 1500)')
parser.add_argument('--layer2-epochs', type=int, default=300,
                    help='number of training epochs (default: 1500)')
parser.add_argument('--layer3-epochs', type=int, default=400,
                    help='number of training epochs (default: 1500)')
parser.add_argument('--layer4-epochs', type=int, default=1000,
                    help='number of training epochs (default: 1500)')
parser.add_argument('--layer5-epochs', type=int, default=1000,
                    help='number of training epochs (default: 1500)')
parser.add_argument('--freeze-layer2', type=int, default=0,
                    help='number of training epochs (default: 1500)')
parser.add_argument('--freeze-layer3', type=int, default=0,
                    help='number of training epochs (default: 1500)')
parser.add_argument('--freeze-layer4', type=int, default=0,
                    help='number of training epochs (default: 1500)')
parser.add_argument('--freeze-layer5', type=int, default=0,
                    help='number of training epochs (default: 1500)')

# network architecture parameters
parser.add_argument('--enc', type=int, nargs='+',
                    help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--dec', type=int, nargs='+',
                    help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
parser.add_argument('--int-steps', type=int, default=7,
                    help='number of integration steps (default: 7)')
parser.add_argument('--int-downsize', type=int, default=2,
                    help='flow downsample factor for integration (default: 2)')
parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')

# loss hyperparameters
parser.add_argument('--image_loss', default='mse',
                    help='image reconstruction loss - can be mse or ncc (default: mse)')
parser.add_argument('--lambda', type=float, dest='weight', default=0.01,
                    help='weight of deformation loss (default: 0.01)')
args = parser.parse_args()


bidir = args.bidir


# # no need to append an extra feature axis if data is multichannel
add_feat_axis = not args.multichannel


# # 加载验证集的数据

def read_file_list(filename, prefix=None, suffix=None):
    '''
    Reads a list of files from a line-seperated text file.

    Parameters:
        filename: Filename to load.
        prefix: File prefix. Default is None.
        suffix: File suffix. Default is None.
    '''
    with open(filename, 'r') as file:
        content = file.readlines()
    filelist = [x.strip() for x in content if x.strip()]
    if prefix is not None:
        filelist = [prefix + f for f in filelist]
    if suffix is not None:
        filelist = [f + suffix for f in filelist]
    return filelist


def load_volfile(
    filename,
    np_var='vol',
    add_batch_axis=False,
    add_feat_axis=False,
    pad_shape=None,
    resize_factor=1,
    ret_affine=False
    ):
    """
    Loads a file in nii, nii.gz, mgz, npz, or npy format. If input file is not a string,
    returns it directly (allows files preloaded in memory to be passed to a generator)

    Parameters:
        filename: Filename to load, or preloaded volume to be returned.
        np_var: If the file is a npz (compressed numpy) with multiple variables,
            the desired variable can be specified with np_var. Default is 'vol'.
        add_batch_axis: Adds an axis to the beginning of the array. Default is False.
        add_feat_axis: Adds an axis to the end of the array. Default is False.
        pad_shape: Zero-pad the array to a target shape. Default is None.
        resize: Volume resize factor. Default is 1
        ret_affine: Additionally returns the affine transform (or None if it doesn't exist).
    """
    if isinstance(filename, str) and not os.path.isfile(filename):
        raise ValueError("'%s' is not a file." % filename)

    if not os.path.isfile(filename):
        if ret_affine:
            (vol, affine) = filename
        else:
            vol = filename
    elif filename.endswith(('.nii', '.nii.gz', '.mgz')):
        import nibabel as nib
        img = nib.load(filename)
        # vol = img.get_data().squeeze()
        vol = np.squeeze(img.dataobj)
        affine = img.affine
    elif filename.endswith('.npy'):
        vol = np.load(filename)
        affine = None
    elif filename.endswith('.npz'):
        npz = np.load(filename)
        vol = next(iter(npz.values())) if len(npz.keys()) == 1 else npz[np_var]
        affine = None
    else:
        raise ValueError('unknown filetype for %s' % filename)

    if pad_shape:
        vol, _ = pad(vol, pad_shape)

    if add_feat_axis:
        vol = vol[..., np.newaxis]

    if resize_factor != 1:
        vol = resize(vol, resize_factor)

    if add_batch_axis:
        vol = vol[np.newaxis, ...]

    return (vol, affine) if ret_affine else vol



def seg_volgen(
    vol_names,
    batch_size=1,
    segs=None,
    np_var='vol',
    pad_shape=None,
    resize_factor=1,
    add_feat_axis=True
    ):
    """
    Base generator for random volume loading. Volumes can be passed as a path to
    the parent directory, a glob pattern, a list of file paths, or a list of
    preloaded volumes. Corresponding segmentations are additionally loaded if
    `segs` is provided as a list (of file paths or preloaded segmentations) or set
    to True. If `segs` is True, npz files with variable names 'vol' and 'seg' are
    expected. Passing in preloaded volumes (with optional preloaded segmentations)
    allows volumes preloaded in memory to be passed to a generator.

    Parameters:
        vol_names: Path, glob pattern, list of volume files to load, or list of
            preloaded volumes.
        batch_size: Batch size. Default is 1.
        segs: Loads corresponding segmentations. Default is None.
        np_var: Name of the volume variable if loading npz files. Default is 'vol'.
        pad_shape: Zero-pads loaded volumes to a given shape. Default is None.
        resize_factor: Volume resize factor. Default is 1.
        add_feat_axis: Load volume arrays with added feature axis. Default is True.
    """

    # convert glob path to filenames
    if isinstance(vol_names, str):
        if os.path.isdir(vol_names):
            vol_names = os.path.join(vol_names, '*')
        vol_names = glob.glob(vol_names)

    if isinstance(segs, list) and len(segs) != len(vol_names):
        raise ValueError('Number of image files must match number of seg files.')

    while True:
        # generate [batchsize] random image indices
        # # load volumes and concatenate
        load_params = dict(np_var=np_var, add_batch_axis=True, add_feat_axis=add_feat_axis,
                            pad_shape=pad_shape, resize_factor=resize_factor)

        for i in range(len(vol_names)):
            imgs = [load_volfile(vol_names[i], **load_params)]
            # print(vol_names[i])
            vols = [np.concatenate(imgs, axis=0)]
            yield tuple(vols)


def seg_scan_to_atlas(vol_names, atlas, bidir=False, batch_size=1, no_warp=False, segs=None, **kwargs):
    """
    Generator for scan-to-atlas registration.

    TODO: This could be merged into scan_to_scan() by adding an optional atlas
    argument like in semisupervised().

    Parameters:
        vol_names: List of volume files to load, or list of preloaded volumes.
        atlas: Atlas volume data.
        bidir: Yield input image as output for bidirectional models. Default is False.
        batch_size: Batch size. Default is 1.
        no_warp: Excludes null warp in output list if set to True (for affine training). 
            Default is False.
        segs: Load segmentations as output, for supervised training. Forwarded to the
            internal volgen generator. Default is None.
        kwargs: Forwarded to the internal volgen generator.
    """
    shape = atlas.shape[1:-1]
    zeros = np.zeros((batch_size, *shape, len(shape)))
    atlas = np.repeat(atlas, batch_size, axis=0)
    gen = seg_volgen(vol_names, batch_size=batch_size, segs=segs, **kwargs)
    while True:
        res = next(gen)
        scan = res[0]
        invols = [scan, atlas]
        if not segs:
            outvols = [atlas, scan] if bidir else [atlas]
        else:
            seg = res[1]
            outvols = [seg, scan] if bidir else [seg]
        if not no_warp:
            outvols.append(zeros)
        yield (invols, outvols)




val_t1_files = read_file_list(args.val_t1_list, prefix=args.img_prefix,
                                          suffix=args.img_suffix)
val_seg_files = read_file_list(args.val_seg_list, prefix=args.img_prefix,
                                          suffix=args.img_suffix)                                       
assert len(val_t1_files) > 0, 'Could not find any val data.'

atlas,atlas_affine = load_volfile("./AbdomenMRCT_1041_0001_image.nii.gz", np_var='vol',
                                  add_batch_axis=True, add_feat_axis=add_feat_axis,ret_affine=True)
seg_atlas = load_volfile(args.seg_atlas, np_var='vol',
                                      add_batch_axis=True, add_feat_axis=add_feat_axis)
val_t1_generator = seg_scan_to_atlas(val_t1_files, atlas,
                                             batch_size=args.batch_size, bidir=args.bidir,
                                             add_feat_axis=add_feat_axis)
val_seg_generator = seg_scan_to_atlas(val_seg_files, seg_atlas,
                                             batch_size=args.batch_size, bidir=args.bidir,
                                             add_feat_axis=add_feat_axis)

# extract shape from sampled input
inshape = (192,160,192)



# device handling
gpus = args.gpu.split(',')
nb_gpus = len(gpus)
device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

torch.backends.cudnn.deterministic = not args.cudnn_nondet


class LoadableModel(nn.Module):
    """
    Base class for easy pytorch model loading without having to manually
    specify the architecture configuration at load time.

    We can cache the arguments used to the construct the initial network, so that
    we can construct the exact same network when loading from file. The arguments
    provided to __init__ are automatically saved into the object (in self.config)
    if the __init__ method is decorated with the @store_config_args utility.
    """

    # this constructor just functions as a check to make sure that every
    # LoadableModel subclass has provided an internal config parameter
    # either manually or via store_config_args
    def __init__(self, *args, **kwargs):
        if not hasattr(self, 'config'):
            raise RuntimeError('models that inherit from LoadableModel must decorate the '
                               'constructor with @store_config_args')
        super().__init__(*args, **kwargs)

    def save(self, path):
        """
        Saves the model configuration and weights to a pytorch file.
        """
        # don't save the transformer_grid buffers - see SpatialTransformer doc for more info
        sd = self.state_dict().copy()
        grid_buffers = [key for key in sd.keys() if key.endswith('.grid')]
        for key in grid_buffers:
            sd.pop(key)
        torch.save({'config': self.config, 'model_state': sd}, path)

    @classmethod
    def load(cls, path, device):
        """
        Load a python model configuration and weights.
        """
        checkpoint = torch.load(path, map_location=torch.device(device))
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['model_state'], strict=False)
        return model



def store_config_args(func):
    """
    Class-method decorator that saves every argument provided to the
    function as a dictionary in 'self.config'. This is used to assist
    model loading - see LoadableModel.
    """

    attrs, varargs, varkw, defaults = inspect.getargspec(func)
    # attrs, varargs, varkw, defaults = inspect.getfullargspec(func)


    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        self.config = {}

        # first save the default values
        if defaults:
            for attr, val in zip(reversed(attrs), reversed(defaults)):
                self.config[attr] = val

        # next handle positional args
        for attr, val in zip(attrs[1:], args):
            self.config[attr] = val

        # lastly handle keyword args
        if kwargs:
            for attr, val in kwargs.items():
                self.config[attr] = val

        return func(self, *args, **kwargs)
    return wrapper

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


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
        self.transformer2 = SpatialTransformer((48,40,48))
        self.transformer3 = SpatialTransformer((96,80,96))
        self.transformer4 = SpatialTransformer((192,160,192))
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
                 inshape=(192,160,192),
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

        
        self.transformer1 = SpatialTransformer(tuple(d // 8 for d in inshape))
        self.transformer2 = SpatialTransformer(tuple(d // 4 for d in inshape))
        self.transformer3 = SpatialTransformer(tuple(d // 2 for d in inshape))
        self.transformer4 = SpatialTransformer(inshape)

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



def compute_dice_coefficient(mask_gt, mask_pred,dim=4):
    """Computes soerensen-dice coefficient.

    compute the soerensen-dice coefficient between the ground truth mask `mask_gt`
    and the predicted mask `mask_pred`.

    Args:
    mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
    mask_pred: 3-dim Numpy array of type bool. The predicted mask.

    Returns:
    the dice coeffcient as float. If both masks are empty, the result is NaN.
    """
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2*volume_intersect / volume_sum

def compute_dice(s1, s2,dim=2):
    dice = 0
    count = 0
    dice_region = []
    
    # liver/ CBCT
    # value = [1.0]
    
    # IXI
    # value = [2.0, 3.0, 4.0, 5.0, 7.0, 8.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 24.0, 26.0, 28.0, 30.0, 31.0, 41.0, 42.0, 43.0, 44.0, 46.0, 47.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 58.0, 60.0, 62.0, 63.0, 77.0, 85.0, 251.0, 252.0, 253.0, 254.0, 255.0]

    # Mind101
    # value = [2.0, 4.0, 5.0, 7.0, 8.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 24.0, 26.0, 28.0, 30.0, 31.0, 41.0, 43.0, 44.0, 46.0, 47.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 58.0, 60.0, 62.0, 63.0, 77.0, 80.0, 85.0, 251.0, 252.0, 253.0, 254.0, 255.0]

    # IBSR18
    # value = [2.0, 8.0, 9.0, 24.0, 30.0]

    # BTCV
    # value = [1.0, 2.0, 3.0, 6.0, 7.0]
    
    # CUMC12
    # value = [97,53,35,57,67,25,15,29,12]
    # LPBA40
    # value = [21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 101.0, 102.0, 121.0, 122.0, 161.0, 162.0, 163.0, 164.0, 165.0, 166.0, 181.0, 182.0]
    # MGH10
    # value = [4.0, 6.0, 7.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 26.0, 28.0, 29.0, 30.0, 31.0, 33.0, 35.0, 36.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 55.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 104.0, 106.0, 107.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0, 122.0, 126.0, 128.0, 129.0, 130.0, 131.0, 133.0, 135.0, 136.0, 138.0, 139.0, 140.0, 141.0, 142.0, 143.0, 144.0, 145.0, 146.0, 147.0, 148.0, 149.0, 150.0, 151.0, 152.0, 155.0, 158.0, 159.0, 160.0, 161.0, 162.0, 163.0, 164.0, 165.0, 166.0, 167.0, 168.0, 169.0]
    # value = [4,5,6,10,35]
    # for label in value:
    for label in range(1, dim):
        Dice = compute_dice_coefficient((s1==label), (s2==label))
        dice += Dice
        count += 1
        # Dice = Dice.cpu().numpy()
        dice_region.append(Dice)
    dice /= count
    return dice,dice_region

def save_volfile(array, filename, affine=None):
    """
    Saves an array to nii, nii.gz, or npz format.

    Parameters:
        array: The array to save.
        filename: Filename to save to.
        affine: Affine vox-to-ras matrix. Saves LIA matrix if None (default).
    """
    if filename.endswith(('.nii', '.nii.gz')):
        import nibabel as nib
        if affine is None and array.ndim >= 3:
            # use LIA transform as default affine
            affine = np.array([[-1, 0, 0, 0],  # nopep8
                               [0, 0, 1, 0],  # nopep8
                               [0, -1, 0, 0],  # nopep8
                               [0, 0, 0, 1]], dtype=float)  # nopep8
            pcrs = np.append(np.array(array.shape[:3]) / 2, 1)
            affine[:3, 3] = -np.matmul(affine, pcrs)[:3]
        nib.save(nib.Nifti1Image(array, affine), filename)
    elif filename.endswith('.npz'):
        np.savez_compressed(filename, vol=array)
    else:
        raise ValueError('unknown filetype for %s' % filename)


def test_MedNext():
    print("test_MedNext_layer4")


    # # # ###########  GDp：APFNet-trans  ##################################
    # model = dual_pyramid_VxmDense_Trans_FFM_normal_adaptive_val(list_num=[7,12,2,2,32,32,32,17,13,16,14])
    # checkpoint = torch.load("./daul_pyramid_PFNet_FFM_normal_GDP_adaptive_trans.pth")
    # model.to(device)
    # model_dict = model.state_dict()
    # state_dict = {k:v for k,v in checkpoint.items() if k in model_dict.keys()}
    # model_dict.update(state_dict)
    # model.load_state_dict(model_dict)

      ###########  GDp：APFNet-huge withpretrain ##################################
    model = dual_pyramid_VxmDense_FFM_huge_adaptive_val(list_num=[8,15,4,2,96,96,96,64,55,49,42])
    checkpoint = torch.load("./daul_pyramid_PDFNet_FFM_huge_GDP_adaptive_withpretrain_7618.pth")
    model.to(device)
    model_dict = model.state_dict()
    state_dict = {k:v for k,v in checkpoint.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

    #  ------------------------------------------------------------------------------------
    # Calculate FLOPs and Parameters:vm,autofuse
    dummy_input = torch.randn(1, 1, 192, 160, 192).to(device)
    flops, params = profile(model, (dummy_input, dummy_input,))
    flops_m = flops / 1000000.0
    params_m = params / 1000000.0
    print('flops: %.2f M, params: %.2f M' % (flops_m, params_m))


    # ################# PDFNet-trans ################################
    case_results = []
    inference_times = []
    cases_to_save = [] # 在这里填入你想用来画图的 Case 编号
    save_all = True

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    num = 0

    print("--------- Validation Dice ---------")
    for step_outer in range(len(val_t1_files)):
        # Extract moving data
        inputs_moving, y_true = next(val_t1_generator)
        inputs_moving = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs_moving]
        
        # Extract moving seg data
        seg_inputs_moving, seg_y_true = next(val_seg_generator)
        seg_inputs_moving = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in seg_inputs_moving]
        
        for step_inner in range(len(val_t1_files)):
            # Extract fixed data
            inputs_fixed, y_true = next(val_t1_generator)
            inputs_fixed = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs_fixed]
            
            # Extract fixed seg data
            seg_inputs_fixed, seg_y_true = next(val_seg_generator)
            seg_inputs_fixed = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in seg_inputs_fixed]
            
            if step_inner < len(val_t1_files) - 1:
                with torch.no_grad():
                    starter.record()
            

                    # PDFNet-large/PDFNet-huge
                    flow_1,flow_2,flow_3,flow,delta_flow_2,delta_flow_3,delta_flow_final,features_xf,features_xm = model(inputs_moving[0], inputs_fixed[0])

                    # PDFNet/PDFnet-diff/PDFNet-trans
                    # flow_1,flow_2,flow_3,flow,delta_flow_2,delta_flow_3,delta_flow_final = model(inputs_moving[0], inputs_fixed[0])

        
                    ender.record()
                    torch.cuda.synchronize() # Synchronize GPU time
                    
                    curr_time = starter.elapsed_time(ender)
                    print(curr_time)
                    inference_times.append(curr_time)
                    
                    # Generate moved image
                    SPT = SpatialTransformer((192, 160, 192), mode="nearest").to(device)
                    seg_pre = SPT(seg_inputs_moving[0], flow)
                    
                    # Calculate Dice
                    dice, dice_region = compute_dice(seg_pre, seg_inputs_fixed[0])
                    # Ensure metrics are standard python floats for DataFrame storage
                    dice_val = dice.mean().item() if isinstance(dice, torch.Tensor) else dice

                    if save_all or (num in cases_to_save):
                        # 动态获取当前图像的空间维度 (D, H, W)，彻底告别写死 (160,192,224)
                        _, _, D, H, W = inputs_moving[0].shape
                        
                        # 重新定义双线性插值的 SPT，用于生成平滑的 moved_image 用于展示
                        SPT_bilinear = SpatialTransformer((D, H, W), mode="bilinear").to(device)
                        moved_image = SPT_bilinear(inputs_moving[0], flow)
                        
                        # 基础保存路径
                        base_visual_dir = "./save_results"
                        
                        # 1. 保存原图与配准图 (动态拼接文件名，例如 RDP_case038_moved.nii.gz)
                        img_moved = moved_image.detach().cpu().numpy().squeeze()
                        save_volfile(img_moved, f"{base_visual_dir}/PFNet_trans_case{num:03d}_moved.nii.gz", affine=atlas_affine)
                        
                        img_moving = inputs_moving[0].detach().cpu().numpy().squeeze()
                        save_volfile(img_moving, f"{base_visual_dir}/case{num:03d}_moving.nii.gz", affine=atlas_affine)
                        
                        img_fixed = inputs_fixed[0].detach().cpu().numpy().squeeze()
                        save_volfile(img_fixed, f"{base_visual_dir}/case{num:03d}_fixed.nii.gz", affine=atlas_affine)

                        # 2. 保存分割标签 (seg_pre 是之前用 nearest 生成的，直接拿来用)
                        seg_mvd = seg_pre.detach().cpu().numpy().squeeze()
                        save_volfile(seg_mvd, f"{base_visual_dir}/PFNet_trans_case{num:03d}_moved_seg.nii.gz", affine=atlas_affine)
                        
                        seg_mov = seg_inputs_moving[0].detach().cpu().numpy().squeeze()
                        save_volfile(seg_mov, f"{base_visual_dir}/case{num:03d}_moving_seg.nii.gz", affine=atlas_affine)
                        
                        seg_fix = seg_inputs_fixed[0].detach().cpu().numpy().squeeze()
                        save_volfile(seg_fix, f"{base_visual_dir}/case{num:03d}_fixed_seg.nii.gz", affine=atlas_affine)
                        
                        # 3. 保存形变场 (极其优雅的写法)
                        # flow 形状通常是 [1, 3, D, H, W]，我们需要转成 [D, H, W, 3]
                        # 直接使用 permute 一步到位，彻底取代你之前逐个通道赋值的笨重写法
                        # whole_field = flow[0].permute(1, 2, 3, 0).detach().cpu().numpy()
                        # vxm.py.utils.save_volfile(whole_field, f"{base_visual_dir}/RDP_case{num:03d}_field.nii.gz", affine=atlas_affine)
                        
                        print(f"  📸 Case {num:03d} 的可视化数据已成功保存！")

                    
                    if dice_val < 1:
                        print(f"Case {num} - Dice: {dice_val:.4f}")
                        case_results.append({
                            'Case': num,
                            'Dice': dice_val
                        })
                        num += 1

        print("---------- Change fixed image ---------")



if __name__ == "__main__":
    test_MedNext()
