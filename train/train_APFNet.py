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
import swanlab
import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
from thop import profile
import statistics
import copy
import psutil

# import voxelmorph with pytorch backend
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm  # nopep8




# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters
parser.add_argument('--img-list', default="./sub_abdomenMRCT_list.txt",help='line-seperated list of training files')
parser.add_argument('--seg-atlas', default="./imagesTr/AbdomenMRCT_0001_0001.nii.gz",
                    help='seg-atlas filename (default: data/atlas_norm.npz)')
parser.add_argument('--val-t1-list', default="./sub_abdomenMRCT_test_list.txt",help='line-seperated list of val files')
parser.add_argument('--val-seg-list', default="./sub_abdomenMRCT_test_seg_list.txt",help='line-seperated list of val files')

parser.add_argument('--train-seg-list', default="./sub_abdomenMRCT_list.txt",help='line-seperated list of val files')
parser.add_argument('--img-prefix', help='optional input image file prefix')
parser.add_argument('--img-suffix', help='optional input image file suffix')
parser.add_argument('--atlas', help='atlas filename (default: data/atlas_norm.npz)')

parser.add_argument('--model-dir', default='./result',
                    help='model output directory (default: models)')

parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')
parser.add_argument('--exp-name', default='dualVxm',  # 
                                    help='Experiment name')  # 保存log和模型对应的实验设置的名字
parser.add_argument('--notes', help='notes for wandb')  # 保存log和模型对应的实验设置的名字
                    
# training parameters
parser.add_argument('--gpu', default='7', help='GPU ID number(s), comma-separated (default: 0)')
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
parser.add_argument('--image_loss', default='ncc',
                    help='image reconstruction loss - can be mse or ncc (default: mse)')
parser.add_argument('--lambda', type=float, dest='weight', default=1,
                    help='weight of deformation loss (default: 0.01)')
args = parser.parse_args()


bidir = args.bidir

# # recorder
# recorder = make_recorder("Log/", args.lr, args.weight, args.image_loss)

# 加载训练数据
train_files = vxm.py.utils.read_file_list(args.img_list, prefix=args.img_prefix,
                                          suffix=args.img_suffix)
train_seg_files = vxm.py.utils.read_file_list(args.train_seg_list, prefix=args.img_prefix,
                                          suffix=args.img_suffix)
assert len(train_files) > 0, 'Could not find any training data.'

# no need to append an extra feature axis if data is multichannel
add_feat_axis = not args.multichannel

if args.atlas:
    # scan-to-atlas generator
    atlas = vxm.py.utils.load_volfile(args.atlas, np_var='vol',
                                      add_batch_axis=True, add_feat_axis=add_feat_axis)
    seg_atlas = vxm.py.utils.load_volfile(args.seg_atlas, np_var='vol',
                                      add_batch_axis=True, add_feat_axis=add_feat_axis)
    generator = vxm.generators.scan_to_atlas(train_files, atlas,
                                             batch_size=args.batch_size, bidir=args.bidir,
                                             add_feat_axis=add_feat_axis)
    train_generator = vxm.generators.seg_scan_to_atlas(train_files, atlas,
                                             batch_size=args.batch_size, bidir=args.bidir,
                                             add_feat_axis=add_feat_axis)
    train_seg_generator = vxm.generators.seg_scan_to_atlas(train_seg_files, seg_atlas,
                                             batch_size=args.batch_size, bidir=args.bidir,
                                             add_feat_axis=add_feat_axis)
else:
    # scan-to-scan generator
    generator = vxm.generators.scan_to_scan(
        train_files, batch_size=args.batch_size, bidir=args.bidir, add_feat_axis=add_feat_axis)
    seg_generator = vxm.generators.semisupervised(
    train_files,train_seg_files,batch_size=args.batch_size)

# # 加载验证集的数据

val_t1_files = vxm.py.utils.read_file_list(args.val_t1_list, prefix=args.img_prefix,
                                          suffix=args.img_suffix)
val_seg_files = vxm.py.utils.read_file_list(args.val_seg_list, prefix=args.img_prefix,
                                          suffix=args.img_suffix)                                       
assert len(val_t1_files) > 0, 'Could not find any val data.'

atlas,atlas_affine = vxm.py.utils.load_volfile("./imagesTr/AbdomenMRCT_0001_0001.nii.gz", np_var='vol',
                                  add_batch_axis=True, add_feat_axis=add_feat_axis,ret_affine=True)
seg_atlas = vxm.py.utils.load_volfile(args.seg_atlas, np_var='vol',
                                      add_batch_axis=True, add_feat_axis=add_feat_axis)
val_t1_generator = vxm.generators.seg_scan_to_atlas(val_t1_files, atlas,
                                             batch_size=args.batch_size, bidir=args.bidir,
                                             add_feat_axis=add_feat_axis)
val_seg_generator = vxm.generators.seg_scan_to_atlas(val_seg_files, seg_atlas,
                                             batch_size=args.batch_size, bidir=args.bidir,
                                             add_feat_axis=add_feat_axis)

# extract shape from sampled input
inshape = next(generator)[0][0].shape[1:-1]

# prepare model folder
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)

# device handling
gpus = args.gpu.split(',')
nb_gpus = len(gpus)
device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# assert np.mod(args.batch_size, nb_gpus) == 0, \
#     'Batch size (%d) should be a multiple of the nr of gpus (%d)' % (args.batch_size, nb_devices)

# enabling cudnn determinism appears to speed up training by a lot
torch.backends.cudnn.deterministic = not args.cudnn_nondet



# prepare image loss
if args.image_loss == 'ncc':
    image_loss_func = vxm.losses.NCC().loss
elif args.image_loss == 'mse':
    image_loss_func = vxm.losses.MSE().loss
elif args.image_loss == 'MI':
    image_loss_func = vxm.losses.MutualInformation()
elif args.image_loss == 'MIND':
    image_loss_func = vxm.losses.MINDLoss().loss
else:
    raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)

# need two image loss functions if bidirectional
if bidir:
    losses = [image_loss_func, image_loss_func]
    weights = [0.5, 0.5]
else:
    losses = [image_loss_func]
    weights = [1]

# prepare deformation loss
# losses += [vxm.losses.Grad3d(penalty='l2')]
losses += [vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss]
weights += [args.weight]

loss_mse = [vxm.losses.MSE().loss]

config = dict(
    threshold=0.4)

def set_random_seed(seed):
    """
    设置随机种子，以确保实验结果的可重复性。
    """
    random.seed(seed)  # 设置 Python random 模块的种子
    np.random.seed(seed)  # 设置 NumPy 随机种子
    torch.manual_seed(seed)  # 设置 PyTorch CPU 随机种子
    torch.cuda.manual_seed(seed)  # 设置 PyTorch GPU 随机种子
    torch.cuda.manual_seed_all(seed)  # 设置所有 GPU 的随机种子
    torch.backends.cudnn.deterministic = True  # 确保 CuDNN 给出确定性结果
    torch.backends.cudnn.benchmark = False  # 禁用 CuDNN 自动调优（对固定输入尺寸有效）

# ------------------------------ 剪枝-------------------------------------------


# 近端算子 针对全局 normal model
def proximal_operator_normal(alpha_parameters, learning_rate, Lambda,non_zero_alpha,theta=2):
    beta1 = 8 # 4096
    beta2 = 7 # 512
    beta3 = 6 # 64
    beta4 = 5 # 8
    # 实现 L1 正则项的软阈值操作作为近端算子
    # non_zero_alpha = [alpha_1,alpha_2,alpha_3,alpha_4,alpha_5,alpha_6_1,alpha_6_2,alpha_7_1,alpha_7_2,alpha_8_1,alpha_8_2]
    with torch.no_grad():  # 不追踪梯度
        for index,alpha in enumerate(alpha_parameters):
            if index == 0: #alpha_1
                # print(alpha.data)
                lambda_prox = learning_rate*Lambda*(beta1 + beta1*1 + beta2*non_zero_alpha[1] + beta2*non_zero_alpha[9] + beta3*non_zero_alpha[7] + beta4*non_zero_alpha[5])
                alpha.data = torch.sign(alpha.data) * torch.clamp(torch.abs(alpha.data) - lambda_prox, min=0)
                formatted_alpha = ", ".join("{:.7f}".format(x) for x in alpha.data.flatten())
                print("alpha_1:", formatted_alpha)
                print("lambda_prox_alpha_1: {:.7f}".format(lambda_prox))
            if index == 1: #alpha_2
                # print(alpha.data)
                lambda_prox = learning_rate*Lambda*(beta2 + beta2*non_zero_alpha[0] + beta3*non_zero_alpha[2] + beta2*non_zero_alpha[9] + beta3*non_zero_alpha[7] + beta4*non_zero_alpha[5])
                alpha.data = torch.sign(alpha.data) * torch.clamp(torch.abs(alpha.data) - lambda_prox, min=0)
                formatted_alpha = ", ".join("{:.7f}".format(x) for x in alpha.data.flatten())
                print("alpha_2:", formatted_alpha)
                print("lambda_prox_alpha_2: {:.7f}".format(lambda_prox))
            if index == 2: #alpha_3
                # print(alpha.data)
                if non_zero_alpha[2] >= 2:
                    lambda_prox = learning_rate*Lambda*(beta3 + beta3*non_zero_alpha[1] + beta4*non_zero_alpha[3] + beta2*non_zero_alpha[9] + beta3*non_zero_alpha[7] + beta4*non_zero_alpha[5])
                else:
                    lambda_prox = 0
                alpha.data = torch.sign(alpha.data) * torch.clamp(torch.abs(alpha.data) - lambda_prox, min=0)
                formatted_alpha = ", ".join("{:.7f}".format(x) for x in alpha.data.flatten())
                print("alpha_3:", formatted_alpha)
                print("lambda_prox_alpha_3: {:.7f}".format(lambda_prox))
            if index == 3: #alpha_4
                # print(alpha.data)
                if non_zero_alpha[3] >= 2:
                    lambda_prox = learning_rate*Lambda*(beta4 + beta4*non_zero_alpha[2] + non_zero_alpha[4] + beta2*non_zero_alpha[9] + beta3*non_zero_alpha[7] + beta4*non_zero_alpha[5])
                else:
                    lambda_prox = 0
                alpha.data = torch.sign(alpha.data) * torch.clamp(torch.abs(alpha.data) - lambda_prox, min=0)
                formatted_alpha = ", ".join("{:.7f}".format(x) for x in alpha.data.flatten())
                print("alpha_4:", formatted_alpha)
                print("lambda_prox_alpha_4: {:.7f}".format(lambda_prox))
            if index == 4: #alpha_5
                # print(alpha.data)
                lambda_prox = learning_rate*Lambda*(1 + non_zero_alpha[3])
                alpha.data = torch.sign(alpha.data) * torch.clamp(torch.abs(alpha.data) - lambda_prox, min=0)
                formatted_alpha = ", ".join("{:.7f}".format(x) for x in alpha.data.flatten())
                print("alpha_5:", formatted_alpha)
                print("lambda_prox_alpha_5: {:.7f}".format(lambda_prox))
            if index == 5: #alpha_6_1
                # print(alpha.data)
                lambda_prox = 0
                alpha.data = torch.sign(alpha.data) * torch.clamp(torch.abs(alpha.data) - lambda_prox, min=0)
                formatted_alpha = ", ".join("{:.7f}".format(x) for x in alpha.data.flatten())
                print("alpha_6_1:", formatted_alpha)
                print("lambda_prox_alpha_6_1: {:.7f}".format(lambda_prox))
            if index == 6: #alpha_6_2
                # print(alpha.data)
                lambda_prox = 0
                alpha.data = torch.sign(alpha.data) * torch.clamp(torch.abs(alpha.data) - lambda_prox, min=0)
                formatted_alpha = ", ".join("{:.7f}".format(x) for x in alpha.data.flatten())
                print("alpha_6_2:", formatted_alpha)
                print("lambda_prox_alpha_6_2: {:.7f}".format(lambda_prox))
            if index == 7: #alpha_7_1
                if non_zero_alpha[7] >= 16:
                    lambda_prox = learning_rate*Lambda*(beta3 + beta3*(non_zero_alpha[0]+non_zero_alpha[1]+non_zero_alpha[2]+non_zero_alpha[3])+beta3*non_zero_alpha[8])
                else:
                    lambda_prox = 0
                # print(alpha.data)
                # lambda_prox = 0
                alpha.data = torch.sign(alpha.data) * torch.clamp(torch.abs(alpha.data) - lambda_prox, min=0)
                formatted_alpha = ", ".join("{:.7f}".format(x) for x in alpha.data.flatten())
                print("alpha_7_1:", formatted_alpha)
                print("lambda_prox_alpha_7_1: {:.7f}".format(lambda_prox))
            if index == 8: #alpha_7_2
                if non_zero_alpha[8] >= 16:
                    lambda_prox = learning_rate*Lambda*(beta3 + beta3*(non_zero_alpha[7])+beta3*32) + 0.0000200
                else:
                    lambda_prox = 0
                # print(alpha.data)
                # lambda_prox = 0
                alpha.data = torch.sign(alpha.data) * torch.clamp(torch.abs(alpha.data) - lambda_prox, min=0) 
                formatted_alpha = ", ".join("{:.7f}".format(x) for x in alpha.data.flatten())
                print("alpha_7_2:", formatted_alpha)
                print("lambda_prox_alpha_7_2: {:.7f}".format(lambda_prox))
            if index == 9: #alpha_8_1
                if non_zero_alpha[9] >= 16:
                    lambda_prox = learning_rate*Lambda*(beta2 + beta2*(non_zero_alpha[0]+non_zero_alpha[1]+non_zero_alpha[2]+non_zero_alpha[3])+beta2*non_zero_alpha[10]) 
                else:
                    lambda_prox = 0
                # print(alpha.data)
                alpha.data = torch.sign(alpha.data) * torch.clamp(torch.abs(alpha.data) - lambda_prox, min=0)
                formatted_alpha = ", ".join("{:.7f}".format(x) for x in alpha.data.flatten())
                print("alpha_8_1:", formatted_alpha)
                print("lambda_prox_alpha_8_1: {:.7f}".format(lambda_prox))
            if index == 10: #alpha_8_2
                if non_zero_alpha[10] >= 16:
                    lambda_prox = learning_rate*Lambda*(beta2 + beta2*(non_zero_alpha[9])+beta2*32) + 0.0000200
                else:
                    lambda_prox = 0
                # print(alpha.data)
                alpha.data = torch.sign(alpha.data) * torch.clamp(torch.abs(alpha.data) - lambda_prox, min=0)
                formatted_alpha = ", ".join("{:.7f}".format(x) for x in alpha.data.flatten())
                print("alpha_8_2:", formatted_alpha)
                print("lambda_prox_alpha_8_2: {:.7f}".format(lambda_prox))
            
    return alpha_parameters

# 近端算子 针对全局 large model
def proximal_operator_large(alpha_parameters, learning_rate, Lambda,non_zero_alpha,theta=2):
    beta1 = 8 # 4096
    beta2 = 7 # 512
    beta3 = 6 # 64
    beta4 = 5 # 8
    # 实现 L1 正则项的软阈值操作作为近端算子
    # non_zero_alpha = [alpha_1,alpha_2,alpha_3,alpha_4,alpha_5,alpha_6_1,alpha_6_2,alpha_7_1,alpha_7_2,alpha_8_1,alpha_8_2]
    with torch.no_grad():  # 不追踪梯度
        for index,alpha in enumerate(alpha_parameters):
            if index == 0: #alpha_1
                # print(alpha.data)
                lambda_prox = learning_rate*Lambda*(beta1 + beta1*1 + beta2*non_zero_alpha[1] + beta2*non_zero_alpha[9] + beta3*non_zero_alpha[7] + beta4*non_zero_alpha[5])
                alpha.data = torch.sign(alpha.data) * torch.clamp(torch.abs(alpha.data) - lambda_prox, min=0)
                formatted_alpha = ", ".join("{:.7f}".format(x) for x in alpha.data.flatten())
                print("alpha_1:", formatted_alpha)
                print("lambda_prox_alpha_1: {:.7f}".format(lambda_prox))
            if index == 1: #alpha_2
                # print(alpha.data)
                lambda_prox = learning_rate*Lambda*(beta2 + beta2*non_zero_alpha[0] + beta3*non_zero_alpha[2] + beta2*non_zero_alpha[9] + beta3*non_zero_alpha[7] + beta4*non_zero_alpha[5])
                alpha.data = torch.sign(alpha.data) * torch.clamp(torch.abs(alpha.data) - lambda_prox, min=0)
                formatted_alpha = ", ".join("{:.7f}".format(x) for x in alpha.data.flatten())
                print("alpha_2:", formatted_alpha)
                print("lambda_prox_alpha_2: {:.7f}".format(lambda_prox))
            if index == 2: #alpha_3
                # print(alpha.data)
                if non_zero_alpha[2] >= 2:
                    lambda_prox = learning_rate*Lambda*(beta3 + beta3*non_zero_alpha[1] + beta4*non_zero_alpha[3] + beta2*non_zero_alpha[9] + beta3*non_zero_alpha[7] + beta4*non_zero_alpha[5])
                else:
                    lambda_prox = 0
                alpha.data = torch.sign(alpha.data) * torch.clamp(torch.abs(alpha.data) - lambda_prox, min=0)
                formatted_alpha = ", ".join("{:.7f}".format(x) for x in alpha.data.flatten())
                print("alpha_3:", formatted_alpha)
                print("lambda_prox_alpha_3: {:.7f}".format(lambda_prox))
            if index == 3: #alpha_4
                # print(alpha.data)
                if non_zero_alpha[3] >= 2:
                    lambda_prox = learning_rate*Lambda*(beta4 + beta4*non_zero_alpha[2] + non_zero_alpha[4] + beta2*non_zero_alpha[9] + beta3*non_zero_alpha[7] + beta4*non_zero_alpha[5])
                else:
                    lambda_prox = 0
                alpha.data = torch.sign(alpha.data) * torch.clamp(torch.abs(alpha.data) - lambda_prox, min=0)
                formatted_alpha = ", ".join("{:.7f}".format(x) for x in alpha.data.flatten())
                print("alpha_4:", formatted_alpha)
                print("lambda_prox_alpha_4: {:.7f}".format(lambda_prox))
            if index == 4: #alpha_5
                # print(alpha.data)
                lambda_prox = learning_rate*Lambda*(1 + non_zero_alpha[3])
                alpha.data = torch.sign(alpha.data) * torch.clamp(torch.abs(alpha.data) - lambda_prox, min=0)
                formatted_alpha = ", ".join("{:.7f}".format(x) for x in alpha.data.flatten())
                print("alpha_5:", formatted_alpha)
                print("lambda_prox_alpha_5: {:.7f}".format(lambda_prox))
            if index == 5: #alpha_6_1
                # print(alpha.data)
                lambda_prox = 0
                alpha.data = torch.sign(alpha.data) * torch.clamp(torch.abs(alpha.data) - lambda_prox, min=0)
                formatted_alpha = ", ".join("{:.7f}".format(x) for x in alpha.data.flatten())
                print("alpha_6_1:", formatted_alpha)
                print("lambda_prox_alpha_6_1: {:.7f}".format(lambda_prox))
            if index == 6: #alpha_6_2
                # print(alpha.data)
                lambda_prox = 0
                alpha.data = torch.sign(alpha.data) * torch.clamp(torch.abs(alpha.data) - lambda_prox, min=0)
                formatted_alpha = ", ".join("{:.7f}".format(x) for x in alpha.data.flatten())
                print("alpha_6_2:", formatted_alpha)
                print("lambda_prox_alpha_6_2: {:.7f}".format(lambda_prox))
            if index == 7: #alpha_7_1
                if non_zero_alpha[7] >= 32:
                    lambda_prox = learning_rate*Lambda*(beta3 + beta3*(non_zero_alpha[0]+non_zero_alpha[1]+non_zero_alpha[2]+non_zero_alpha[3])+beta3*non_zero_alpha[8])
                else:
                    lambda_prox = 0
                # print(alpha.data)
                # lambda_prox = 0
                alpha.data = torch.sign(alpha.data) * torch.clamp(torch.abs(alpha.data) - lambda_prox, min=0)
                formatted_alpha = ", ".join("{:.7f}".format(x) for x in alpha.data.flatten())
                print("alpha_7_1:", formatted_alpha)
                print("lambda_prox_alpha_7_1: {:.7f}".format(lambda_prox))
            if index == 8: #alpha_7_2
                if non_zero_alpha[8] >= 32:
                    lambda_prox = learning_rate*Lambda*(beta3 + beta3*(non_zero_alpha[7])+beta3*32) + 0.0000200
                else:
                    lambda_prox = 0
                # print(alpha.data)
                # lambda_prox = 0
                alpha.data = torch.sign(alpha.data) * torch.clamp(torch.abs(alpha.data) - lambda_prox, min=0) 
                formatted_alpha = ", ".join("{:.7f}".format(x) for x in alpha.data.flatten())
                print("alpha_7_2:", formatted_alpha)
                print("lambda_prox_alpha_7_2: {:.7f}".format(lambda_prox))
            if index == 9: #alpha_8_1
                if non_zero_alpha[9] >= 32:
                    lambda_prox = learning_rate*Lambda*(beta2 + beta2*(non_zero_alpha[0]+non_zero_alpha[1]+non_zero_alpha[2]+non_zero_alpha[3])+beta2*non_zero_alpha[10]) 
                else:
                    lambda_prox = 0
                # print(alpha.data)
                alpha.data = torch.sign(alpha.data) * torch.clamp(torch.abs(alpha.data) - lambda_prox, min=0)
                formatted_alpha = ", ".join("{:.7f}".format(x) for x in alpha.data.flatten())
                print("alpha_8_1:", formatted_alpha)
                print("lambda_prox_alpha_8_1: {:.7f}".format(lambda_prox))
            if index == 10: #alpha_8_2
                if non_zero_alpha[10] >= 32:
                    lambda_prox = learning_rate*Lambda*(beta2 + beta2*(non_zero_alpha[9])+beta2*32) + 0.0000200
                else:
                    lambda_prox = 0
                # print(alpha.data)
                alpha.data = torch.sign(alpha.data) * torch.clamp(torch.abs(alpha.data) - lambda_prox, min=0)
                formatted_alpha = ", ".join("{:.7f}".format(x) for x in alpha.data.flatten())
                print("alpha_8_2:", formatted_alpha)
                print("lambda_prox_alpha_8_2: {:.7f}".format(lambda_prox))
            
    return alpha_parameters

# 近端算子 针对全局
def proximal_operator_huge(alpha_parameters, learning_rate, Lambda, non_zero_alpha, theta=2):
    beta1 = theta ^ 4  # 4096
    beta2 = theta ^ 3  # 512
    beta3 = theta ^ 2  # 64
    beta4 = theta ^ 1  # 8
    # 实现 L1 正则项的软阈值操作作为近端算子
    # non_zero_alpha = [alpha_1,alpha_2,alpha_3,alpha_4,alpha_5,alpha_6_1,alpha_6_2,alpha_7_1,alpha_7_2,alpha_8_1,alpha_8_2]
    with torch.no_grad():  # 不追踪梯度
        for index, alpha in enumerate(alpha_parameters):
            if index == 0:  # alpha_1
                # print(alpha.data)
                lambda_prox = learning_rate * Lambda * (
                            beta1 + beta1 * 1 + beta2 * non_zero_alpha[1] + beta2 * non_zero_alpha[9] + beta3 *
                            non_zero_alpha[7] + beta4 * non_zero_alpha[5])
                alpha.data = torch.sign(alpha.data) * torch.clamp(torch.abs(alpha.data) - lambda_prox, min=0)
                formatted_alpha = ", ".join("{:.7f}".format(x) for x in alpha.data.flatten())
                print("alpha_1:", formatted_alpha)
                print("lambda_prox_alpha_1: {:.7f}".format(lambda_prox))
            if index == 1:  # alpha_2
                # print(alpha.data)
                lambda_prox = learning_rate * Lambda * (
                            beta2 + beta2 * non_zero_alpha[0] + beta3 * non_zero_alpha[2] + beta2 * non_zero_alpha[
                        9] + beta3 * non_zero_alpha[7] + beta4 * non_zero_alpha[5])
                alpha.data = torch.sign(alpha.data) * torch.clamp(torch.abs(alpha.data) - lambda_prox, min=0)
                formatted_alpha = ", ".join("{:.7f}".format(x) for x in alpha.data.flatten())
                print("alpha_2:", formatted_alpha)
                print("lambda_prox_alpha_2: {:.7f}".format(lambda_prox))
            if index == 2:  # alpha_3
                # print(alpha.data)
                if non_zero_alpha[2] >= 2:
                    lambda_prox = learning_rate * Lambda * (
                                beta3 + beta3 * non_zero_alpha[1] + beta4 * non_zero_alpha[3] + beta2 * non_zero_alpha[
                            9] + beta3 * non_zero_alpha[7] + beta4 * non_zero_alpha[5])
                else:
                    lambda_prox = 0
                alpha.data = torch.sign(alpha.data) * torch.clamp(torch.abs(alpha.data) - lambda_prox, min=0)
                formatted_alpha = ", ".join("{:.7f}".format(x) for x in alpha.data.flatten())
                print("alpha_3:", formatted_alpha)
                print("lambda_prox_alpha_3: {:.7f}".format(lambda_prox))
            if index == 3:  # alpha_4
                # print(alpha.data)
                if non_zero_alpha[3] >= 2:
                    lambda_prox = learning_rate * Lambda * (
                                beta4 + beta4 * non_zero_alpha[2] + non_zero_alpha[4] + beta2 * non_zero_alpha[
                            9] + beta3 * non_zero_alpha[7] + beta4 * non_zero_alpha[5])
                else:
                    lambda_prox = 0
                alpha.data = torch.sign(alpha.data) * torch.clamp(torch.abs(alpha.data) - lambda_prox, min=0)
                formatted_alpha = ", ".join("{:.7f}".format(x) for x in alpha.data.flatten())
                print("alpha_4:", formatted_alpha)
                print("lambda_prox_alpha_4: {:.7f}".format(lambda_prox))
            if index == 4:  # alpha_5
                # print(alpha.data)
                lambda_prox = learning_rate * Lambda * (1 + non_zero_alpha[3])
                alpha.data = torch.sign(alpha.data) * torch.clamp(torch.abs(alpha.data) - lambda_prox, min=0)
                formatted_alpha = ", ".join("{:.7f}".format(x) for x in alpha.data.flatten())
                print("alpha_5:", formatted_alpha)
                print("lambda_prox_alpha_5: {:.7f}".format(lambda_prox))
            if index == 5:  # alpha_6_1
                # print(alpha.data)
                lambda_prox = 0
                alpha.data = torch.sign(alpha.data) * torch.clamp(torch.abs(alpha.data) - lambda_prox, min=0)
                formatted_alpha = ", ".join("{:.7f}".format(x) for x in alpha.data.flatten())
                print("alpha_6_1:", formatted_alpha)
                print("lambda_prox_alpha_6_1: {:.7f}".format(lambda_prox))
            if index == 6:  # alpha_6_2
                # print(alpha.data)
                lambda_prox = 0
                alpha.data = torch.sign(alpha.data) * torch.clamp(torch.abs(alpha.data) - lambda_prox, min=0)
                formatted_alpha = ", ".join("{:.7f}".format(x) for x in alpha.data.flatten())
                print("alpha_6_2:", formatted_alpha)
                print("lambda_prox_alpha_6_2: {:.7f}".format(lambda_prox))
            if index == 7:  # alpha_7_1
                if non_zero_alpha[7] >= 64:
                    lambda_prox = learning_rate * Lambda * (beta3 + beta3 * (
                                non_zero_alpha[0] + non_zero_alpha[1] + non_zero_alpha[2] + non_zero_alpha[3]) + beta3 *
                                                            non_zero_alpha[8])
                else:
                    lambda_prox = 0
                # print(alpha.data)
                # lambda_prox = 0
                alpha.data = torch.sign(alpha.data) * torch.clamp(torch.abs(alpha.data) - lambda_prox, min=0)
                formatted_alpha = ", ".join("{:.7f}".format(x) for x in alpha.data.flatten())
                print("alpha_7_1:", formatted_alpha)
                print("lambda_prox_alpha_7_1: {:.7f}".format(lambda_prox))
            if index == 8:  # alpha_7_2
                if non_zero_alpha[8] >= 64:
                    lambda_prox = learning_rate * Lambda * (
                                beta3 + beta3 * (non_zero_alpha[7]) + beta3 * 32) + 0.0000200
                else:
                    lambda_prox = 0
                # print(alpha.data)
                # lambda_prox = 0
                alpha.data = torch.sign(alpha.data) * torch.clamp(torch.abs(alpha.data) - lambda_prox, min=0)
                formatted_alpha = ", ".join("{:.7f}".format(x) for x in alpha.data.flatten())
                print("alpha_7_2:", formatted_alpha)
                print("lambda_prox_alpha_7_2: {:.7f}".format(lambda_prox))
            if index == 9:  # alpha_8_1
                if non_zero_alpha[9] >= 48:
                    lambda_prox = learning_rate * Lambda * (beta2 + beta2 * (
                                non_zero_alpha[0] + non_zero_alpha[1] + non_zero_alpha[2] + non_zero_alpha[3]) + beta2 *
                                                            non_zero_alpha[10])
                else:
                    lambda_prox = 0
                # print(alpha.data)
                alpha.data = torch.sign(alpha.data) * torch.clamp(torch.abs(alpha.data) - lambda_prox, min=0)
                formatted_alpha = ", ".join("{:.7f}".format(x) for x in alpha.data.flatten())
                print("alpha_8_1:", formatted_alpha)
                print("lambda_prox_alpha_8_1: {:.7f}".format(lambda_prox))
            if index == 10:  # alpha_8_2
                if non_zero_alpha[10] >= 48:
                    lambda_prox = learning_rate * Lambda * (
                                beta2 + beta2 * (non_zero_alpha[9]) + beta2 * 32) + 0.0000200
                else:
                    lambda_prox = 0
                # print(alpha.data)
                alpha.data = torch.sign(alpha.data) * torch.clamp(torch.abs(alpha.data) - lambda_prox, min=0)
                formatted_alpha = ", ".join("{:.7f}".format(x) for x in alpha.data.flatten())
                print("alpha_8_2:", formatted_alpha)
                print("lambda_prox_alpha_8_2: {:.7f}".format(lambda_prox))

    return alpha_parameters


# loss function
def nondifferentiable_loss(alpha,theta=2,threshold=1e-9):
    alpha_1 = torch.count_nonzero(alpha[0].abs() > threshold)
    alpha_2 = torch.count_nonzero(alpha[1].abs() > threshold)
    alpha_3 = torch.count_nonzero(alpha[2].abs() > threshold)
    alpha_4 = torch.count_nonzero(alpha[3].abs() > threshold)
    alpha_5 = torch.count_nonzero(alpha[4].abs() > threshold)
    alpha_6_1 = torch.count_nonzero(alpha[5].abs() > threshold)
    alpha_6_2 = torch.count_nonzero(alpha[6].abs() > threshold)
    alpha_7_1 = torch.count_nonzero(alpha[7].abs() > threshold)
    alpha_7_2 = torch.count_nonzero(alpha[8].abs() > threshold)
    alpha_8_1 = torch.count_nonzero(alpha[9].abs() > threshold)
    alpha_8_2 = torch.count_nonzero(alpha[10].abs() > threshold)
    beta1 = theta^4
    beta2 = theta^3
    beta3 = theta^2
    beta4 = theta^1
    R1 = beta1*alpha_1 + beta2*(alpha_2 + alpha_8_1 + alpha_8_2) + beta3*(alpha_3 + alpha_7_1 + alpha_7_2) + beta4*(alpha_4 + alpha_6_1 + alpha_6_2) + alpha_5
    R2 = beta1*alpha_1 + beta2*(alpha_1*alpha_2 + alpha_8_1*(alpha_8_2 + alpha_1 + alpha_2 + alpha_3 + alpha_4)) + beta3*(alpha_2*alpha_3 + alpha_7_1*(alpha_7_2 + alpha_1 + alpha_2 + alpha_3 + alpha_4)) + beta4*(alpha_3*alpha_4 + alpha_6_1*(alpha_6_2 + alpha_1 + alpha_2 + alpha_3 + alpha_4)) + alpha_4*alpha_5
    R = R1 + R2
    non_zero_alpha = [alpha_1,alpha_2,alpha_3,alpha_4,alpha_5,alpha_6_1,alpha_6_2,alpha_7_1,alpha_7_2,alpha_8_1,alpha_8_2]
    return R,non_zero_alpha

# ------------------- 不同尺寸的模型--------------------------
def train_dual_pyramid_vxm_trans_FFM_4layer_normal_GDP(config=config):
    # 设置固定的随机种子
    set_random_seed(42)  # 你可以将42替换为任何其他整数种子值

    run = swanlab.init(project="APFNet", config=config)
        
    print("Training dual_pyramid_vxm_FFM_4layer_trans")
    

    '''注意检查是否需要多卡模型并行训练'''
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    device1 = torch.device('cuda:0')  # 分别表示用os.environ里面的第一个和第二个编号的gpu
    

    
    # 准备model_layer4
    newshape = (192,160,192)    
    model = vxm.networks.dual_pyramid_VxmDense_Trans_FFM_normal_GDP()
    # 传递mednext的参数

    model.to(device1)
    model.train()



    # # 冻结相应层
    # for n,param in model_layer4.named_parameters():
    #     if "enc_block_3" in n or "dec_block_3" in n or "down_3" in n or "up_3" in n or "bottleneck" in n or "enc_block_2" in n or "dec_block_2" in n or "down_2" in n or "up_2" in n or "enc_block_1" in n or "dec_block_1" in n or "down_1" in n or "up_1" in n:
    #         param.requires_grad = False

    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,model_layer4.parameters()), lr=args.lr)
    # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad,model_layer4.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    # 验证集
    val_interval = 5
    best_metric = -1
    decay = 0.5

    # for epoch in range(args.layer4_epochs):
    for epoch in range(1000):

        epoch_loss = []
        epoch_total_loss = []
        epoch_total_loss_2 = []
        epoch_step_time = []
        epoch_total_diceloss = []
        decay = decay*0.98

        print(decay)
        print("epoch",epoch)
        model.set_decay(decay)


        for step in range(100):


            step_start_time = time.time()
            # recorder.step += 1

            # generate inputs (and true outputs) and convert them to tensors
            inputs, y_true = next(seg_generator)
            inputs = [torch.from_numpy(d).to(device1).float().permute(0, 4, 1, 2, 3) for d in inputs]
            y_true = [torch.from_numpy(d).to(device1).float().permute(0, 4, 1, 2, 3) for d in y_true]
    
            flow_1,flow_2,flow_3,y_flow,delta_flow_2,delta_flow_3,delta_flow_final,ori_alpha,gate_alpha = model(inputs[0],inputs[1])
            
            

            # 对moving图像进行变换得到moved image
            # 定义空间转换层
            SPT = vxm.torch.layers.SpatialTransformer(newshape,mode="bilinear").to(device1)
            y_pre = SPT(inputs[0], y_flow)
            y_pred = []
            y_pred.append(y_pre)
            y_pred.append(y_flow)

            # calculate total loss
            loss = 0
            loss_list = []
            for n, loss_function in enumerate(losses):
                if n==2:
                    '''对应的是雅克比行列式的情况, 自己添加的'''
                    pred_flow = y_pred[1]  # y_pred的最后一个是预测的变形场flow, shape=[B,C,W,H,D]
                    pred_flow = pred_flow.permute(0, 3, 2, 4, 1)  # 为了适应雅克比行列式的输入shape=[B,H,W,D,C]
                    curr_loss = loss_function(pred_flow)
                else:
                    curr_loss = loss_function(inputs[1], y_pred[n]) * weights[n]
                    loss_list.append(curr_loss.item())
                    loss += curr_loss

            # loss_list.append(curr_loss.item())
            # loss += curr_loss


            print(f"loss: {loss_list[0] + loss_list[1]:.4f}  sim_loss: {loss_list[0]:.4f}  reg_loss: {loss_list[1]}")


            epoch_loss.append(loss_list)
            epoch_total_loss.append(loss.item())

            # backpropagate and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            # get compute time
            epoch_step_time.append(time.time() - step_start_time)

            theta = 4
            non_diff_loss_value,non_zero_alpha = nondifferentiable_loss(ori_alpha,theta=theta)
            # print("reg loss in alpha",non_diff_loss_value)
            # print("non_zero_alpha_num",non_zero_alpha)
            # print("ori_alpha",ori_alpha)
            

            if (step + 1) % 1 == 0:
                # 近端映射步骤来处理不可微部分
                input_alpha = [model.alpha_1, model.alpha_2, model.alpha_3, model.alpha_4, model.alpha_5, model.alpha_6_1, model.alpha_6_2, model.alpha_7_1, model.alpha_7_2, model.alpha_8_1,model.alpha_8_2]
                learning_rate = 0.00001
                Lambda = 0.0036
                proximal_operator_normal(input_alpha, learning_rate, Lambda,non_zero_alpha,theta=theta)
                # print("model_alpha_1",model.alpha_1)
            
        print("ori_alpha",ori_alpha)
        print("non_zero_alpha_num",non_zero_alpha)
        print("gate_alpha",gate_alpha)

        # print epoch info
        # scheduler.step()
        epoch_info = 'Epoch %d/%d' % (epoch + 1, args.epochs)
        time_info = '%.4f sec/step' % np.mean(epoch_step_time)
        losses_info = ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
        loss_info = 'loss: %.4e  (%s)' % (np.mean(epoch_total_loss), losses_info)
        print(' - '.join((epoch_info, time_info, loss_info)), flush=True)
        swanlab.log({"total_loss": np.mean(epoch_total_loss), 'epoch':epoch})

        # # 对验证集进行测试

        if (epoch + 1) % val_interval == 0:
            name = "daul_pyramid_vxm_FFM_4layer_4_trans_normal_GDP_Lambda0.0036_theta4_decay0.5_decayrate0.98_nopretrain_abdomenct_" + str(epoch) + ".pth"
            Dice = 0
            mean_dice = 0
            num = 0
            model.eval()
            print("---------验证集的Dice----------")
            SPT = vxm.torch.layers.SpatialTransformer(inshape, mode="nearest").to(device1)
            for step in range(len(val_t1_files)):
                # 提取t1数据
                inputs_fixed, y_true = next(val_t1_generator)
                inputs_fixed = [torch.from_numpy(d).to(device1).float().permute(0, 4, 1, 2, 3) for d in inputs_fixed]
                # 提取对应t1的seg数据
                seg_inputs_fixed, seg_y_true = next(val_seg_generator)
                seg_inputs_fixed = [torch.from_numpy(d).to(device1).float().permute(0, 4, 1, 2, 3) for d in seg_inputs_fixed]
                for step in range(len(val_t1_files)):
                    # 提取t1数据
                    inputs_moving, y_true = next(val_t1_generator)
                    inputs_moving = [torch.from_numpy(d).to(device1).float().permute(0, 4, 1, 2, 3) for d in inputs_moving]
                    # 提取对应t1的seg数据
                    seg_inputs_moving, seg_y_true = next(val_seg_generator)
                    seg_inputs_moving = [torch.from_numpy(d).to(device1).float().permute(0, 4, 1, 2, 3) for d in seg_inputs_moving]
                    if step < len(val_t1_files) - 1:
                        with torch.no_grad():
                            # _,layer1_flow,_,_ = model_layer1(inputs_moving[0], inputs_fixed[0])
                            flow_1,flow_2,flow_3,flow,delta_flow_2,delta_flow_3,delta_flow_final,ori_alpha,gate_alpha = model(inputs_moving[0],inputs_fixed[0])
                            seg_pre = SPT(seg_inputs_moving[0], flow)
                            dice,dice_region = vxm.torch.losses.compute_dice(seg_pre, seg_inputs_fixed[0])
                            if dice < 1:
                                print("dice", dice)
                                Dice += dice
                                # print(Dice)
                                num += 1
                print("----------换fixed image---------")
        
            metric = Dice / num
            swanlab.log({"mean_dice": metric, 'epoch':epoch})

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1

            torch.save(model.state_dict(),os.path.join(model_dir, name))
                # torch.save(model_layer2.state_dict(),os.path.join(model_dir, 'best_model_CRNet_3layer.pth'))
                # torch.save(model,os.path.join(model_dir, 'best_model_MedNext.pt'))
                # print("saved new best metric MedNext layer4")

            print(f"Current epoch: {epoch+1} current dice in val_data: {metric} ")
            print(f"Best dice: {best_metric} at epoch {best_metric_epoch}")
            swanlab.log({"best_mean_dice": best_metric, 'epoch':epoch})
        torch.cuda.empty_cache()
    print("layer4 训练完成！")


def train_dual_pyramid_vxm_FFM_4layer_normal_GDP(config=config):
    # 设置固定的随机种子
    set_random_seed(42)  # 你可以将42替换为任何其他整数种子值

    run = swanlab.init(project="APFNet", config=config)
        
    print("Training dual_pyramid_vxm_FFM_4layer_large")
    

    '''注意检查是否需要多卡模型并行训练'''
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    device1 = torch.device('cuda:0')  # 分别表示用os.environ里面的第一个和第二个编号的gpu
    

    
    # 准备model_layer4
    newshape = (192,160,192)    
    model = vxm.networks.dual_pyramid_VxmDense_FFM_normal_GDP()
    # 传递mednext的参数
    model_dict = model.state_dict()
    checkpoint = torch.load("./dual_pyramid_vxm_FFM_4layer_normal.pth")
    state_dict = {k:v for k,v in checkpoint.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

    model.to(device1)
    model.train()



    # # 冻结相应层
    # for n,param in model_layer4.named_parameters():
    #     if "enc_block_3" in n or "dec_block_3" in n or "down_3" in n or "up_3" in n or "bottleneck" in n or "enc_block_2" in n or "dec_block_2" in n or "down_2" in n or "up_2" in n or "enc_block_1" in n or "dec_block_1" in n or "down_1" in n or "up_1" in n:
    #         param.requires_grad = False

    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,model_layer4.parameters()), lr=args.lr)
    # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad,model_layer4.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    # 验证集
    val_interval = 5
    best_metric = -1
    decay = 0.5

    # for epoch in range(args.layer4_epochs):
    for epoch in range(1000):

        epoch_loss = []
        epoch_total_loss = []
        epoch_total_loss_2 = []
        epoch_step_time = []
        epoch_total_diceloss = []
        decay = decay*0.98

        print(decay)
        print("epoch",epoch)
        model.set_decay(decay)


        for step in range(100):


            step_start_time = time.time()
            # recorder.step += 1

            # generate inputs (and true outputs) and convert them to tensors
            inputs, y_true = next(seg_generator)
            inputs = [torch.from_numpy(d).to(device1).float().permute(0, 4, 1, 2, 3) for d in inputs]
            y_true = [torch.from_numpy(d).to(device1).float().permute(0, 4, 1, 2, 3) for d in y_true]
    
            flow_1,flow_2,flow_3,y_flow,delta_flow_2,delta_flow_3,delta_flow_final,ori_alpha,gate_alpha = model(inputs[0],inputs[1])
            
            

            # 对moving图像进行变换得到moved image
            # 定义空间转换层
            SPT = vxm.torch.layers.SpatialTransformer(newshape,mode="bilinear").to(device1)
            y_pre = SPT(inputs[0], y_flow)
            y_pred = []
            y_pred.append(y_pre)
            y_pred.append(y_flow)

            # calculate total loss
            loss = 0
            loss_list = []
            for n, loss_function in enumerate(losses):
                if n==2:
                    '''对应的是雅克比行列式的情况, 自己添加的'''
                    pred_flow = y_pred[1]  # y_pred的最后一个是预测的变形场flow, shape=[B,C,W,H,D]
                    pred_flow = pred_flow.permute(0, 3, 2, 4, 1)  # 为了适应雅克比行列式的输入shape=[B,H,W,D,C]
                    curr_loss = loss_function(pred_flow)
                else:
                    curr_loss = loss_function(inputs[1], y_pred[n]) * weights[n]
                    loss_list.append(curr_loss.item())
                    loss += curr_loss

            # loss_list.append(curr_loss.item())
            # loss += curr_loss


            print(f"loss: {loss_list[0] + loss_list[1]:.4f}  sim_loss: {loss_list[0]:.4f}  reg_loss: {loss_list[1]}")


            epoch_loss.append(loss_list)
            epoch_total_loss.append(loss.item())

            # backpropagate and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            # get compute time
            epoch_step_time.append(time.time() - step_start_time)

            theta = 4
            non_diff_loss_value,non_zero_alpha = nondifferentiable_loss(ori_alpha,theta=theta)
            # print("reg loss in alpha",non_diff_loss_value)
            # print("non_zero_alpha_num",non_zero_alpha)
            # print("ori_alpha",ori_alpha)
            

            if (step + 1) % 1 == 0:
                # 近端映射步骤来处理不可微部分
                input_alpha = [model.alpha_1, model.alpha_2, model.alpha_3, model.alpha_4, model.alpha_5, model.alpha_6_1, model.alpha_6_2, model.alpha_7_1, model.alpha_7_2, model.alpha_8_1,model.alpha_8_2]
                learning_rate = 0.00001
                Lambda = 0.0036
                proximal_operator_normal(input_alpha, learning_rate, Lambda,non_zero_alpha,theta=theta)
                # print("model_alpha_1",model.alpha_1)
            
        print("ori_alpha",ori_alpha)
        print("non_zero_alpha_num",non_zero_alpha)
        print("gate_alpha",gate_alpha)

        # print epoch info
        # scheduler.step()
        epoch_info = 'Epoch %d/%d' % (epoch + 1, args.epochs)
        time_info = '%.4f sec/step' % np.mean(epoch_step_time)
        losses_info = ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
        loss_info = 'loss: %.4e  (%s)' % (np.mean(epoch_total_loss), losses_info)
        print(' - '.join((epoch_info, time_info, loss_info)), flush=True)
        swanlab.log({"total_loss": np.mean(epoch_total_loss), 'epoch':epoch})

        # # 对验证集进行测试

        if (epoch + 1) % val_interval == 0:
            name = "daul_pyramid_vxm_FFM_4layer_4_normal_GDP_Lambda0.0036_theta4_decay0.5_decayrate0.98_withpretrain_abdomenmrct_" + str(epoch) + ".pth"
            Dice = 0
            mean_dice = 0
            num = 0
            model.eval()
            print("---------验证集的Dice----------")
            SPT = vxm.torch.layers.SpatialTransformer(inshape, mode="nearest").to(device1)
            for step in range(len(val_t1_files)):
                # 提取t1数据
                inputs_fixed, y_true = next(val_t1_generator)
                inputs_fixed = [torch.from_numpy(d).to(device1).float().permute(0, 4, 1, 2, 3) for d in inputs_fixed]
                # 提取对应t1的seg数据
                seg_inputs_fixed, seg_y_true = next(val_seg_generator)
                seg_inputs_fixed = [torch.from_numpy(d).to(device1).float().permute(0, 4, 1, 2, 3) for d in seg_inputs_fixed]
                for step in range(len(val_t1_files)):
                    # 提取t1数据
                    inputs_moving, y_true = next(val_t1_generator)
                    inputs_moving = [torch.from_numpy(d).to(device1).float().permute(0, 4, 1, 2, 3) for d in inputs_moving]
                    # 提取对应t1的seg数据
                    seg_inputs_moving, seg_y_true = next(val_seg_generator)
                    seg_inputs_moving = [torch.from_numpy(d).to(device1).float().permute(0, 4, 1, 2, 3) for d in seg_inputs_moving]
                    if step < len(val_t1_files) - 1:
                        with torch.no_grad():
                            # _,layer1_flow,_,_ = model_layer1(inputs_moving[0], inputs_fixed[0])
                            flow_1,flow_2,flow_3,flow,delta_flow_2,delta_flow_3,delta_flow_final,ori_alpha,gate_alpha = model(inputs_moving[0],inputs_fixed[0])
                            seg_pre = SPT(seg_inputs_moving[0], flow)
                            dice,dice_region = vxm.torch.losses.compute_dice(seg_pre, seg_inputs_fixed[0])
                            if dice < 1:
                                print("dice", dice)
                                Dice += dice
                                # print(Dice)
                                num += 1
                print("----------换fixed image---------")

        
            metric = Dice / num
            swanlab.log({"mean_dice": metric, 'epoch':epoch})

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1

            torch.save(model.state_dict(),os.path.join(model_dir, name))
                # torch.save(model_layer2.state_dict(),os.path.join(model_dir, 'best_model_CRNet_3layer.pth'))
                # torch.save(model,os.path.join(model_dir, 'best_model_MedNext.pt'))
            print("saved new best model")

            print(f"Current epoch: {epoch+1} current dice in val_data: {metric} ")
            print(f"Best dice: {best_metric} at epoch {best_metric_epoch}")
            swanlab.log({"best_mean_dice": best_metric, 'epoch':epoch})
        torch.cuda.empty_cache()
    print("layer4 训练完成！")


def train_dual_pyramid_vxm_FFM_4layer_large_GDP(config=config):
    # 设置固定的随机种子
    set_random_seed(42)  # 你可以将42替换为任何其他整数种子值

    run = swanlab.init(project="APFNet", config=config)
        
    print("Training dual_pyramid_vxm_FFM_4layer_large")
    

    '''注意检查是否需要多卡模型并行训练'''
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    device1 = torch.device('cuda:0')  # 分别表示用os.environ里面的第一个和第二个编号的gpu
    

    
    # 准备model_layer4
    newshape = (192,160,192)    
    model = vxm.networks.dual_pyramid_VxmDense_FFM_large_GDP()
    # 传递mednext的参数
    model_dict = model.state_dict()
    checkpoint = torch.load("./dual_pyramid_vxm_FFM_4layer_large.pth")
    state_dict = {k:v for k,v in checkpoint.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

    model.to(device1)
    model.train()



    # # 冻结相应层
    # for n,param in model_layer4.named_parameters():
    #     if "enc_block_3" in n or "dec_block_3" in n or "down_3" in n or "up_3" in n or "bottleneck" in n or "enc_block_2" in n or "dec_block_2" in n or "down_2" in n or "up_2" in n or "enc_block_1" in n or "dec_block_1" in n or "down_1" in n or "up_1" in n:
    #         param.requires_grad = False

    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,model_layer4.parameters()), lr=args.lr)
    # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad,model_layer4.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    # 验证集
    val_interval = 5
    best_metric = -1
    decay = 0.5

    # for epoch in range(args.layer4_epochs):
    for epoch in range(1000):

        epoch_loss = []
        epoch_total_loss = []
        epoch_total_loss_2 = []
        epoch_step_time = []
        epoch_total_diceloss = []
        decay = decay*0.98

        print(decay)
        print("epoch",epoch)
        model.set_decay(decay)


        for step in range(100):


            step_start_time = time.time()
            # recorder.step += 1

            # generate inputs (and true outputs) and convert them to tensors
            inputs, y_true = next(seg_generator)
            inputs = [torch.from_numpy(d).to(device1).float().permute(0, 4, 1, 2, 3) for d in inputs]
            y_true = [torch.from_numpy(d).to(device1).float().permute(0, 4, 1, 2, 3) for d in y_true]
    
            flow_1,flow_2,flow_3,y_flow,delta_flow_2,delta_flow_3,delta_flow_final,ori_alpha,gate_alpha = model(inputs[0],inputs[1])
            
            

            # 对moving图像进行变换得到moved image
            # 定义空间转换层
            SPT = vxm.torch.layers.SpatialTransformer(newshape,mode="bilinear").to(device1)
            y_pre = SPT(inputs[0], y_flow)
            y_pred = []
            y_pred.append(y_pre)
            y_pred.append(y_flow)

            # calculate total loss
            loss = 0
            loss_list = []
            for n, loss_function in enumerate(losses):
                if n==2:
                    '''对应的是雅克比行列式的情况, 自己添加的'''
                    pred_flow = y_pred[1]  # y_pred的最后一个是预测的变形场flow, shape=[B,C,W,H,D]
                    pred_flow = pred_flow.permute(0, 3, 2, 4, 1)  # 为了适应雅克比行列式的输入shape=[B,H,W,D,C]
                    curr_loss = loss_function(pred_flow)
                else:
                    curr_loss = loss_function(inputs[1], y_pred[n]) * weights[n]
                    loss_list.append(curr_loss.item())
                    loss += curr_loss

            # loss_list.append(curr_loss.item())
            # loss += curr_loss


            print(f"loss: {loss_list[0] + loss_list[1]:.4f}  sim_loss: {loss_list[0]:.4f}  reg_loss: {loss_list[1]}")


            epoch_loss.append(loss_list)
            epoch_total_loss.append(loss.item())

            # backpropagate and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            # get compute time
            epoch_step_time.append(time.time() - step_start_time)

            theta = 4
            non_diff_loss_value,non_zero_alpha = nondifferentiable_loss(ori_alpha,theta=theta)
            # print("reg loss in alpha",non_diff_loss_value)
            # print("non_zero_alpha_num",non_zero_alpha)
            # print("ori_alpha",ori_alpha)
            

            if (step + 1) % 1 == 0:
                # 近端映射步骤来处理不可微部分
                input_alpha = [model.alpha_1, model.alpha_2, model.alpha_3, model.alpha_4, model.alpha_5, model.alpha_6_1, model.alpha_6_2, model.alpha_7_1, model.alpha_7_2, model.alpha_8_1,model.alpha_8_2]
                learning_rate = 0.00001
                Lambda = 0.0018
                proximal_operator_large(input_alpha, learning_rate, Lambda,non_zero_alpha,theta=theta)
                # print("model_alpha_1",model.alpha_1)
            
        print("ori_alpha",ori_alpha)
        print("non_zero_alpha_num",non_zero_alpha)
        print("gate_alpha",gate_alpha)

        # print epoch info
        # scheduler.step()
        epoch_info = 'Epoch %d/%d' % (epoch + 1, args.epochs)
        time_info = '%.4f sec/step' % np.mean(epoch_step_time)
        losses_info = ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
        loss_info = 'loss: %.4e  (%s)' % (np.mean(epoch_total_loss), losses_info)
        print(' - '.join((epoch_info, time_info, loss_info)), flush=True)
        swanlab.log({"total_loss": np.mean(epoch_total_loss), 'epoch':epoch})

        # # 对验证集进行测试

        if (epoch + 1) % val_interval == 0:
            name = "daul_pyramid_vxm_FFM_4layer_4_large_GDP_Lambda0.0018_theta4_decay0.5_decayrate0.98_withpretrain_abdomenmrct_quarter_" + str(epoch) + ".pth"
            Dice = 0
            mean_dice = 0
            num = 0
            model.eval()
            print("---------验证集的Dice----------")
            SPT = vxm.torch.layers.SpatialTransformer(inshape, mode="nearest").to(device1)
            for step in range(len(val_t1_files)):
                # 提取t1数据
                inputs_fixed, y_true = next(val_t1_generator)
                inputs_fixed = [torch.from_numpy(d).to(device1).float().permute(0, 4, 1, 2, 3) for d in inputs_fixed]
                # 提取对应t1的seg数据
                seg_inputs_fixed, seg_y_true = next(val_seg_generator)
                seg_inputs_fixed = [torch.from_numpy(d).to(device1).float().permute(0, 4, 1, 2, 3) for d in seg_inputs_fixed]
                for step in range(len(val_t1_files)):
                    # 提取t1数据
                    inputs_moving, y_true = next(val_t1_generator)
                    inputs_moving = [torch.from_numpy(d).to(device1).float().permute(0, 4, 1, 2, 3) for d in inputs_moving]
                    # 提取对应t1的seg数据
                    seg_inputs_moving, seg_y_true = next(val_seg_generator)
                    seg_inputs_moving = [torch.from_numpy(d).to(device1).float().permute(0, 4, 1, 2, 3) for d in seg_inputs_moving]
                    if step < len(val_t1_files) - 1:
                        with torch.no_grad():
                            # _,layer1_flow,_,_ = model_layer1(inputs_moving[0], inputs_fixed[0])
                            flow_1,flow_2,flow_3,flow,delta_flow_2,delta_flow_3,delta_flow_final,ori_alpha,gate_alpha = model(inputs_moving[0],inputs_fixed[0])
                            seg_pre = SPT(seg_inputs_moving[0], flow)
                            dice,dice_region = vxm.torch.losses.compute_dice(seg_pre, seg_inputs_fixed[0])
                            if dice < 1:
                                print("dice", dice)
                                Dice += dice
                                # print(Dice)
                                num += 1
                print("----------换fixed image---------")

        
            metric = Dice / num
            swanlab.log({"mean_dice": metric, 'epoch':epoch})

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1

            torch.save(model.state_dict(),os.path.join(model_dir, name))
                # torch.save(model_layer2.state_dict(),os.path.join(model_dir, 'best_model_CRNet_3layer.pth'))
                # torch.save(model,os.path.join(model_dir, 'best_model_MedNext.pt'))
                # print("saved new best metric MedNext layer4")

            print(f"Current epoch: {epoch+1} current dice in val_data: {metric} ")
            print(f"Best dice: {best_metric} at epoch {best_metric_epoch}")
            swanlab.log({"best_mean_dice": best_metric, 'epoch':epoch})
        torch.cuda.empty_cache()
    print("layer4 训练完成！")


# 针对全局进行剪枝 huge
def train_dual_pyramid_vxm_FFM_4layer_huge_GDP(config=config):
    set_random_seed(42)  # 你可以将42替换为任何其他整数种子值

    run = swanlab.init(project="APFNet", config=config)
    print("Training dual_pyramid_vxm_FFM_huge_GDP")

    model = vxm.networks.dual_pyramid_VxmDense_FFM_huge_GDP()
    model.to(device)

    checkpoint = torch.load(
        "./dual_pyramid_vxm_FFM_4layer_huge.pth")

    model_dict = model.state_dict()
    state_dict = {k: v for k, v in checkpoint.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

    model.train()

    '''注意检查是否需要多卡模型并行训练'''
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    device1 = torch.device('cuda:0')  # 分别表示用os.environ里面的第一个和第二个编号的gpu
    # device2 = torch.device('cuda:1')

    # wandb.watch(model,log="all", log_freq=10)

    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,model_layer4.parameters()), lr=args.lr)
    # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad,model_layer4.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    # 验证集
    val_interval = 5
    best_metric = -1
    decay = 0.5

    # for epoch in range(args.layer4_epochs):
    for epoch in range(1000):

        epoch_loss = []
        epoch_total_loss = []
        epoch_total_loss_2 = []
        epoch_step_time = []
        epoch_total_diceloss = []
        decay = decay * 0.98

        print(decay)
        print("epoch", epoch)
        model.set_decay(decay)

        for step in range(100):

            step_start_time = time.time()
            # recorder.step += 1

            # generate inputs (and true outputs) and convert them to tensors
            inputs, y_true = next(seg_generator)
            inputs = [torch.from_numpy(d).to(device1).float().permute(0, 4, 1, 2, 3) for d in inputs]
            y_true = [torch.from_numpy(d).to(device1).float().permute(0, 4, 1, 2, 3) for d in y_true]

            # 生成flow
            flow_1, flow_2, flow_3, flow_final, delta_flow_2, delta_flow_3, _, ori_alpha, gate_alpha = model(inputs[0],
                                                                                                             inputs[1])

            # 对moving图像进行变换得到moved image
            # 定义空间转换层
            SPT4 = vxm.torch.layers.SpatialTransformer((192, 160, 192), mode="bilinear").to(device1)
            moved_4 = SPT4(inputs[0], flow_final)

            y_pred_4 = []
            y_pred_4.append(moved_4)
            y_pred_4.append(flow_final)

            # calculate total loss
            loss = 0
            loss_list = []

            for n, loss_function in enumerate(losses):
                curr_loss = loss_function(inputs[1], y_pred_4[n]) * weights[n]
                loss_list.append(curr_loss.item())
                loss += curr_loss

            print(f"loss: {loss_list[0] + loss_list[1]:.4f}  sim_loss: {loss_list[0]:.4f}  reg_loss: {loss_list[1]} ")

            epoch_loss.append(loss_list)
            epoch_total_loss.append(loss.item())

            # backpropagate and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            # get compute time
            epoch_step_time.append(time.time() - step_start_time)

            theta = 4
            non_diff_loss_value, non_zero_alpha = nondifferentiable_loss(ori_alpha, theta=theta)
            # print("reg loss in alpha",non_diff_loss_value)
            # print("non_zero_alpha_num",non_zero_alpha)
            # print("ori_alpha",ori_alpha)

            if (step + 1) % 1 == 0:
                # 近端映射步骤来处理不可微部分
                input_alpha = [model.alpha_1, model.alpha_2, model.alpha_3, model.alpha_4, model.alpha_5,
                               model.alpha_6_1, model.alpha_6_2, model.alpha_7_1, model.alpha_7_2, model.alpha_8_1,
                               model.alpha_8_2]
                learning_rate = 0.00001
                Lambda = 0.0012
                proximal_operator_huge(input_alpha, learning_rate, Lambda, non_zero_alpha, theta=theta)
                # print("model_alpha_1",model.alpha_1)

        print("ori_alpha", ori_alpha)
        print("non_zero_alpha_num", non_zero_alpha)
        print("gate_alpha", gate_alpha)

        # print epoch info
        # scheduler.step()
        epoch_info = 'Epoch %d/%d' % (epoch + 1, args.epochs)
        time_info = '%.4f sec/step' % np.mean(epoch_step_time)
        losses_info = ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
        loss_info = 'loss: %.4e  (%s)' % (np.mean(epoch_total_loss), losses_info)
        print(' - '.join((epoch_info, time_info, loss_info)), flush=True)
        swanlab.log({"total_loss": np.mean(epoch_total_loss), 'epoch': epoch})

        # # 对验证集进行测试

        if (epoch + 1) % val_interval == 0:
            name = "daul_pyramid_vxm_FFM_4layer_4_huge_GDP_Lambda0.0012_theta4_decay0.5_decayrate0.98_withpretrain_abdomenCT_" + str(
                epoch) + ".pth"
            print(name)
            Dice = 0
            Dice_1 = 0
            Dice_2 = 0
            Dice_3 = 0
            mean_dice = 0
            num = 0
            model.eval()
            print("---------验证集的Dice----------")
            SPT = vxm.torch.layers.SpatialTransformer(inshape, mode="nearest").to(device1)
            for step in range(len(val_t1_files)):
                # 提取t1数据
                inputs_fixed, y_true = next(val_t1_generator)
                inputs_fixed = [torch.from_numpy(d).to(device1).float().permute(0, 4, 1, 2, 3) for d in inputs_fixed]
                # 提取对应t1的seg数据
                seg_inputs_fixed, seg_y_true = next(val_seg_generator)
                seg_inputs_fixed = [torch.from_numpy(d).to(device1).float().permute(0, 4, 1, 2, 3) for d in
                                    seg_inputs_fixed]
                for step in range(len(val_t1_files)):
                    # 提取t1数据
                    inputs_moving, y_true = next(val_t1_generator)
                    inputs_moving = [torch.from_numpy(d).to(device1).float().permute(0, 4, 1, 2, 3) for d in
                                     inputs_moving]
                    # 提取对应t1的seg数据
                    seg_inputs_moving, seg_y_true = next(val_seg_generator)
                    seg_inputs_moving = [torch.from_numpy(d).to(device1).float().permute(0, 4, 1, 2, 3) for d in
                                         seg_inputs_moving]
                    if step < len(val_t1_files) - 1:
                        with torch.no_grad():
                            # 生成flow
                            # x_in = torch.cat((inputs_moving[0],inputs_fixed[0]), dim=1)
                            flow_1, flow_2, flow_3, flow, delta_flow_2, delta_flow_3, _, _, _ = model(inputs_moving[0],
                                                                                                      inputs_fixed[0])

                            # 上采样
                            flow_1_up = nn.functional.interpolate(flow_1, scale_factor=8, mode="trilinear") * 8
                            flow_2_up = nn.functional.interpolate(flow_2, scale_factor=4, mode="trilinear") * 4
                            flow_3_up = nn.functional.interpolate(flow_3, scale_factor=2, mode="trilinear") * 2

                            seg_pre_1 = SPT(seg_inputs_moving[0], flow_1_up)
                            seg_pre_2 = SPT(seg_inputs_moving[0], flow_2_up)
                            seg_pre_3 = SPT(seg_inputs_moving[0], flow_3_up)
                            dice_1, _ = vxm.torch.losses.compute_dice(seg_pre_1, seg_inputs_fixed[0])
                            dice_2, _ = vxm.torch.losses.compute_dice(seg_pre_2, seg_inputs_fixed[0])
                            dice_3, _ = vxm.torch.losses.compute_dice(seg_pre_3, seg_inputs_fixed[0])

                            # 得到变形场2
                            seg_pre = SPT(seg_inputs_moving[0], flow)
                            dice, dice_region = vxm.torch.losses.compute_dice(seg_pre, seg_inputs_fixed[0])
                            if dice < 1:
                                print("dice", dice)
                                # print("dice_1", dice_1)
                                # print("dice_2", dice_2)
                                # print("dice_3", dice_3)
                                Dice += dice
                                Dice_1 += dice_1
                                Dice_2 += dice_2
                                Dice_3 += dice_3
                                # print(Dice)
                                num += 1
                print("----------换fixed image---------")

            metric = Dice / num
            metric_1 = Dice_1 / num
            metric_2 = Dice_2 / num
            metric_3 = Dice_3 / num
            swanlab.log({"mean_dice": metric, 'epoch': epoch})
            swanlab.log({"mean_dice_1": metric_1, 'epoch': epoch})
            swanlab.log({"mean_dice_2": metric_2, 'epoch': epoch})
            swanlab.log({"mean_dice_3": metric_3, 'epoch': epoch})

            torch.save(model.state_dict(), os.path.join(model_dir, name))

            print("saved new best metric MedNext layer4")

            print(f"Current epoch: {epoch + 1} current dice in val_data: {metric} ")
            # print(f"Best dice: {best_metric} at epoch {best_metric_epoch}")
            # wandb.log({"best_mean_dice": best_metric, 'epoch':epoch})
        torch.cuda.empty_cache()
    print("voxelmorph 训练完成！")


def train_dual_pyramid_vxm_PDFNet_normal_FFM_diff_GDP(config=config):
    set_random_seed(42)  # 你可以将42替换为任何其他整数种子值

    run = swanlab.init(project="APFNet", config=config)
    print("Training dual_pyramid_vxm_plus_deepKD")

    model = vxm.networks.dual_pyramid_VxmDense_FFM_normal_diff_new_GDP(inshape=(192, 160, 192))

    model.to(device)


    model.train()

    '''注意检查是否需要多卡模型并行训练'''
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    device1 = torch.device('cuda:0')  # 分别表示用os.environ里面的第一个和第二个编号的gpu
    # device2 = torch.device('cuda:1')

    # wandb.watch(model,log="all", log_freq=10)

    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,model_layer4.parameters()), lr=args.lr)
    # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad,model_layer4.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    # 验证集
    val_interval = 5
    best_metric = -1
    decay = 0.5

    # for epoch in range(args.layer4_epochs):
    for epoch in range(1000):

        epoch_loss = []
        epoch_total_loss = []
        epoch_total_loss_2 = []
        epoch_step_time = []
        epoch_total_diceloss = []
        check_total = 0
        check_total_lastlayer = 0

        decay = decay * 0.98
        print(decay)
        print("epoch", epoch)
        model.set_decay(decay)

        for step in range(100):

            step_start_time = time.time()
            # recorder.step += 1

            # generate inputs (and true outputs) and convert them to tensors
            inputs, y_true = next(seg_generator)
            inputs = [torch.from_numpy(d).to(device1).float().permute(0, 4, 1, 2, 3) for d in inputs]
            y_true = [torch.from_numpy(d).to(device1).float().permute(0, 4, 1, 2, 3) for d in y_true]

            # 生成flow
            flow_1, flow_2, flow_3, flow_final, delta_flow_2, delta_flow_3, _, ori_alpha, gate_alpha = model(inputs[0],
                                                                                                             inputs[1])

            # 对moving图像进行变换得到moved image
            # 定义空间转换层
            SPT4 = vxm.torch.layers.SpatialTransformer((192, 160, 192), mode="bilinear").to(device1)

            moved_4 = SPT4(inputs[0], flow_final)

            y_pred_4 = []
            y_pred_4.append(moved_4)
            y_pred_4.append(flow_final)

            # calculate total loss
            loss = 0
            loss_list = []
            sim_loss = 0
            sim_loss_last = 0

            for n, loss_function in enumerate(losses):
                curr_loss = loss_function(inputs[1], y_pred_4[n]) * weights[n]
                loss_list.append(curr_loss.item())
                loss += curr_loss
                if n == 0:
                    sim_loss += curr_loss
                    sim_loss_last += curr_loss

            # print(f"loss: {loss_list[0] + loss_list[1]+loss_list[2] + loss_list[3]+loss_list[4] + loss_list[5]+loss_list[6] + loss_list[7]:.4f}  sim_loss: {loss_list[0]+loss_list[2]+loss_list[4]+loss_list[6]:.4f}  reg_loss: {loss_list[1]+loss_list[3]+loss_list[5]+loss_list[7]} ")
            print(f"loss: {loss_list[0] + loss_list[1]:.4f}  sim_loss: {loss_list[0]:.4f}  reg_loss: {loss_list[1]}")

            epoch_loss.append(loss_list)
            epoch_total_loss.append(loss.item())

            # backpropagate and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            # get compute time
            epoch_step_time.append(time.time() - step_start_time)

            theta = 4
            non_diff_loss_value, non_zero_alpha = nondifferentiable_loss(ori_alpha, theta=theta)
            # print("reg loss in alpha",non_diff_loss_value)
            # print("non_zero_alpha_num",non_zero_alpha)
            # print("ori_alpha",ori_alpha)

            if (step + 1) % 1 == 0:
                # 近端映射步骤来处理不可微部分
                input_alpha = [model.alpha_1, model.alpha_2, model.alpha_3, model.alpha_4, model.alpha_5,
                               model.alpha_6_1, model.alpha_6_2, model.alpha_7_1, model.alpha_7_2, model.alpha_8_1,
                               model.alpha_8_2]
                learning_rate = 0.00001
                Lambda = 0.0036
                proximal_operator_normal(input_alpha, learning_rate, Lambda, non_zero_alpha, theta=theta)

                # print("model_alpha_1",model.alpha_1)
        print("ori_alpha", ori_alpha)
        print("non_zero_alpha_num", non_zero_alpha)
        print("gate_alpha", gate_alpha)

        # print epoch info
        # scheduler.step()
        epoch_info = 'Epoch %d/%d' % (epoch + 1, args.epochs)
        time_info = '%.4f sec/step' % np.mean(epoch_step_time)
        losses_info = ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
        loss_info = 'loss: %.4e  (%s)' % (np.mean(epoch_total_loss), losses_info)
        print(' - '.join((epoch_info, time_info, loss_info)), flush=True)
        swanlab.log({"total_loss": np.mean(epoch_total_loss), 'epoch': epoch})

        # # 对验证集进行测试

        if (epoch + 1) % val_interval == 0:
            name = "daul_pyramid_baseline_PDFNet_normal_FFM_diff_GDP_" + str(epoch) + ".pth"
            print(name)
            Dice = 0
            Dice_1 = 0
            Dice_2 = 0
            Dice_3 = 0
            Dice_tea = 0
            mean_dice = 0
            num = 0
            total_jet = 0
            model.eval()
            print("---------验证集的Dice----------")
            SPT = vxm.torch.layers.SpatialTransformer(inshape, mode="nearest").to(device1)
            for step in range(len(val_t1_files)):
                # 提取t1数据
                inputs_fixed, y_true = next(val_t1_generator)
                inputs_fixed = [torch.from_numpy(d).to(device1).float().permute(0, 4, 1, 2, 3) for d in inputs_fixed]
                # 提取对应t1的seg数据
                seg_inputs_fixed, seg_y_true = next(val_seg_generator)
                seg_inputs_fixed = [torch.from_numpy(d).to(device1).float().permute(0, 4, 1, 2, 3) for d in
                                    seg_inputs_fixed]
                for step in range(len(val_t1_files)):
                    # 提取t1数据
                    inputs_moving, y_true = next(val_t1_generator)
                    inputs_moving = [torch.from_numpy(d).to(device1).float().permute(0, 4, 1, 2, 3) for d in
                                     inputs_moving]
                    # 提取对应t1的seg数据
                    seg_inputs_moving, seg_y_true = next(val_seg_generator)
                    seg_inputs_moving = [torch.from_numpy(d).to(device1).float().permute(0, 4, 1, 2, 3) for d in
                                         seg_inputs_moving]
                    if step < len(val_t1_files) - 1:
                        with torch.no_grad():
                            # 生成flow
                            # x_in = torch.cat((inputs_moving[0],inputs_fixed[0]), dim=1)
                            flow_1, flow_2, flow_3, flow, delta_flow_1, delta_flow_2, _, _, _ = model(inputs_moving[0],
                                                                                                      inputs_fixed[0])
                            # y_flow,_,_,_,_,_,_ = model_teacher(inputs_moving[0],inputs_fixed[0])

                            # 上采样
                            flow_1_up = nn.functional.interpolate(flow_1, scale_factor=8, mode="trilinear") * 8
                            flow_2_up = nn.functional.interpolate(flow_2, scale_factor=4, mode="trilinear") * 4
                            flow_3_up = nn.functional.interpolate(flow_3, scale_factor=2, mode="trilinear") * 2

                            seg_pre_1 = SPT(seg_inputs_moving[0], flow_1_up)
                            seg_pre_2 = SPT(seg_inputs_moving[0], flow_2_up)
                            seg_pre_3 = SPT(seg_inputs_moving[0], flow_3_up)
                            # seg_pre_tea = SPT(seg_inputs_moving[0], y_flow)
                            dice_1, _ = vxm.torch.losses.compute_dice(seg_pre_1, seg_inputs_fixed[0])
                            dice_2, _ = vxm.torch.losses.compute_dice(seg_pre_2, seg_inputs_fixed[0])
                            dice_3, _ = vxm.torch.losses.compute_dice(seg_pre_3, seg_inputs_fixed[0])
                            # dice_tea,_ = vxm.torch.losses.compute_dice(seg_pre_tea, seg_inputs_fixed[0])

                            # 得到变形场2
                            seg_pre = SPT(seg_inputs_moving[0], flow)
                            dice, dice_region = vxm.torch.losses.compute_dice(seg_pre, seg_inputs_fixed[0])

                            # 计算负雅可比的个数
                            filed = torch.zeros([192, 160, 192, 3]).to(device)
                            filed[:, :, :, 0] = flow[0, 0, :, :, :]
                            filed[:, :, :, 1] = flow[0, 1, :, :, :]
                            filed[:, :, :, 2] = flow[0, 2, :, :, :]
                            jac_det = vxm.torch.losses.jacobian_determinant(filed.cpu())
                            Jd = np.sum(jac_det < 0)
                            if dice < 1:
                                print("dice", dice)
                                # print("dice_1", dice_1)
                                # print("dice_2", dice_2)
                                # print("dice_3", dice_3)
                                # print("dice_tea", dice_tea)
                                Dice += dice
                                Dice_1 += dice_1
                                Dice_2 += dice_2
                                Dice_3 += dice_3
                                # Dice_tea += dice_tea
                                # print(Dice)
                                num += 1
                                total_jet += Jd
                                print("JD", Jd)
                print("----------换fixed image---------")

            metric = Dice / num
            metric_1 = Dice_1 / num
            metric_2 = Dice_2 / num
            metric_3 = Dice_3 / num
            mean_jet = total_jet / num
            print("mean_jet", mean_jet)
            # metric_tea = Dice_tea / num
            print("mean_dice", metric)
            swanlab.log({"mean_dice": metric, 'epoch': epoch})
            swanlab.log({"mean_dice_1": metric_1, 'epoch': epoch})
            swanlab.log({"mean_dice_2": metric_2, 'epoch': epoch})
            swanlab.log({"mean_dice_3": metric_3, 'epoch': epoch})

            torch.save(model.state_dict(), os.path.join(model_dir, name))

            print("saved new best metric MedNext layer4")

            print(f"Current epoch: {epoch + 1} current dice in val_data: {metric} ")
        torch.cuda.empty_cache()
    print("voxelmorph 训练完成！")


# 将大模型的参数赋值给小模型并保存 huge
def test_dual_pyramid_vxm_FFM_adaptive_huge(config=config):
    print("Training dual_pyramid_vxm_plus")

    # 读取huge model
    model_pretrain = vxm.networks.dual_pyramid_VxmDense_FFM_huge_GDP()

    checkpoint = torch.load(
        "./daul_pyramid_vxm_FFM_4layer_4_huge_GDP_Lambda0.0012_theta4_decay0.5_decayrate0.98_nopretrain_abdomenCT_894.pth")

    model_pretrain.to(device)
    model_dict = model_pretrain.state_dict()
    state_dict = {k: v for k, v in checkpoint.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model_pretrain.load_state_dict(model_dict)

    # model_pretrain.train()
    decay = 0.5 * (0.98 ** 894)
    model_pretrain.set_decay(decay)

    inputs, y_true = next(seg_generator)
    inputs = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs]
    y_true = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in y_true]

    # 生成flow
    with torch.no_grad():
        # 得到非零索引
        flow_1, flow_2, flow_3, flow_final, delta_flow_2, delta_flow_3, _, ori_alpha, gate_alpha = model_pretrain(
            inputs[0], inputs[1])

        # print("gate_alpha",gate_alpha)
    # print("ori_aplha",ori_alpha)
    # print(ori_alpha)

    index = []
    for tensor in gate_alpha:
        indices = torch.nonzero(tensor).squeeze()  # 获取非零元素的索引
        index.append(indices)
    print(len(index[0]))
    print(len(index[1]))
    print(len(index[2]))
    print(len(index[3]))
    print(len(index[4]))
    print(len(index[5]))
    print(len(index[6]))
    print(len(index[7]))
    print(len(index[8]))
    print(len(index[9]))
    print(len(index[10]))

    list_num = [len(index[0]), len(index[1]), len(index[2]), len(index[3]), len(index[4]), len(index[5]), len(index[6]),
                len(index[7]), len(index[8]), len(index[9]), len(index[10])]
    new_alpha = []
    for i in range(len(list_num)):
        ori_alpha_i = ori_alpha[i]
        new_alpha_i = torch.ones(1, list_num[i])
        for j in range(list_num[i]):
            # if i == 2:
            #     index[2] = torch.unsqueeze(index[2], 0)
            #     print(index[2])
            # print(i)
            # print(j)
            location = index[i][j][1]
            print(location)
            new_alpha_i[0][j] = ori_alpha_i[0][location]
        new_alpha.append(new_alpha_i)

    # 初始化一个轻量的小模型
    # model = vxm.networks.dual_pyramid_VxmDense_FFM_4layer_4_huge_adaptive_val_2(alpha1=new_alpha[0], alpha2=new_alpha[1],
    #         alpha3=new_alpha[2], alpha4=new_alpha[3],alpha5=new_alpha[4], alpha6_1=new_alpha[5],alpha6_2=new_alpha[6],
    #         alpha7_1=new_alpha[7], alpha7_2=new_alpha[8],alpha8_1=new_alpha[9],alpha8_2=new_alpha[10],list_num=list_num)

    # model.set_decay(decay)

    model = vxm.networks.dual_pyramid_VxmDense_FFM_huge_adaptive_val(list_num=list_num)

    model.to(device)
    model_dict = model.state_dict()

    # # 验证回传是否正确
    # with torch.no_grad():
    #     # 得到非零索引
    #     flow_1,flow_2,flow_3,flow_final,delta_flow_2,delta_flow_3,_,ori_alpha,gate_alpha_new = model(inputs[0],inputs[1])
    #     print(gate_alpha_new)

    # index = index1

    state_dict = {}
    # # 加载参数
    for k, v in checkpoint.items():
        # print(k)
        # print(v.shape)

        if k == "encoder1_m.main.weight":
            print("encoder1_m.main.weight")
            new_v = torch.zeros(list_num[0], 1, 3, 3, 3)
            index_list = []
            for j in range(list_num[0]):
                location = index[0][j][1]
                index_list.append(location)
                # print(location)
            for i, j in enumerate(index_list):
                new_v[i] = v[j]
            state_dict["encoder1_m.main.weight"] = new_v

        if k == "encoder1_m.main.bias":
            print("encoder1_m.main.bias")

            new_v = torch.zeros(list_num[0])
            index_list = []
            for j in range(list_num[0]):
                location = index[0][j][1]
                index_list.append(location)
                # print(location)
            for i, j in enumerate(index_list):
                new_v[i] = v[j]
            state_dict["encoder1_m.main.bias"] = new_v

        if k == "encoder2_m.main.weight":
            print("encoder2_m.main.weight")
            new_v = torch.zeros(list_num[1], list_num[0], 3, 3, 3)

            last_index_list = []
            for j in range(list_num[0]):
                location = index[0][j][1]
                last_index_list.append(location)

            index_list = []
            for j in range(list_num[1]):
                location = index[1][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list):
                    # print(f"i: {i}, m: {m}, j: {j}, n: {n}")

                    new_v[i][m] = v[j][n]
            state_dict["encoder2_m.main.weight"] = new_v

        if k == "encoder2_m.main.bias":
            print("encoder2_m.main.bias")
            # print(v.shape)
            new_v = torch.zeros(list_num[1])

            index_list = []
            for j in range(list_num[1]):
                location = index[1][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                new_v[i] = v[j]
            state_dict["encoder2_m.main.bias"] = new_v

        if k == "encoder3_m.main.weight":
            print("encoder3_m.main.weight")
            new_v = torch.zeros(list_num[2], list_num[1], 3, 3, 3)

            last_index_list = []
            for j in range(list_num[1]):
                location = index[1][j][1]
                last_index_list.append(location)

            index_list = []
            for j in range(list_num[2]):
                location = index[2][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list):
                    # print(f"i: {i}, m: {m}, j: {j}, n: {n}")

                    new_v[i][m] = v[j][n]
            state_dict["encoder3_m.main.weight"] = new_v

        if k == "encoder3_m.main.bias":
            print("encoder3_m.main.bias")
            new_v = torch.zeros(list_num[2])

            index_list = []
            for j in range(list_num[2]):
                location = index[2][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                new_v[i] = v[j]
            state_dict["encoder3_m.main.bias"] = new_v

        if k == "encoder4_m.main.weight":
            print("encoder4_m.main.weight")
            new_v = torch.zeros(list_num[3], list_num[2], 3, 3, 3)

            last_index_list = []
            for j in range(list_num[2]):
                location = index[2][j][1]
                last_index_list.append(location)

            index_list = []
            for j in range(list_num[3]):
                location = index[3][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list):
                    # print(f"i: {i}, m: {m}, j: {j}, n: {n}")

                    new_v[i][m] = v[j][n]
            state_dict["encoder4_m.main.weight"] = new_v

        if k == "encoder4_m.main.bias":
            print("encoder4_m.main.bias")
            new_v = torch.zeros(list_num[3])

            index_list = []
            for j in range(list_num[3]):
                location = index[3][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                new_v[i] = v[j]
            state_dict["encoder4_m.main.bias"] = new_v

        if k == "encoder5_m.main.weight":
            print("encoder5_m.main.weight")
            new_v = torch.zeros(list_num[4], list_num[3], 3, 3, 3)

            last_index_list = []
            for j in range(list_num[3]):
                location = index[3][j][1]
                last_index_list.append(location)

            index_list = []
            for j in range(list_num[4]):
                location = index[4][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list):
                    # print(f"i: {i}, m: {m}, j: {j}, n: {n}")

                    new_v[i][m] = v[j][n]
            state_dict["encoder5_m.main.weight"] = new_v

        if k == "encoder5_m.main.bias":
            print("encoder5_m.main.bias")
            new_v = torch.zeros(list_num[4])

            index_list = []
            for j in range(list_num[4]):
                location = index[4][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                new_v[i] = v[j]
            state_dict["encoder5_m.main.bias"] = new_v

        if k == "decoder1.main.weight":
            print("decoder1.main.weight")
            new_v = torch.zeros(32, 192, 3, 3, 3)  # v(64,128,3,3,3)
            new_v = v
            state_dict["decoder1.main.weight"] = new_v

        if k == "decoder1.main.bias":
            print("decoder1.main.bias")
            new_v = torch.zeros(32)
            new_v = v
            state_dict["decoder1.main.bias"] = new_v

        if k == "decoder2.main.weight":
            print("decoder2.main.weight")
            new_v = torch.zeros(32, 224, 3, 3, 3)  # v(64,128,3,3,3)
            new_v = v
            state_dict["decoder2.main.weight"] = new_v

        if k == "decoder2.main.bias":
            print("decoder2.main.bias")
            new_v = torch.zeros(32)
            new_v = v
            state_dict["decoder2.main.bias"] = new_v

        if k == "decoder3.main.weight":
            print("decoder3.main.weight")
            new_v = torch.zeros(32, 32 + list_num[8] + list_num[8], 3, 3, 3)  # v(64,128,3,3,3)

            last_index_list_0 = []
            for j in range(list_num[8]):
                location = index[8][j][1]
                last_index_list_0.append(location)

            for i in range(32):
                for j in range(32):
                    new_v[i][j] = v[i][j]

            for i in range(32):
                for m, n in enumerate(last_index_list_0):  # 16
                    new_v[i][m + 32] = v[i][n + 32]

            for i in range(32):
                for m, n in enumerate(last_index_list_0):  # 9
                    new_v[i][m + list_num[8] + 32] = v[i][n + 32 + 96]

            state_dict["decoder3.main.weight"] = new_v

        if k == "decoder3.main.bias":
            print("decoder3.main.bias")
            new_v = torch.zeros(32)
            new_v = v
            state_dict["decoder3.main.bias"] = new_v

        if k == "decoder4.main.weight":
            print("decoder4.main.weight")
            new_v = torch.zeros(16, list_num[10] + list_num[10] + 32, 3, 3, 3)  # v(64,128,3,3,3)

            last_index_list_0 = []
            for j in range(list_num[10]):
                location = index[10][j][1]
                last_index_list_0.append(location)

            for i in range(16):
                for j in range(32):
                    new_v[i][j] = v[i][j]

            for i in range(16):
                for m, n in enumerate(last_index_list_0):  # 16
                    new_v[i][m + 32] = v[i][n + 32]

            for i in range(16):
                for m, n in enumerate(last_index_list_0):  # 9
                    new_v[i][m + list_num[10] + 32] = v[i][n + 32 + 96]

            state_dict["decoder4.main.weight"] = new_v

        if k == "decoder4.main.bias":
            print("decoder4.main.bias")
            new_v = torch.zeros(16)
            new_v = v
            state_dict["decoder4.main.bias"] = new_v

        if k == "decoder5.main.weight":
            print("decoder5.main.weight")
            new_v = torch.zeros(16, 16 + list_num[0] + list_num[0], 3, 3, 3)  # v (32,96,3,3,3)

            last_index_list_0 = []
            for j in range(list_num[0]):
                location = index[0][j][1]
                last_index_list_0.append(location)

            for i in range(16):
                for j in range(16):
                    new_v[i][j] = v[i][j]

            for i in range(16):
                for m, n in enumerate(last_index_list_0):  # 16
                    new_v[i][m + 16] = v[i][n + 16]

            for i in range(16):
                for m, n in enumerate(last_index_list_0):  # 9
                    new_v[i][m + list_num[0] + 16] = v[i][n + 48 + 16]

            state_dict["decoder5.main.weight"] = new_v

        if k == "decoder5.main.bias":
            print("decoder5.main.bias")
            new_v = v
            state_dict["decoder5.main.bias"] = new_v

        if k == "fusion_m_scale4.main.weight":
            print("fusion_m_scale4.main.weight")
            new_v = torch.zeros(96, list_num[0] + list_num[1] + list_num[2] + list_num[3], 3, 3, 3)  # v (64,224,3,3,3)

            last_index_list_0 = []
            for j in range(list_num[0]):
                location = index[0][j][1]
                last_index_list_0.append(location)

            last_index_list_1 = []
            for j in range(list_num[1]):
                location = index[1][j][1]
                last_index_list_1.append(location)

            last_index_list_2 = []
            for j in range(list_num[2]):
                location = index[2][j][1]
                last_index_list_2.append(location)

            last_index_list_3 = []
            for j in range(list_num[3]):
                location = index[3][j][1]
                last_index_list_3.append(location)

            index_list = []
            for j in range(list_num[5]):
                location = index[5][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_0):  # 9
                    # print(f"i: {i}, m: {m}, j: {j}, n: {n}")
                    new_v[i][m] = v[j][n]

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_1):  # 12
                    new_v[i][m + list_num[0]] = v[j][n + 48]

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_2):  # 2
                    new_v[i][m + list_num[0] + list_num[1]] = v[j][n + 48 + 96]

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_3):  # 1
                    new_v[i][m + list_num[0] + list_num[1] + list_num[2]] = v[j][n + 48 + 96 + 96]

            state_dict["fusion_m_scale4.main.weight"] = new_v

        if k == "fusion_m_scale4.main.bias":
            print("fusion_m_scale4.main.bias")
            new_v = torch.zeros(96)
            new_v = v
            state_dict["fusion_m_scale4.main.bias"] = new_v

        if k == "fusion_m_scale3.main.weight":
            print("fusion_m_scale3.main.weight")
            new_v = torch.zeros(list_num[7], list_num[0] + list_num[1] + list_num[2] + list_num[3], 3, 3,
                                3)  # v (64,224,3,3,3)

            last_index_list_0 = []
            for j in range(list_num[0]):
                location = index[0][j][1]
                last_index_list_0.append(location)

            last_index_list_1 = []
            for j in range(list_num[1]):
                location = index[1][j][1]
                last_index_list_1.append(location)

            last_index_list_2 = []
            for j in range(list_num[2]):
                location = index[2][j][1]
                last_index_list_2.append(location)

            last_index_list_3 = []
            for j in range(list_num[3]):
                location = index[3][j][1]
                last_index_list_3.append(location)

            index_list = []
            for j in range(list_num[7]):
                location = index[7][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_0):  # 9
                    # print(f"i: {i}, m: {m}, j: {j}, n: {n}")
                    new_v[i][m] = v[j][n]

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_1):  # 12
                    new_v[i][m + list_num[0]] = v[j][n + 48]

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_2):  # 2
                    new_v[i][m + list_num[0] + list_num[1]] = v[j][n + 48 + 96]

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_3):  # 1
                    new_v[i][m + list_num[0] + list_num[1] + list_num[2]] = v[j][n + 48 + 96 + 96]

            state_dict["fusion_m_scale3.main.weight"] = new_v

        if k == "fusion_m_scale3.main.bias":
            print("fusion_m_scale3.main.bias")
            new_v = torch.zeros(list_num[7])
            index_list = []
            for j in range(list_num[7]):
                location = index[7][j][1]
                index_list.append(location)
                # print(location)
            for i, j in enumerate(index_list):
                new_v[i] = v[j]
            state_dict["fusion_m_scale3.main.bias"] = new_v

        if k == "fusion_m_scale2.main.weight":
            print("fusion_m_scale2.main.weight")
            new_v = torch.zeros(list_num[9], list_num[0] + list_num[1] + list_num[2] + list_num[3], 3, 3,
                                3)  # v (64,224,3,3,3)

            last_index_list_0 = []
            for j in range(list_num[0]):
                location = index[0][j][1]
                last_index_list_0.append(location)

            last_index_list_1 = []
            for j in range(list_num[1]):
                location = index[1][j][1]
                last_index_list_1.append(location)

            last_index_list_2 = []
            for j in range(list_num[2]):
                location = index[2][j][1]
                last_index_list_2.append(location)

            last_index_list_3 = []
            for j in range(list_num[3]):
                location = index[3][j][1]
                last_index_list_3.append(location)

            index_list = []
            for j in range(list_num[9]):
                location = index[9][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_0):  # 9
                    # print(f"i: {i}, m: {m}, j: {j}, n: {n}")
                    new_v[i][m] = v[j][n]

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_1):  # 12
                    new_v[i][m + list_num[0]] = v[j][n + 48]

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_2):  # 2
                    new_v[i][m + list_num[0] + list_num[1]] = v[j][n + 48 + 96]

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_3):  # 1
                    new_v[i][m + list_num[0] + list_num[1] + list_num[2]] = v[j][n + 48 + 96 + 96]

            state_dict["fusion_m_scale2.main.weight"] = new_v

        if k == "fusion_m_scale2.main.bias":
            print("fusion_m_scale2.main.bias")
            new_v = torch.zeros(list_num[9])
            index_list = []
            for j in range(list_num[9]):
                location = index[9][j][1]
                index_list.append(location)
                # print(location)
            for i, j in enumerate(index_list):
                new_v[i] = v[j]
            state_dict["fusion_m_scale2.main.bias"] = new_v

        if k == "fusion_m_scale4_2.main.weight":
            print("fusion_m_scale4_2.main.weight")
            new_v = torch.zeros(96, 96, 3, 3, 3)

            last_index_list = []
            for j in range(list_num[5]):
                location = index[5][j][1]
                last_index_list.append(location)

            index_list = []
            for j in range(list_num[6]):
                location = index[6][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list):
                    # print(f"i: {i}, m: {m}, j: {j}, n: {n}")

                    new_v[i][m] = v[j][n]
            state_dict["fusion_m_scale4_2.main.weight"] = new_v

        if k == "fusion_m_scale4_2.main.bias":
            print("fusion_m_scale4_2.main.bias")
            new_v = torch.zeros(96)

            index_list = []
            for j in range(list_num[6]):
                location = index[6][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                new_v[i] = v[j]
            state_dict["fusion_m_scale4_2.main.bias"] = new_v

        if k == "fusion_m_scale3_2.main.weight":
            print("fusion_m_scale3_2.main.weight")
            new_v = torch.zeros(list_num[8], list_num[7], 3, 3, 3)

            last_index_list = []
            for j in range(list_num[7]):
                location = index[7][j][1]
                last_index_list.append(location)

            index_list = []
            for j in range(list_num[8]):
                location = index[8][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list):
                    # print(f"i: {i}, m: {m}, j: {j}, n: {n}")

                    new_v[i][m] = v[j][n]
            state_dict["fusion_m_scale3_2.main.weight"] = new_v

        if k == "fusion_m_scale3_2.main.bias":
            print("fusion_m_scale3_2.main.bias")
            new_v = torch.zeros(list_num[8])

            index_list = []
            for j in range(list_num[8]):
                location = index[8][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                new_v[i] = v[j]
            state_dict["fusion_m_scale3_2.main.bias"] = new_v

        if k == "fusion_m_scale2_2.main.weight":
            print("fusion_m_scale2_2.main.weight")
            new_v = torch.zeros(list_num[10], list_num[9], 3, 3, 3)

            last_index_list = []
            for j in range(list_num[9]):
                location = index[9][j][1]
                last_index_list.append(location)

            index_list = []
            for j in range(list_num[10]):
                location = index[10][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list):
                    # print(f"i: {i}, m: {m}, j: {j}, n: {n}")

                    new_v[i][m] = v[j][n]
            state_dict["fusion_m_scale2_2.main.weight"] = new_v

        if k == "fusion_m_scale2_2.main.bias":
            print("fusion_m_scale2_2.main.bias")
            new_v = torch.zeros(list_num[10])

            index_list = []
            for j in range(list_num[10]):
                location = index[10][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                new_v[i] = v[j]
            state_dict["fusion_m_scale2_2.main.bias"] = new_v

        if k == "output_block_1.0.main.weight":
            new_v = v
            state_dict["output_block_1.0.main.weight"] = new_v

        if k == "output_block_1.0.main.bias":
            new_v = v
            state_dict["output_block_1.0.main.bias"] = new_v

        if k == "output_block_1.1.main.weight":
            new_v = v
            state_dict["output_block_1.1.main.weight"] = new_v

        if k == "output_block_1.1.main.bias":
            new_v = v
            state_dict["output_block_1.1.main.bias"] = new_v

        if k == "output_block_2.0.main.weight":
            new_v = v
            state_dict["output_block_2.0.main.weight"] = new_v

        if k == "output_block_2.0.main.bias":
            new_v = v
            state_dict["output_block_2.0.main.bias"] = new_v

        if k == "output_block_2.1.main.weight":
            new_v = v
            state_dict["output_block_2.1.main.weight"] = new_v

        if k == "output_block_2.1.main.bias":
            new_v = v
            state_dict["output_block_2.1.main.bias"] = new_v

        if k == "output_block_3.0.main.weight":
            new_v = v
            state_dict["output_block_3.0.main.weight"] = new_v

        if k == "output_block_3.0.main.bias":
            new_v = v
            state_dict["output_block_3.0.main.bias"] = new_v

        if k == "output_block_3.1.main.weight":
            new_v = v
            state_dict["output_block_3.1.main.weight"] = new_v

        if k == "output_block_3.1.main.bias":
            new_v = v
            state_dict["output_block_3.1.main.bias"] = new_v

        if k == "output_block.0.main.weight":
            new_v = v
            state_dict["output_block.0.main.weight"] = new_v

        if k == "output_block.0.main.bias":
            new_v = v
            state_dict["output_block.0.main.bias"] = new_v

        if k == "output_block.1.main.weight":
            new_v = v
            state_dict["output_block.1.main.weight"] = new_v

        if k == "output_block.1.main.bias":
            new_v = v
            state_dict["output_block.1.main.bias"] = new_v

        if k == "flow.weight":
            new_v = v
            state_dict["flow.weight"] = new_v

        if k == "flow.bias":
            new_v = v
            state_dict["flow.bias"] = new_v

    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

    # # 统计参数量
    dummy_input = torch.randn(1, 1, 192, 160, 192).to(device)
    flops, params = profile(model, (dummy_input, dummy_input,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))

    # # 对验证集进行测试
    Dice = 0
    Dice_1 = 0
    Dice_2 = 0
    Dice_3 = 0
    mean_dice = 0
    num = 0
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = torch.zeros(900)
    model.eval()
    print("---------验证集的Dice----------")
    SPT = vxm.torch.layers.SpatialTransformer(inshape, mode="nearest").to(device)
    for step in range(len(val_t1_files)):
        # 提取t1数据
        inputs_fixed, y_true = next(val_t1_generator)
        inputs_fixed = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs_fixed]
        # 提取对应t1的seg数据
        seg_inputs_fixed, seg_y_true = next(val_seg_generator)
        seg_inputs_fixed = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in seg_inputs_fixed]
        for step in range(len(val_t1_files)):
            # 提取t1数据
            inputs_moving, y_true = next(val_t1_generator)
            inputs_moving = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs_moving]
            # 提取对应t1的seg数据
            seg_inputs_moving, seg_y_true = next(val_seg_generator)
            seg_inputs_moving = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in
                                 seg_inputs_moving]
            if step < len(val_t1_files) - 1:
                with torch.no_grad():
                    # 生成flow
                    starter.record()

                    flow_1, flow_2, flow_3, flow, delta_flow_2, delta_flow_3, delta_flow_final, features_xf, features_xm = model(
                        inputs_moving[0], inputs_fixed[0])
                    # flow_1,flow_2,flow_3,flow,delta_flow_2,delta_flow_3,delta_flow_final,features_xf,features_xm = model_pretrain(inputs_moving[0],inputs_fixed[0])

                    ender.record()
                    # 同步GPU时间
                    torch.cuda.synchronize()
                    curr_time = starter.elapsed_time(ender)  # 计算时间
                    times[num] = curr_time
                    # num = num + 1
                    print(curr_time)

                    # x_in = torch.cat((inputs_moving[0],inputs_fixed[0]), dim=1)
                    # flow_1,flow_2,flow_3,flow,delta_flow_2,delta_flow_3,_ = model(inputs_moving[0],inputs_fixed[0])

                    # 上采样
                    flow_1_up = nn.functional.interpolate(flow_1, scale_factor=8, mode="trilinear") * 8
                    flow_2_up = nn.functional.interpolate(flow_2, scale_factor=4, mode="trilinear") * 4
                    flow_3_up = nn.functional.interpolate(flow_3, scale_factor=2, mode="trilinear") * 2

                    seg_pre_1 = SPT(seg_inputs_moving[0], flow_1_up)
                    seg_pre_2 = SPT(seg_inputs_moving[0], flow_2_up)
                    seg_pre_3 = SPT(seg_inputs_moving[0], flow_3_up)
                    dice_1, _ = vxm.torch.losses.compute_dice(seg_pre_1, seg_inputs_fixed[0])
                    dice_2, _ = vxm.torch.losses.compute_dice(seg_pre_2, seg_inputs_fixed[0])
                    dice_3, _ = vxm.torch.losses.compute_dice(seg_pre_3, seg_inputs_fixed[0])

                    # 得到变形场2
                    seg_pre = SPT(seg_inputs_moving[0], flow)
                    dice, dice_region = vxm.torch.losses.compute_dice(seg_pre, seg_inputs_fixed[0])
                    if dice < 1:
                        print("dice", dice)
                        Dice += dice
                        num += 1
        print("----------换fixed image---------")

    metric = Dice / num
    print("mean_dice", metric)
    mean_time = times.mean().item()
    print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000 / mean_time))

    torch.save(model.state_dict(),
               "/home/boys/project/voxelmorph/voxelmorph_code/models/comparison_methods/abdomenCT/daul_pyramid_PDFNet_FFM_huge_GDP_adaptive_nopretrain.pth")


# 将大模型的参数赋值给小模型并保存 large
def test_dual_pyramid_vxm_FFM_adaptive_large(config=config):
    print("Training dual_pyramid_vxm_plus")

    # 读取huge model
    model_pretrain = vxm.networks.dual_pyramid_VxmDense_FFM_large_GDP()

    checkpoint = torch.load(
        "./daul_pyramid_vxm_FFM_4layer_4_large_GDP_Lambda0.0018_theta4_decay0.5_decayrate0.98_nopretrain_abdomenmrct_new_839.pth")

    model_pretrain.to(device)
    model_dict = model_pretrain.state_dict()
    state_dict = {k: v for k, v in checkpoint.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model_pretrain.load_state_dict(model_dict)

    # model_pretrain.train()
    decay = 0.5 * (0.98 ** 839)
    # decay = 0.00000000000000000000001
    model_pretrain.set_decay(decay)

    inputs, y_true = next(seg_generator)
    inputs = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs]
    y_true = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in y_true]

    # 生成flow
    with torch.no_grad():
        # 得到非零索引
        flow_1, flow_2, flow_3, flow_final, delta_flow_2, delta_flow_3, _, ori_alpha, gate_alpha = model_pretrain(
            inputs[0], inputs[1])

        # print("gate_alpha",gate_alpha)
    # print("ori_aplha",ori_alpha)
    # print(ori_alpha)

    index = []
    for tensor in gate_alpha:
        indices = torch.nonzero(tensor).squeeze()  # 获取非零元素的索引
        index.append(indices)
    print(len(index[0]))
    print(len(index[1]))
    print(len(index[2]))
    print(len(index[3]))
    print(len(index[4]))
    print(len(index[5]))
    print(len(index[6]))
    print(len(index[7]))
    print(len(index[8]))
    print(len(index[9]))
    print(len(index[10]))

    list_num = [len(index[0]), len(index[1]), len(index[2]), len(index[3]), len(index[4]), len(index[5]), len(index[6]),
                len(index[7]), len(index[8]), len(index[9]), len(index[10])]
    new_alpha = []
    for i in range(len(list_num)):
        ori_alpha_i = ori_alpha[i]
        new_alpha_i = torch.ones(1, list_num[i])
        for j in range(list_num[i]):
            # if i == 2:
            #     index[2] = torch.unsqueeze(index[2], 0)
            #     print(index[2])
            # print(i)
            # print(j)
            location = index[i][j][1]
            print(location)
            new_alpha_i[0][j] = ori_alpha_i[0][location]
        new_alpha.append(new_alpha_i)

    # 初始化一个轻量的小模型
    # model = vxm.networks.dual_pyramid_VxmDense_FFM_4layer_4_huge_adaptive_val_2(alpha1=new_alpha[0], alpha2=new_alpha[1],
    #         alpha3=new_alpha[2], alpha4=new_alpha[3],alpha5=new_alpha[4], alpha6_1=new_alpha[5],alpha6_2=new_alpha[6],
    #         alpha7_1=new_alpha[7], alpha7_2=new_alpha[8],alpha8_1=new_alpha[9],alpha8_2=new_alpha[10],list_num=list_num)

    # model.set_decay(decay)

    model = vxm.networks.dual_pyramid_VxmDense_FFM_large_adaptive_val(list_num=list_num)

    model.to(device)
    model_dict = model.state_dict()

    # # 验证回传是否正确
    # with torch.no_grad():
    #     # 得到非零索引
    #     flow_1,flow_2,flow_3,flow_final,delta_flow_2,delta_flow_3,_,ori_alpha,gate_alpha_new = model(inputs[0],inputs[1])
    #     print(gate_alpha_new)

    # index = index1

    state_dict = {}
    # # 加载参数
    for k, v in checkpoint.items():
        # print(k)
        # print(v.shape)

        if k == "encoder1_m.main.weight":
            print("encoder1_m.main.weight")
            new_v = torch.zeros(list_num[0], 1, 3, 3, 3)
            index_list = []
            for j in range(list_num[0]):
                location = index[0][j][1]
                index_list.append(location)
                # print(location)
            for i, j in enumerate(index_list):
                new_v[i] = v[j]
            state_dict["encoder1_m.main.weight"] = new_v

        if k == "encoder1_m.main.bias":
            print("encoder1_m.main.bias")

            new_v = torch.zeros(list_num[0])
            index_list = []
            for j in range(list_num[0]):
                location = index[0][j][1]
                index_list.append(location)
                # print(location)
            for i, j in enumerate(index_list):
                new_v[i] = v[j]
            state_dict["encoder1_m.main.bias"] = new_v

        if k == "encoder2_m.main.weight":
            print("encoder2_m.main.weight")
            new_v = torch.zeros(list_num[1], list_num[0], 3, 3, 3)

            last_index_list = []
            for j in range(list_num[0]):
                location = index[0][j][1]
                last_index_list.append(location)

            index_list = []
            for j in range(list_num[1]):
                location = index[1][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list):
                    # print(f"i: {i}, m: {m}, j: {j}, n: {n}")

                    new_v[i][m] = v[j][n]
            state_dict["encoder2_m.main.weight"] = new_v

        if k == "encoder2_m.main.bias":
            print("encoder2_m.main.bias")
            # print(v.shape)
            new_v = torch.zeros(list_num[1])

            index_list = []
            for j in range(list_num[1]):
                location = index[1][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                new_v[i] = v[j]
            state_dict["encoder2_m.main.bias"] = new_v

        if k == "encoder3_m.main.weight":
            print("encoder3_m.main.weight")
            new_v = torch.zeros(list_num[2], list_num[1], 3, 3, 3)

            last_index_list = []
            for j in range(list_num[1]):
                location = index[1][j][1]
                last_index_list.append(location)

            index_list = []
            for j in range(list_num[2]):
                location = index[2][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list):
                    # print(f"i: {i}, m: {m}, j: {j}, n: {n}")

                    new_v[i][m] = v[j][n]
            state_dict["encoder3_m.main.weight"] = new_v

        if k == "encoder3_m.main.bias":
            print("encoder3_m.main.bias")
            new_v = torch.zeros(list_num[2])

            index_list = []
            for j in range(list_num[2]):
                location = index[2][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                new_v[i] = v[j]
            state_dict["encoder3_m.main.bias"] = new_v

        if k == "encoder4_m.main.weight":
            print("encoder4_m.main.weight")
            new_v = torch.zeros(list_num[3], list_num[2], 3, 3, 3)

            last_index_list = []
            for j in range(list_num[2]):
                location = index[2][j][1]
                last_index_list.append(location)

            index_list = []
            for j in range(list_num[3]):
                location = index[3][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list):
                    # print(f"i: {i}, m: {m}, j: {j}, n: {n}")

                    new_v[i][m] = v[j][n]
            state_dict["encoder4_m.main.weight"] = new_v

        if k == "encoder4_m.main.bias":
            print("encoder4_m.main.bias")
            new_v = torch.zeros(list_num[3])

            index_list = []
            for j in range(list_num[3]):
                location = index[3][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                new_v[i] = v[j]
            state_dict["encoder4_m.main.bias"] = new_v

        if k == "encoder5_m.main.weight":
            print("encoder5_m.main.weight")
            new_v = torch.zeros(list_num[4], list_num[3], 3, 3, 3)

            last_index_list = []
            for j in range(list_num[3]):
                location = index[3][j][1]
                last_index_list.append(location)

            index_list = []
            for j in range(list_num[4]):
                location = index[4][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list):
                    # print(f"i: {i}, m: {m}, j: {j}, n: {n}")

                    new_v[i][m] = v[j][n]
            state_dict["encoder5_m.main.weight"] = new_v

        if k == "encoder5_m.main.bias":
            print("encoder5_m.main.bias")
            new_v = torch.zeros(list_num[4])

            index_list = []
            for j in range(list_num[4]):
                location = index[4][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                new_v[i] = v[j]
            state_dict["encoder5_m.main.bias"] = new_v

        if k == "decoder1.main.weight":
            print("decoder1.main.weight")
            new_v = torch.zeros(32, 128, 3, 3, 3)  # v(64,128,3,3,3)
            new_v = v
            state_dict["decoder1.main.weight"] = new_v

        if k == "decoder1.main.bias":
            print("decoder1.main.bias")
            new_v = torch.zeros(32)
            new_v = v
            state_dict["decoder1.main.bias"] = new_v

        if k == "decoder2.main.weight":
            print("decoder2.main.weight")
            new_v = torch.zeros(32, 160, 3, 3, 3)  # v(64,128,3,3,3)
            new_v = v
            state_dict["decoder2.main.weight"] = new_v

        if k == "decoder2.main.bias":
            print("decoder2.main.bias")
            new_v = torch.zeros(32)
            new_v = v
            state_dict["decoder2.main.bias"] = new_v

        if k == "decoder3.main.weight":
            print("decoder3.main.weight")
            new_v = torch.zeros(32, 32 + list_num[8] + list_num[8], 3, 3, 3)  # v(64,128,3,3,3)

            last_index_list_0 = []
            for j in range(list_num[8]):
                location = index[8][j][1]
                last_index_list_0.append(location)

            for i in range(32):
                for j in range(32):
                    new_v[i][j] = v[i][j]

            for i in range(32):
                for m, n in enumerate(last_index_list_0):  # 16
                    new_v[i][m + 32] = v[i][n + 32]

            for i in range(32):
                for m, n in enumerate(last_index_list_0):  # 9
                    new_v[i][m + list_num[8] + 32] = v[i][n + 32 + 64]

            state_dict["decoder3.main.weight"] = new_v

        if k == "decoder3.main.bias":
            print("decoder3.main.bias")
            new_v = torch.zeros(32)
            new_v = v
            state_dict["decoder3.main.bias"] = new_v

        if k == "decoder4.main.weight":
            print("decoder4.main.weight")
            new_v = torch.zeros(16, list_num[10] + list_num[10] + 32, 3, 3, 3)  # v(64,128,3,3,3)

            last_index_list_0 = []
            for j in range(list_num[10]):
                location = index[10][j][1]
                last_index_list_0.append(location)

            for i in range(16):
                for j in range(32):
                    new_v[i][j] = v[i][j]

            for i in range(16):
                for m, n in enumerate(last_index_list_0):  # 16
                    new_v[i][m + 32] = v[i][n + 32]

            for i in range(16):
                for m, n in enumerate(last_index_list_0):  # 9
                    new_v[i][m + list_num[10] + 32] = v[i][n + 32 + 64]

            state_dict["decoder4.main.weight"] = new_v

        if k == "decoder4.main.bias":
            print("decoder4.main.bias")
            new_v = torch.zeros(16)
            new_v = v
            state_dict["decoder4.main.bias"] = new_v

        if k == "decoder5.main.weight":
            print("decoder5.main.weight")
            new_v = torch.zeros(16, 16 + list_num[0] + list_num[0], 3, 3, 3)  # v (32,96,3,3,3)

            last_index_list_0 = []
            for j in range(list_num[0]):
                location = index[0][j][1]
                last_index_list_0.append(location)

            for i in range(16):
                for j in range(16):
                    new_v[i][j] = v[i][j]

            for i in range(16):
                for m, n in enumerate(last_index_list_0):  # 16
                    new_v[i][m + 16] = v[i][n + 16]

            for i in range(16):
                for m, n in enumerate(last_index_list_0):  # 9
                    new_v[i][m + list_num[0] + 16] = v[i][n + 32 + 16]

            state_dict["decoder5.main.weight"] = new_v

        if k == "decoder5.main.bias":
            print("decoder5.main.bias")
            new_v = v
            state_dict["decoder5.main.bias"] = new_v

        if k == "fusion_m_scale4.main.weight":
            print("fusion_m_scale4.main.weight")
            new_v = torch.zeros(64, list_num[0] + list_num[1] + list_num[2] + list_num[3], 3, 3, 3)  # v (64,224,3,3,3)

            last_index_list_0 = []
            for j in range(list_num[0]):
                location = index[0][j][1]
                last_index_list_0.append(location)

            last_index_list_1 = []
            for j in range(list_num[1]):
                location = index[1][j][1]
                last_index_list_1.append(location)

            last_index_list_2 = []
            for j in range(list_num[2]):
                location = index[2][j][1]
                last_index_list_2.append(location)

            last_index_list_3 = []
            for j in range(list_num[3]):
                location = index[3][j][1]
                last_index_list_3.append(location)

            index_list = []
            for j in range(list_num[5]):
                location = index[5][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_0):  # 9
                    # print(f"i: {i}, m: {m}, j: {j}, n: {n}")
                    new_v[i][m] = v[j][n]

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_1):  # 12
                    new_v[i][m + list_num[0]] = v[j][n + 32]

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_2):  # 2
                    new_v[i][m + list_num[0] + list_num[1]] = v[j][n + 32 + 64]

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_3):  # 1
                    new_v[i][m + list_num[0] + list_num[1] + list_num[2]] = v[j][n + 32 + 64 + 64]

            state_dict["fusion_m_scale4.main.weight"] = new_v

        if k == "fusion_m_scale4.main.bias":
            print("fusion_m_scale4.main.bias")
            new_v = torch.zeros(64)
            new_v = v
            state_dict["fusion_m_scale4.main.bias"] = new_v

        if k == "fusion_m_scale3.main.weight":
            print("fusion_m_scale3.main.weight")
            new_v = torch.zeros(list_num[7], list_num[0] + list_num[1] + list_num[2] + list_num[3], 3, 3,
                                3)  # v (64,224,3,3,3)

            last_index_list_0 = []
            for j in range(list_num[0]):
                location = index[0][j][1]
                last_index_list_0.append(location)

            last_index_list_1 = []
            for j in range(list_num[1]):
                location = index[1][j][1]
                last_index_list_1.append(location)

            last_index_list_2 = []
            for j in range(list_num[2]):
                location = index[2][j][1]
                last_index_list_2.append(location)

            last_index_list_3 = []
            for j in range(list_num[3]):
                location = index[3][j][1]
                last_index_list_3.append(location)

            index_list = []
            for j in range(list_num[7]):
                location = index[7][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_0):  # 9
                    # print(f"i: {i}, m: {m}, j: {j}, n: {n}")
                    new_v[i][m] = v[j][n]

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_1):  # 12
                    new_v[i][m + list_num[0]] = v[j][n + 32]

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_2):  # 2
                    new_v[i][m + list_num[0] + list_num[1]] = v[j][n + 32 + 64]

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_3):  # 1
                    new_v[i][m + list_num[0] + list_num[1] + list_num[2]] = v[j][n + 32 + 64 + 64]

            state_dict["fusion_m_scale3.main.weight"] = new_v

        if k == "fusion_m_scale3.main.bias":
            print("fusion_m_scale3.main.bias")
            new_v = torch.zeros(list_num[7])
            index_list = []
            for j in range(list_num[7]):
                location = index[7][j][1]
                index_list.append(location)
                # print(location)
            for i, j in enumerate(index_list):
                new_v[i] = v[j]
            state_dict["fusion_m_scale3.main.bias"] = new_v

        if k == "fusion_m_scale2.main.weight":
            print("fusion_m_scale2.main.weight")
            new_v = torch.zeros(list_num[9], list_num[0] + list_num[1] + list_num[2] + list_num[3], 3, 3,
                                3)  # v (64,224,3,3,3)

            last_index_list_0 = []
            for j in range(list_num[0]):
                location = index[0][j][1]
                last_index_list_0.append(location)

            last_index_list_1 = []
            for j in range(list_num[1]):
                location = index[1][j][1]
                last_index_list_1.append(location)

            last_index_list_2 = []
            for j in range(list_num[2]):
                location = index[2][j][1]
                last_index_list_2.append(location)

            last_index_list_3 = []
            for j in range(list_num[3]):
                location = index[3][j][1]
                last_index_list_3.append(location)

            index_list = []
            for j in range(list_num[9]):
                location = index[9][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_0):  # 9
                    # print(f"i: {i}, m: {m}, j: {j}, n: {n}")
                    new_v[i][m] = v[j][n]

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_1):  # 12
                    new_v[i][m + list_num[0]] = v[j][n + 32]

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_2):  # 2
                    new_v[i][m + list_num[0] + list_num[1]] = v[j][n + 32 + 64]

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_3):  # 1
                    new_v[i][m + list_num[0] + list_num[1] + list_num[2]] = v[j][n + 32 + 64 + 64]

            state_dict["fusion_m_scale2.main.weight"] = new_v

        if k == "fusion_m_scale2.main.bias":
            print("fusion_m_scale2.main.bias")
            new_v = torch.zeros(list_num[9])
            index_list = []
            for j in range(list_num[9]):
                location = index[9][j][1]
                index_list.append(location)
                # print(location)
            for i, j in enumerate(index_list):
                new_v[i] = v[j]
            state_dict["fusion_m_scale2.main.bias"] = new_v

        if k == "fusion_m_scale4_2.main.weight":
            print("fusion_m_scale4_2.main.weight")
            new_v = torch.zeros(64, 64, 3, 3, 3)

            last_index_list = []
            for j in range(list_num[5]):
                location = index[5][j][1]
                last_index_list.append(location)

            index_list = []
            for j in range(list_num[6]):
                location = index[6][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list):
                    # print(f"i: {i}, m: {m}, j: {j}, n: {n}")

                    new_v[i][m] = v[j][n]
            state_dict["fusion_m_scale4_2.main.weight"] = new_v

        if k == "fusion_m_scale4_2.main.bias":
            print("fusion_m_scale4_2.main.bias")
            new_v = torch.zeros(64)

            index_list = []
            for j in range(list_num[6]):
                location = index[6][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                new_v[i] = v[j]
            state_dict["fusion_m_scale4_2.main.bias"] = new_v

        if k == "fusion_m_scale3_2.main.weight":
            print("fusion_m_scale3_2.main.weight")
            new_v = torch.zeros(list_num[8], list_num[7], 3, 3, 3)

            last_index_list = []
            for j in range(list_num[7]):
                location = index[7][j][1]
                last_index_list.append(location)

            index_list = []
            for j in range(list_num[8]):
                location = index[8][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list):
                    # print(f"i: {i}, m: {m}, j: {j}, n: {n}")

                    new_v[i][m] = v[j][n]
            state_dict["fusion_m_scale3_2.main.weight"] = new_v

        if k == "fusion_m_scale3_2.main.bias":
            print("fusion_m_scale3_2.main.bias")
            new_v = torch.zeros(list_num[8])

            index_list = []
            for j in range(list_num[8]):
                location = index[8][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                new_v[i] = v[j]
            state_dict["fusion_m_scale3_2.main.bias"] = new_v

        if k == "fusion_m_scale2_2.main.weight":
            print("fusion_m_scale2_2.main.weight")
            new_v = torch.zeros(list_num[10], list_num[9], 3, 3, 3)

            last_index_list = []
            for j in range(list_num[9]):
                location = index[9][j][1]
                last_index_list.append(location)

            index_list = []
            for j in range(list_num[10]):
                location = index[10][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list):
                    # print(f"i: {i}, m: {m}, j: {j}, n: {n}")

                    new_v[i][m] = v[j][n]
            state_dict["fusion_m_scale2_2.main.weight"] = new_v

        if k == "fusion_m_scale2_2.main.bias":
            print("fusion_m_scale2_2.main.bias")
            new_v = torch.zeros(list_num[10])

            index_list = []
            for j in range(list_num[10]):
                location = index[10][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                new_v[i] = v[j]
            state_dict["fusion_m_scale2_2.main.bias"] = new_v

        if k == "output_block_1.0.main.weight":
            new_v = v
            state_dict["output_block_1.0.main.weight"] = new_v

        if k == "output_block_1.0.main.bias":
            new_v = v
            state_dict["output_block_1.0.main.bias"] = new_v

        if k == "output_block_1.1.main.weight":
            new_v = v
            state_dict["output_block_1.1.main.weight"] = new_v

        if k == "output_block_1.1.main.bias":
            new_v = v
            state_dict["output_block_1.1.main.bias"] = new_v

        if k == "output_block_2.0.main.weight":
            new_v = v
            state_dict["output_block_2.0.main.weight"] = new_v

        if k == "output_block_2.0.main.bias":
            new_v = v
            state_dict["output_block_2.0.main.bias"] = new_v

        if k == "output_block_2.1.main.weight":
            new_v = v
            state_dict["output_block_2.1.main.weight"] = new_v

        if k == "output_block_2.1.main.bias":
            new_v = v
            state_dict["output_block_2.1.main.bias"] = new_v

        if k == "output_block_3.0.main.weight":
            new_v = v
            state_dict["output_block_3.0.main.weight"] = new_v

        if k == "output_block_3.0.main.bias":
            new_v = v
            state_dict["output_block_3.0.main.bias"] = new_v

        if k == "output_block_3.1.main.weight":
            new_v = v
            state_dict["output_block_3.1.main.weight"] = new_v

        if k == "output_block_3.1.main.bias":
            new_v = v
            state_dict["output_block_3.1.main.bias"] = new_v

        if k == "output_block.0.main.weight":
            new_v = v
            state_dict["output_block.0.main.weight"] = new_v

        if k == "output_block.0.main.bias":
            new_v = v
            state_dict["output_block.0.main.bias"] = new_v

        if k == "output_block.1.main.weight":
            new_v = v
            state_dict["output_block.1.main.weight"] = new_v

        if k == "output_block.1.main.bias":
            new_v = v
            state_dict["output_block.1.main.bias"] = new_v

        if k == "flow.weight":
            new_v = v
            state_dict["flow.weight"] = new_v

        if k == "flow.bias":
            new_v = v
            state_dict["flow.bias"] = new_v

    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

    # # 统计参数量
    dummy_input = torch.randn(1, 1, 192, 160, 192).to(device)
    flops, params = profile(model, (dummy_input, dummy_input,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))

    # # 对验证集进行测试
    Dice = 0
    Dice_1 = 0
    Dice_2 = 0
    Dice_3 = 0
    mean_dice = 0
    num = 0
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = torch.zeros(900)
    model.eval()
    print("---------验证集的Dice----------")
    SPT = vxm.torch.layers.SpatialTransformer(inshape, mode="nearest").to(device)
    for step in range(len(val_t1_files)):
        # 提取t1数据
        inputs_fixed, y_true = next(val_t1_generator)
        inputs_fixed = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs_fixed]
        # 提取对应t1的seg数据
        seg_inputs_fixed, seg_y_true = next(val_seg_generator)
        seg_inputs_fixed = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in seg_inputs_fixed]
        for step in range(len(val_t1_files)):
            # 提取t1数据
            inputs_moving, y_true = next(val_t1_generator)
            inputs_moving = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs_moving]
            # 提取对应t1的seg数据
            seg_inputs_moving, seg_y_true = next(val_seg_generator)
            seg_inputs_moving = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in
                                 seg_inputs_moving]
            if step < len(val_t1_files) - 1:
                with torch.no_grad():
                    # 生成flow
                    starter.record()

                    flow_1, flow_2, flow_3, flow, delta_flow_2, delta_flow_3, _, _, _ = model(inputs_moving[0],
                                                                                              inputs_fixed[0])
                    # flow_1,flow_2,flow_3,flow,delta_flow_2,delta_flow_3,_,_,_ = model_pretrain(inputs_moving[0],inputs_fixed[0])

                    ender.record()
                    # 同步GPU时间
                    torch.cuda.synchronize()
                    curr_time = starter.elapsed_time(ender)  # 计算时间
                    times[num] = curr_time
                    # num = num + 1
                    print(curr_time)

                    # x_in = torch.cat((inputs_moving[0],inputs_fixed[0]), dim=1)
                    # flow_1,flow_2,flow_3,flow,delta_flow_2,delta_flow_3,_ = model(inputs_moving[0],inputs_fixed[0])

                    # 上采样
                    flow_1_up = nn.functional.interpolate(flow_1, scale_factor=8, mode="trilinear") * 8
                    flow_2_up = nn.functional.interpolate(flow_2, scale_factor=4, mode="trilinear") * 4
                    flow_3_up = nn.functional.interpolate(flow_3, scale_factor=2, mode="trilinear") * 2

                    seg_pre_1 = SPT(seg_inputs_moving[0], flow_1_up)
                    seg_pre_2 = SPT(seg_inputs_moving[0], flow_2_up)
                    seg_pre_3 = SPT(seg_inputs_moving[0], flow_3_up)
                    dice_1, _ = vxm.torch.losses.compute_dice(seg_pre_1, seg_inputs_fixed[0])
                    dice_2, _ = vxm.torch.losses.compute_dice(seg_pre_2, seg_inputs_fixed[0])
                    dice_3, _ = vxm.torch.losses.compute_dice(seg_pre_3, seg_inputs_fixed[0])

                    # 得到变形场2
                    seg_pre = SPT(seg_inputs_moving[0], flow)
                    dice, dice_region = vxm.torch.losses.compute_dice(seg_pre, seg_inputs_fixed[0])
                    if dice < 1:
                        print("dice", dice)
                        Dice += dice
                        num += 1
        print("----------换fixed image---------")

    metric = Dice / num
    print("mean_dice", metric)
    mean_time = times.mean().item()
    print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000 / mean_time))

    torch.save(model.state_dict(),
               "/home/boys/project/voxelmorph/voxelmorph_code/models/comparison_methods/abdomenCT/daul_pyramid_PDFNet_FFM_large_GDP_adaptive_nopretrain.pth")


# 将大模型的参数赋值给小模型并保存 normal
def test_dual_pyramid_vxm_FFM_adaptive_normal(config=config):
    print("Training dual_pyramid_vxm_plus")

    # 读取huge model
    model_pretrain = vxm.networks.dual_pyramid_VxmDense_FFM_normal_GDP()

    checkpoint = torch.load(
        "./daul_pyramid_vxm_FFM_4layer_4_normal_GDP_Lambda0.0036_theta4_decay0.5_decayrate0.98_nopretrain_abdomenmrct_919.pth")

    model_pretrain.to(device)
    model_dict = model_pretrain.state_dict()
    state_dict = {k: v for k, v in checkpoint.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model_pretrain.load_state_dict(model_dict)

    # model_pretrain.train()
    decay = 0.5 * (0.98 ** 919)
    # decay = 1
    model_pretrain.set_decay(decay)

    inputs, y_true = next(seg_generator)
    inputs = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs]
    y_true = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in y_true]

    # 生成flow
    with torch.no_grad():
        # 得到非零索引
        flow_1, flow_2, flow_3, flow_final, delta_flow_2, delta_flow_3, _, ori_alpha, gate_alpha = model_pretrain(
            inputs[0], inputs[1])

        # print("gate_alpha",gate_alpha)
    # print("ori_aplha",ori_alpha)
    # print(ori_alpha)

    index = []
    for tensor in gate_alpha:
        indices = torch.nonzero(tensor).squeeze()  # 获取非零元素的索引
        index.append(indices)
    print(index[0])
    print(index[1])
    print(index[2])
    print(index[3])
    print(len(index[4]))
    print(len(index[5]))
    print(len(index[6]))
    print(len(index[7]))
    print(len(index[8]))
    print(len(index[9]))
    print(len(index[10]))

    list_num = [len(index[0]), len(index[1]), len(index[2]), len(index[3]), len(index[4]), len(index[5]), len(index[6]),
                len(index[7]), len(index[8]), len(index[9]), len(index[10])]
    new_alpha = []
    for i in range(len(list_num)):
        ori_alpha_i = ori_alpha[i]
        new_alpha_i = torch.ones(1, list_num[i])
        for j in range(list_num[i]):
            # if i == 2:
            #     index[2] = torch.unsqueeze(index[2], 0)
            #     print(index[2])
            # print(i)
            # print(j)
            location = index[i][j][1]
            new_alpha_i[0][j] = ori_alpha_i[0][location]
        new_alpha.append(new_alpha_i)

    # # 初始化一个轻量的小模型
    # model = vxm.networks.dual_pyramid_VxmDense_FFM_4layer_4_huge_adaptive_val_2(alpha1=new_alpha[0], alpha2=new_alpha[1],
    #         alpha3=new_alpha[2], alpha4=new_alpha[3],alpha5=new_alpha[4], alpha6_1=new_alpha[5],alpha6_2=new_alpha[6],
    #         alpha7_1=new_alpha[7], alpha7_2=new_alpha[8],alpha8_1=new_alpha[9],alpha8_2=new_alpha[10],list_num=list_num)

    # model.set_decay(decay)

    model = vxm.networks.dual_pyramid_VxmDense_FFM_normal_adaptive_val(list_num=list_num)

    model.to(device)
    model_dict = model.state_dict()

    # # 验证回传是否正确
    # with torch.no_grad():
    #     # 得到非零索引
    #     flow_1,flow_2,flow_3,flow_final,delta_flow_2,delta_flow_3,_,ori_alpha,gate_alpha_new = model(inputs[0],inputs[1])
    #     print(gate_alpha_new)

    # index = index1

    state_dict = {}
    # # 加载参数
    for k, v in checkpoint.items():
        # print(k)
        # print(v.shape)

        if k == "encoder1_m.main.weight":
            print("encoder1_m.main.weight")
            new_v = torch.zeros(list_num[0], 1, 3, 3, 3)
            index_list = []
            for j in range(list_num[0]):
                location = index[0][j][1]
                index_list.append(location)
                # print(location)
            for i, j in enumerate(index_list):
                new_v[i] = v[j]
            state_dict["encoder1_m.main.weight"] = new_v

        if k == "encoder1_m.main.bias":
            print("encoder1_m.main.bias")

            new_v = torch.zeros(list_num[0])
            index_list = []
            for j in range(list_num[0]):
                location = index[0][j][1]
                index_list.append(location)
                # print(location)
            for i, j in enumerate(index_list):
                new_v[i] = v[j]
            state_dict["encoder1_m.main.bias"] = new_v

        if k == "encoder2_m.main.weight":
            print("encoder2_m.main.weight")
            new_v = torch.zeros(list_num[1], list_num[0], 3, 3, 3)

            last_index_list = []
            for j in range(list_num[0]):
                location = index[0][j][1]
                last_index_list.append(location)

            index_list = []
            for j in range(list_num[1]):
                location = index[1][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list):
                    # print(f"i: {i}, m: {m}, j: {j}, n: {n}")

                    new_v[i][m] = v[j][n]
            state_dict["encoder2_m.main.weight"] = new_v

        if k == "encoder2_m.main.bias":
            print("encoder2_m.main.bias")
            # print(v.shape)
            new_v = torch.zeros(list_num[1])

            index_list = []
            for j in range(list_num[1]):
                location = index[1][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                new_v[i] = v[j]
            state_dict["encoder2_m.main.bias"] = new_v

        if k == "encoder3_m.main.weight":
            print("encoder3_m.main.weight")
            new_v = torch.zeros(list_num[2], list_num[1], 3, 3, 3)

            last_index_list = []
            for j in range(list_num[1]):
                location = index[1][j][1]
                last_index_list.append(location)

            index_list = []
            for j in range(list_num[2]):
                location = index[2][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list):
                    # print(f"i: {i}, m: {m}, j: {j}, n: {n}")

                    new_v[i][m] = v[j][n]
            state_dict["encoder3_m.main.weight"] = new_v

        if k == "encoder3_m.main.bias":
            print("encoder3_m.main.bias")
            new_v = torch.zeros(list_num[2])

            index_list = []
            for j in range(list_num[2]):
                location = index[2][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                new_v[i] = v[j]
            state_dict["encoder3_m.main.bias"] = new_v

        if k == "encoder4_m.main.weight":
            print("encoder4_m.main.weight")
            new_v = torch.zeros(list_num[3], list_num[2], 3, 3, 3)

            last_index_list = []
            for j in range(list_num[2]):
                location = index[2][j][1]
                last_index_list.append(location)

            index_list = []
            for j in range(list_num[3]):
                location = index[3][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list):
                    # print(f"i: {i}, m: {m}, j: {j}, n: {n}")

                    new_v[i][m] = v[j][n]
            state_dict["encoder4_m.main.weight"] = new_v

        if k == "encoder4_m.main.bias":
            print("encoder4_m.main.bias")
            new_v = torch.zeros(list_num[3])

            index_list = []
            for j in range(list_num[3]):
                location = index[3][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                new_v[i] = v[j]
            state_dict["encoder4_m.main.bias"] = new_v

        if k == "encoder5_m.main.weight":
            print("encoder5_m.main.weight")
            new_v = torch.zeros(list_num[4], list_num[3], 3, 3, 3)

            last_index_list = []
            for j in range(list_num[3]):
                location = index[3][j][1]
                last_index_list.append(location)

            index_list = []
            for j in range(list_num[4]):
                location = index[4][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list):
                    # print(f"i: {i}, m: {m}, j: {j}, n: {n}")

                    new_v[i][m] = v[j][n]
            state_dict["encoder5_m.main.weight"] = new_v

        if k == "encoder5_m.main.bias":
            print("encoder5_m.main.bias")
            new_v = torch.zeros(list_num[4])

            index_list = []
            for j in range(list_num[4]):
                location = index[4][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                new_v[i] = v[j]
            state_dict["encoder5_m.main.bias"] = new_v

        if k == "decoder1.main.weight":
            print("decoder1.main.weight")
            new_v = torch.zeros(32, 64, 3, 3, 3)  # v(64,128,3,3,3)
            new_v = v
            state_dict["decoder1.main.weight"] = new_v

        if k == "decoder1.main.bias":
            print("decoder1.main.bias")
            new_v = torch.zeros(32)
            new_v = v
            state_dict["decoder1.main.bias"] = new_v

        if k == "decoder2.main.weight":
            print("decoder2.main.weight")
            new_v = torch.zeros(32, 96, 3, 3, 3)  # v(64,128,3,3,3)
            new_v = v
            state_dict["decoder2.main.weight"] = new_v

        if k == "decoder2.main.bias":
            print("decoder2.main.bias")
            new_v = torch.zeros(32)
            new_v = v
            state_dict["decoder2.main.bias"] = new_v

        if k == "decoder3.main.weight":
            print("decoder3.main.weight")
            new_v = torch.zeros(32, 32 + list_num[8] + list_num[8], 3, 3, 3)  # v(64,128,3,3,3)

            last_index_list_0 = []
            for j in range(list_num[8]):
                location = index[8][j][1]
                last_index_list_0.append(location)

            for i in range(32):
                for j in range(32):
                    new_v[i][j] = v[i][j]

            for i in range(32):
                for m, n in enumerate(last_index_list_0):  # 16
                    new_v[i][m + 32] = v[i][n + 32]

            for i in range(32):
                for m, n in enumerate(last_index_list_0):  # 9
                    new_v[i][m + list_num[8] + 32] = v[i][n + 32 + 32]

            state_dict["decoder3.main.weight"] = new_v

        if k == "decoder3.main.bias":
            print("decoder3.main.bias")
            new_v = torch.zeros(32)
            new_v = v
            state_dict["decoder3.main.bias"] = new_v

        if k == "decoder4.main.weight":
            print("decoder4.main.weight")
            new_v = torch.zeros(16, list_num[10] + list_num[10] + 32, 3, 3, 3)  # v(64,128,3,3,3)

            last_index_list_0 = []
            for j in range(list_num[10]):
                location = index[10][j][1]
                last_index_list_0.append(location)

            for i in range(16):
                for j in range(32):
                    new_v[i][j] = v[i][j]

            for i in range(16):
                for m, n in enumerate(last_index_list_0):  # 16
                    new_v[i][m + 32] = v[i][n + 32]

            for i in range(16):
                for m, n in enumerate(last_index_list_0):  # 9
                    new_v[i][m + list_num[10] + 32] = v[i][n + 32 + 32]

            state_dict["decoder4.main.weight"] = new_v

        if k == "decoder4.main.bias":
            print("decoder4.main.bias")
            new_v = torch.zeros(16)
            new_v = v
            state_dict["decoder4.main.bias"] = new_v

        if k == "decoder5.main.weight":
            print("decoder5.main.weight")
            new_v = torch.zeros(16, 16 + list_num[0] + list_num[0], 3, 3, 3)  # v (32,96,3,3,3)

            last_index_list_0 = []
            for j in range(list_num[0]):
                location = index[0][j][1]
                last_index_list_0.append(location)

            for i in range(16):
                for j in range(16):
                    new_v[i][j] = v[i][j]

            for i in range(16):
                for m, n in enumerate(last_index_list_0):  # 16
                    new_v[i][m + 16] = v[i][n + 16]

            for i in range(16):
                for m, n in enumerate(last_index_list_0):  # 9
                    new_v[i][m + list_num[0] + 16] = v[i][n + 16 + 16]

            state_dict["decoder5.main.weight"] = new_v

        if k == "decoder5.main.bias":
            print("decoder5.main.bias")
            new_v = v
            state_dict["decoder5.main.bias"] = new_v

        if k == "fusion_m_scale4.main.weight":
            print("fusion_m_scale4.main.weight")
            new_v = torch.zeros(32, list_num[0] + list_num[1] + list_num[2] + list_num[3], 3, 3, 3)  # v (64,224,3,3,3)

            last_index_list_0 = []
            for j in range(list_num[0]):
                location = index[0][j][1]
                last_index_list_0.append(location)

            last_index_list_1 = []
            for j in range(list_num[1]):
                location = index[1][j][1]
                last_index_list_1.append(location)

            last_index_list_2 = []
            for j in range(list_num[2]):
                location = index[2][j][1]
                last_index_list_2.append(location)

            last_index_list_3 = []
            for j in range(list_num[3]):
                location = index[3][j][1]
                last_index_list_3.append(location)

            index_list = []
            for j in range(list_num[5]):
                location = index[5][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_0):  # 9
                    # print(f"i: {i}, m: {m}, j: {j}, n: {n}")
                    new_v[i][m] = v[j][n]

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_1):  # 12
                    new_v[i][m + list_num[0]] = v[j][n + 16]

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_2):  # 2
                    new_v[i][m + list_num[0] + list_num[1]] = v[j][n + 16 + 32]

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_3):  # 1
                    new_v[i][m + list_num[0] + list_num[1] + list_num[2]] = v[j][n + 16 + 32 + 32]

            state_dict["fusion_m_scale4.main.weight"] = new_v

        if k == "fusion_m_scale4.main.bias":
            print("fusion_m_scale4.main.bias")
            new_v = torch.zeros(32)
            new_v = v
            state_dict["fusion_m_scale4.main.bias"] = new_v

        if k == "fusion_m_scale3.main.weight":
            print("fusion_m_scale3.main.weight")
            new_v = torch.zeros(list_num[7], list_num[0] + list_num[1] + list_num[2] + list_num[3], 3, 3,
                                3)  # v (64,224,3,3,3)

            last_index_list_0 = []
            for j in range(list_num[0]):
                location = index[0][j][1]
                last_index_list_0.append(location)

            last_index_list_1 = []
            for j in range(list_num[1]):
                location = index[1][j][1]
                last_index_list_1.append(location)

            last_index_list_2 = []
            for j in range(list_num[2]):
                location = index[2][j][1]
                last_index_list_2.append(location)

            last_index_list_3 = []
            for j in range(list_num[3]):
                location = index[3][j][1]
                last_index_list_3.append(location)

            index_list = []
            for j in range(list_num[7]):
                location = index[7][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_0):  # 9
                    # print(f"i: {i}, m: {m}, j: {j}, n: {n}")
                    new_v[i][m] = v[j][n]

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_1):  # 12
                    new_v[i][m + list_num[0]] = v[j][n + 16]

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_2):  # 2
                    new_v[i][m + list_num[0] + list_num[1]] = v[j][n + 16 + 32]

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_3):  # 1
                    new_v[i][m + list_num[0] + list_num[1] + list_num[2]] = v[j][n + 16 + 32 + 32]

            state_dict["fusion_m_scale3.main.weight"] = new_v

        if k == "fusion_m_scale3.main.bias":
            print("fusion_m_scale3.main.bias")
            new_v = torch.zeros(list_num[7])
            index_list = []
            for j in range(list_num[7]):
                location = index[7][j][1]
                index_list.append(location)
                # print(location)
            for i, j in enumerate(index_list):
                new_v[i] = v[j]
            state_dict["fusion_m_scale3.main.bias"] = new_v

        if k == "fusion_m_scale2.main.weight":
            print("fusion_m_scale2.main.weight")
            new_v = torch.zeros(list_num[9], list_num[0] + list_num[1] + list_num[2] + list_num[3], 3, 3,
                                3)  # v (64,224,3,3,3)

            last_index_list_0 = []
            for j in range(list_num[0]):
                location = index[0][j][1]
                last_index_list_0.append(location)

            last_index_list_1 = []
            for j in range(list_num[1]):
                location = index[1][j][1]
                last_index_list_1.append(location)

            last_index_list_2 = []
            for j in range(list_num[2]):
                location = index[2][j][1]
                last_index_list_2.append(location)

            last_index_list_3 = []
            for j in range(list_num[3]):
                location = index[3][j][1]
                last_index_list_3.append(location)

            index_list = []
            for j in range(list_num[9]):
                location = index[9][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_0):  # 9
                    # print(f"i: {i}, m: {m}, j: {j}, n: {n}")
                    new_v[i][m] = v[j][n]

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_1):  # 12
                    new_v[i][m + list_num[0]] = v[j][n + 16]

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_2):  # 2
                    new_v[i][m + list_num[0] + list_num[1]] = v[j][n + 16 + 32]

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_3):  # 1
                    new_v[i][m + list_num[0] + list_num[1] + list_num[2]] = v[j][n + 16 + 32 + 32]

            state_dict["fusion_m_scale2.main.weight"] = new_v

        if k == "fusion_m_scale2.main.bias":
            print("fusion_m_scale2.main.bias")
            new_v = torch.zeros(list_num[9])
            index_list = []
            for j in range(list_num[9]):
                location = index[9][j][1]
                index_list.append(location)
                # print(location)
            for i, j in enumerate(index_list):
                new_v[i] = v[j]
            state_dict["fusion_m_scale2.main.bias"] = new_v

        if k == "fusion_m_scale4_2.main.weight":
            print("fusion_m_scale4_2.main.weight")
            new_v = torch.zeros(32, 32, 3, 3, 3)

            last_index_list = []
            for j in range(list_num[5]):
                location = index[5][j][1]
                last_index_list.append(location)

            index_list = []
            for j in range(list_num[6]):
                location = index[6][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list):
                    # print(f"i: {i}, m: {m}, j: {j}, n: {n}")

                    new_v[i][m] = v[j][n]
            state_dict["fusion_m_scale4_2.main.weight"] = new_v

        if k == "fusion_m_scale4_2.main.bias":
            print("fusion_m_scale4_2.main.bias")
            new_v = torch.zeros(32)

            index_list = []
            for j in range(list_num[6]):
                location = index[6][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                new_v[i] = v[j]
            state_dict["fusion_m_scale4_2.main.bias"] = new_v

        if k == "fusion_m_scale3_2.main.weight":
            print("fusion_m_scale3_2.main.weight")
            new_v = torch.zeros(list_num[8], list_num[7], 3, 3, 3)

            last_index_list = []
            for j in range(list_num[7]):
                location = index[7][j][1]
                last_index_list.append(location)

            index_list = []
            for j in range(list_num[8]):
                location = index[8][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list):
                    # print(f"i: {i}, m: {m}, j: {j}, n: {n}")

                    new_v[i][m] = v[j][n]
            state_dict["fusion_m_scale3_2.main.weight"] = new_v

        if k == "fusion_m_scale3_2.main.bias":
            print("fusion_m_scale3_2.main.bias")
            new_v = torch.zeros(list_num[8])

            index_list = []
            for j in range(list_num[8]):
                location = index[8][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                new_v[i] = v[j]
            state_dict["fusion_m_scale3_2.main.bias"] = new_v

        if k == "fusion_m_scale2_2.main.weight":
            print("fusion_m_scale2_2.main.weight")
            new_v = torch.zeros(list_num[10], list_num[9], 3, 3, 3)

            last_index_list = []
            for j in range(list_num[9]):
                location = index[9][j][1]
                last_index_list.append(location)

            index_list = []
            for j in range(list_num[10]):
                location = index[10][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list):
                    # print(f"i: {i}, m: {m}, j: {j}, n: {n}")

                    new_v[i][m] = v[j][n]
            state_dict["fusion_m_scale2_2.main.weight"] = new_v

        if k == "fusion_m_scale2_2.main.bias":
            print("fusion_m_scale2_2.main.bias")
            new_v = torch.zeros(list_num[10])

            index_list = []
            for j in range(list_num[10]):
                location = index[10][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                new_v[i] = v[j]
            state_dict["fusion_m_scale2_2.main.bias"] = new_v

        if k == "output_block_1.0.main.weight":
            new_v = v
            state_dict["output_block_1.0.main.weight"] = new_v

        if k == "output_block_1.0.main.bias":
            new_v = v
            state_dict["output_block_1.0.main.bias"] = new_v

        if k == "output_block_1.1.main.weight":
            new_v = v
            state_dict["output_block_1.1.main.weight"] = new_v

        if k == "output_block_1.1.main.bias":
            new_v = v
            state_dict["output_block_1.1.main.bias"] = new_v

        if k == "output_block_2.0.main.weight":
            new_v = v
            state_dict["output_block_2.0.main.weight"] = new_v

        if k == "output_block_2.0.main.bias":
            new_v = v
            state_dict["output_block_2.0.main.bias"] = new_v

        if k == "output_block_2.1.main.weight":
            new_v = v
            state_dict["output_block_2.1.main.weight"] = new_v

        if k == "output_block_2.1.main.bias":
            new_v = v
            state_dict["output_block_2.1.main.bias"] = new_v

        if k == "output_block_3.0.main.weight":
            new_v = v
            state_dict["output_block_3.0.main.weight"] = new_v

        if k == "output_block_3.0.main.bias":
            new_v = v
            state_dict["output_block_3.0.main.bias"] = new_v

        if k == "output_block_3.1.main.weight":
            new_v = v
            state_dict["output_block_3.1.main.weight"] = new_v

        if k == "output_block_3.1.main.bias":
            new_v = v
            state_dict["output_block_3.1.main.bias"] = new_v

        if k == "output_block.0.main.weight":
            new_v = v
            state_dict["output_block.0.main.weight"] = new_v

        if k == "output_block.0.main.bias":
            new_v = v
            state_dict["output_block.0.main.bias"] = new_v

        if k == "output_block.1.main.weight":
            new_v = v
            state_dict["output_block.1.main.weight"] = new_v

        if k == "output_block.1.main.bias":
            new_v = v
            state_dict["output_block.1.main.bias"] = new_v

        if k == "flow.weight":
            new_v = v
            state_dict["flow.weight"] = new_v

        if k == "flow.bias":
            new_v = v
            state_dict["flow.bias"] = new_v

    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

    # # 统计参数量
    dummy_input = torch.randn(1, 1, 192, 160, 192).to(device)
    flops, params = profile(model, (dummy_input, dummy_input,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))

    # # 对验证集进行测试
    Dice = 0
    Dice_1 = 0
    Dice_2 = 0
    Dice_3 = 0
    mean_dice = 0
    num = 0
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = torch.zeros(900)
    model.eval()
    print("---------验证集的Dice----------")
    SPT = vxm.torch.layers.SpatialTransformer(inshape, mode="nearest").to(device)
    for step in range(len(val_t1_files)):
        # 提取t1数据
        inputs_fixed, y_true = next(val_t1_generator)
        inputs_fixed = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs_fixed]
        # 提取对应t1的seg数据
        seg_inputs_fixed, seg_y_true = next(val_seg_generator)
        seg_inputs_fixed = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in seg_inputs_fixed]
        for step in range(len(val_t1_files)):
            # 提取t1数据
            inputs_moving, y_true = next(val_t1_generator)
            inputs_moving = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs_moving]
            # 提取对应t1的seg数据
            seg_inputs_moving, seg_y_true = next(val_seg_generator)
            seg_inputs_moving = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in
                                 seg_inputs_moving]
            if step < len(val_t1_files) - 1:
                with torch.no_grad():
                    # 生成flow
                    starter.record()

                    flow_1, flow_2, flow_3, flow, delta_flow_2, delta_flow_3, _ = model(inputs_moving[0],
                                                                                        inputs_fixed[0])

                    # flow_1,flow_2,flow_3,flow,delta_flow_2,delta_flow_3,_,_,_ = model_pretrain(inputs_moving[0],inputs_fixed[0])

                    ender.record()
                    # 同步GPU时间
                    torch.cuda.synchronize()
                    curr_time = starter.elapsed_time(ender)  # 计算时间
                    times[num] = curr_time
                    # num = num + 1
                    print(curr_time)

                    # x_in = torch.cat((inputs_moving[0],inputs_fixed[0]), dim=1)
                    # flow_1,flow_2,flow_3,flow,delta_flow_2,delta_flow_3,_ = model(inputs_moving[0],inputs_fixed[0])

                    # 上采样
                    flow_1_up = nn.functional.interpolate(flow_1, scale_factor=8, mode="trilinear") * 8
                    flow_2_up = nn.functional.interpolate(flow_2, scale_factor=4, mode="trilinear") * 4
                    flow_3_up = nn.functional.interpolate(flow_3, scale_factor=2, mode="trilinear") * 2

                    seg_pre_1 = SPT(seg_inputs_moving[0], flow_1_up)
                    seg_pre_2 = SPT(seg_inputs_moving[0], flow_2_up)
                    seg_pre_3 = SPT(seg_inputs_moving[0], flow_3_up)
                    dice_1, _ = vxm.torch.losses.compute_dice(seg_pre_1, seg_inputs_fixed[0])
                    dice_2, _ = vxm.torch.losses.compute_dice(seg_pre_2, seg_inputs_fixed[0])
                    dice_3, _ = vxm.torch.losses.compute_dice(seg_pre_3, seg_inputs_fixed[0])

                    # 得到变形场2
                    seg_pre = SPT(seg_inputs_moving[0], flow)
                    dice, dice_region = vxm.torch.losses.compute_dice(seg_pre, seg_inputs_fixed[0])
                    if dice < 1:
                        print("dice", dice)
                        Dice += dice
                        num += 1
        print("----------换fixed image---------")

    metric = Dice / num
    print("mean_dice", metric)
    mean_time = times.mean().item()
    print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000 / mean_time))

    torch.save(model.state_dict(),
               "/home/boys/project/voxelmorph/voxelmorph_code/models/comparison_methods/abdomenCT/daul_pyramid_PDFNet_FFM_normal_GDP_adaptive_nopretrain.pth")


# 将大模型的参数赋值给小模型并保存 normal_diff
def test_dual_pyramid_vxm_FFM_adaptive_normal_diff(config=config):
    print("Training dual_pyramid_vxm_plus")

    # 读取huge model
    model_pretrain = vxm.networks.dual_pyramid_VxmDense_FFM_normal_diff_new_GDP()

    checkpoint = torch.load(
        "./daul_pyramid_baseline_PDFNet_normal_FFM_diff_GDP_804.pth")

    model_pretrain.to(device)
    model_dict = model_pretrain.state_dict()
    state_dict = {k: v for k, v in checkpoint.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model_pretrain.load_state_dict(model_dict)

    # model_pretrain.train()
    decay = 0.5 * (0.98 ** 804)
    # decay = 1
    model_pretrain.set_decay(decay)

    inputs, y_true = next(seg_generator)
    inputs = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs]
    y_true = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in y_true]

    # 生成flow
    with torch.no_grad():
        # 得到非零索引
        flow_1, flow_2, flow_3, flow_final, delta_flow_2, delta_flow_3, _, ori_alpha, gate_alpha = model_pretrain(
            inputs[0], inputs[1])

        # print("gate_alpha",gate_alpha)
    # print("ori_aplha",ori_alpha)
    # print(ori_alpha)

    index = []
    for tensor in gate_alpha:
        indices = torch.nonzero(tensor).squeeze()  # 获取非零元素的索引
        index.append(indices)
    print(index[0])
    print(index[1])
    print(index[2])
    print(index[3])
    print(len(index[4]))
    print(len(index[5]))
    print(len(index[6]))
    print(len(index[7]))
    print(len(index[8]))
    print(len(index[9]))
    print(len(index[10]))

    list_num = [len(index[0]), len(index[1]), len(index[2]), len(index[3]), len(index[4]), len(index[5]), len(index[6]),
                len(index[7]), len(index[8]), len(index[9]), len(index[10])]
    new_alpha = []
    for i in range(len(list_num)):
        ori_alpha_i = ori_alpha[i]
        new_alpha_i = torch.ones(1, list_num[i])
        for j in range(list_num[i]):
            # if i == 2:
            #     index[2] = torch.unsqueeze(index[2], 0)
            #     print(index[2])
            # # if i == 3:
            #     index[3] = torch.unsqueeze(index[3], 0)
            #     print(index[3])
            # print(i)
            # print(j)
            location = index[i][j][1]
            new_alpha_i[0][j] = ori_alpha_i[0][location]
        new_alpha.append(new_alpha_i)

    # # 初始化一个轻量的小模型
    # model = vxm.networks.dual_pyramid_VxmDense_FFM_4layer_4_huge_adaptive_val_2(alpha1=new_alpha[0], alpha2=new_alpha[1],
    #         alpha3=new_alpha[2], alpha4=new_alpha[3],alpha5=new_alpha[4], alpha6_1=new_alpha[5],alpha6_2=new_alpha[6],
    #         alpha7_1=new_alpha[7], alpha7_2=new_alpha[8],alpha8_1=new_alpha[9],alpha8_2=new_alpha[10],list_num=list_num)

    # model.set_decay(decay)

    model = vxm.networks.dual_pyramid_VxmDense_FFM_normal_diff_new_adaptive_val(list_num=list_num)

    model.to(device)
    model_dict = model.state_dict()

    # # 验证回传是否正确
    # with torch.no_grad():
    #     # 得到非零索引
    #     flow_1,flow_2,flow_3,flow_final,delta_flow_2,delta_flow_3,_,ori_alpha,gate_alpha_new = model(inputs[0],inputs[1])
    #     print(gate_alpha_new)

    # index = index1

    state_dict = {}
    # # # 加载参数
    for k, v in checkpoint.items():
        # print(k)
        # print(v.shape)

        if k == "encoder1_m.main.weight":
            print("encoder1_m.main.weight")
            new_v = torch.zeros(list_num[0], 1, 3, 3, 3)
            index_list = []
            for j in range(list_num[0]):
                location = index[0][j][1]
                index_list.append(location)
                # print(location)
            for i, j in enumerate(index_list):
                new_v[i] = v[j]
            state_dict["encoder1_m.main.weight"] = new_v

        if k == "encoder1_m.main.bias":
            print("encoder1_m.main.bias")

            new_v = torch.zeros(list_num[0])
            index_list = []
            for j in range(list_num[0]):
                location = index[0][j][1]
                index_list.append(location)
                # print(location)
            for i, j in enumerate(index_list):
                new_v[i] = v[j]
            state_dict["encoder1_m.main.bias"] = new_v

        if k == "encoder2_m.main.weight":
            print("encoder2_m.main.weight")
            new_v = torch.zeros(list_num[1], list_num[0], 3, 3, 3)

            last_index_list = []
            for j in range(list_num[0]):
                location = index[0][j][1]
                last_index_list.append(location)

            index_list = []
            for j in range(list_num[1]):
                location = index[1][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list):
                    # print(f"i: {i}, m: {m}, j: {j}, n: {n}")

                    new_v[i][m] = v[j][n]
            state_dict["encoder2_m.main.weight"] = new_v

        if k == "encoder2_m.main.bias":
            print("encoder2_m.main.bias")
            # print(v.shape)
            new_v = torch.zeros(list_num[1])

            index_list = []
            for j in range(list_num[1]):
                location = index[1][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                new_v[i] = v[j]
            state_dict["encoder2_m.main.bias"] = new_v

        if k == "encoder3_m.main.weight":
            print("encoder3_m.main.weight")
            new_v = torch.zeros(list_num[2], list_num[1], 3, 3, 3)

            last_index_list = []
            for j in range(list_num[1]):
                location = index[1][j][1]
                last_index_list.append(location)

            index_list = []
            for j in range(list_num[2]):
                location = index[2][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list):
                    # print(f"i: {i}, m: {m}, j: {j}, n: {n}")

                    new_v[i][m] = v[j][n]
            state_dict["encoder3_m.main.weight"] = new_v

        if k == "encoder3_m.main.bias":
            print("encoder3_m.main.bias")
            new_v = torch.zeros(list_num[2])

            index_list = []
            for j in range(list_num[2]):
                location = index[2][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                new_v[i] = v[j]
            state_dict["encoder3_m.main.bias"] = new_v

        if k == "encoder4_m.main.weight":
            print("encoder4_m.main.weight")
            new_v = torch.zeros(list_num[3], list_num[2], 3, 3, 3)

            last_index_list = []
            for j in range(list_num[2]):
                location = index[2][j][1]
                last_index_list.append(location)

            index_list = []
            for j in range(list_num[3]):
                location = index[3][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list):
                    # print(f"i: {i}, m: {m}, j: {j}, n: {n}")

                    new_v[i][m] = v[j][n]
            state_dict["encoder4_m.main.weight"] = new_v

        if k == "encoder4_m.main.bias":
            print("encoder4_m.main.bias")
            new_v = torch.zeros(list_num[3])

            index_list = []
            for j in range(list_num[3]):
                location = index[3][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                new_v[i] = v[j]
            state_dict["encoder4_m.main.bias"] = new_v

        if k == "encoder5_m.main.weight":
            print("encoder5_m.main.weight")
            new_v = torch.zeros(list_num[4], list_num[3], 3, 3, 3)

            last_index_list = []
            for j in range(list_num[3]):
                location = index[3][j][1]
                last_index_list.append(location)

            index_list = []
            for j in range(list_num[4]):
                location = index[4][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list):
                    # print(f"i: {i}, m: {m}, j: {j}, n: {n}")

                    new_v[i][m] = v[j][n]
            state_dict["encoder5_m.main.weight"] = new_v

        if k == "encoder5_m.main.bias":
            print("encoder5_m.main.bias")
            new_v = torch.zeros(list_num[4])

            index_list = []
            for j in range(list_num[4]):
                location = index[4][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                new_v[i] = v[j]
            state_dict["encoder5_m.main.bias"] = new_v

        if k == "decoder1.main.weight":
            print("decoder1.main.weight")
            new_v = torch.zeros(32, 64, 3, 3, 3)  # v(64,128,3,3,3)
            new_v = v
            state_dict["decoder1.main.weight"] = new_v

        if k == "decoder1.main.bias":
            print("decoder1.main.bias")
            new_v = torch.zeros(32)
            new_v = v
            state_dict["decoder1.main.bias"] = new_v

        if k == "decoder2.main.weight":
            print("decoder2.main.weight")
            new_v = torch.zeros(32, 96, 3, 3, 3)  # v(64,128,3,3,3)
            new_v = v
            state_dict["decoder2.main.weight"] = new_v

        if k == "decoder2.main.bias":
            print("decoder2.main.bias")
            new_v = torch.zeros(32)
            new_v = v
            state_dict["decoder2.main.bias"] = new_v

        if k == "decoder3.main.weight":
            print("decoder3.main.weight")
            new_v = torch.zeros(32, 32 + list_num[8] + list_num[8], 3, 3, 3)  # v(64,128,3,3,3)

            last_index_list_0 = []
            for j in range(list_num[8]):
                location = index[8][j][1]
                last_index_list_0.append(location)

            for i in range(32):
                for j in range(32):
                    new_v[i][j] = v[i][j]

            for i in range(32):
                for m, n in enumerate(last_index_list_0):  # 16
                    new_v[i][m + 32] = v[i][n + 32]

            for i in range(32):
                for m, n in enumerate(last_index_list_0):  # 9
                    new_v[i][m + list_num[8] + 32] = v[i][n + 32 + 32]

            state_dict["decoder3.main.weight"] = new_v

        if k == "decoder3.main.bias":
            print("decoder3.main.bias")
            new_v = torch.zeros(32)
            new_v = v
            state_dict["decoder3.main.bias"] = new_v

        if k == "decoder4.main.weight":
            print("decoder4.main.weight")
            new_v = torch.zeros(16, list_num[10] + list_num[10] + 32, 3, 3, 3)  # v(64,128,3,3,3)

            last_index_list_0 = []
            for j in range(list_num[10]):
                location = index[10][j][1]
                last_index_list_0.append(location)

            for i in range(16):
                for j in range(32):
                    new_v[i][j] = v[i][j]

            for i in range(16):
                for m, n in enumerate(last_index_list_0):  # 16
                    new_v[i][m + 32] = v[i][n + 32]

            for i in range(16):
                for m, n in enumerate(last_index_list_0):  # 9
                    new_v[i][m + list_num[10] + 32] = v[i][n + 32 + 32]

            state_dict["decoder4.main.weight"] = new_v

        if k == "decoder4.main.bias":
            print("decoder4.main.bias")
            new_v = torch.zeros(16)
            new_v = v
            state_dict["decoder4.main.bias"] = new_v

        if k == "decoder5.main.weight":
            print("decoder5.main.weight")
            new_v = torch.zeros(16, 16 + list_num[0] + list_num[0], 3, 3, 3)  # v (32,96,3,3,3)

            last_index_list_0 = []
            for j in range(list_num[0]):
                location = index[0][j][1]
                last_index_list_0.append(location)

            for i in range(16):
                for j in range(16):
                    new_v[i][j] = v[i][j]

            for i in range(16):
                for m, n in enumerate(last_index_list_0):  # 16
                    new_v[i][m + 16] = v[i][n + 16]

            for i in range(16):
                for m, n in enumerate(last_index_list_0):  # 9
                    new_v[i][m + list_num[0] + 16] = v[i][n + 16 + 16]

            state_dict["decoder5.main.weight"] = new_v

        if k == "decoder5.main.bias":
            print("decoder5.main.bias")
            new_v = v
            state_dict["decoder5.main.bias"] = new_v

        if k == "fusion_m_scale4.main.weight":
            print("fusion_m_scale4.main.weight")
            new_v = torch.zeros(32, list_num[0] + list_num[1] + list_num[2] + list_num[3], 3, 3, 3)  # v (64,224,3,3,3)

            last_index_list_0 = []
            for j in range(list_num[0]):
                location = index[0][j][1]
                last_index_list_0.append(location)

            last_index_list_1 = []
            for j in range(list_num[1]):
                location = index[1][j][1]
                last_index_list_1.append(location)

            last_index_list_2 = []
            for j in range(list_num[2]):
                location = index[2][j][1]
                last_index_list_2.append(location)

            last_index_list_3 = []
            for j in range(list_num[3]):
                location = index[3][j][1]
                last_index_list_3.append(location)

            index_list = []
            for j in range(list_num[5]):
                location = index[5][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_0):  # 9
                    # print(f"i: {i}, m: {m}, j: {j}, n: {n}")
                    new_v[i][m] = v[j][n]

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_1):  # 12
                    new_v[i][m + list_num[0]] = v[j][n + 16]

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_2):  # 2
                    new_v[i][m + list_num[0] + list_num[1]] = v[j][n + 16 + 32]

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_3):  # 1
                    new_v[i][m + list_num[0] + list_num[1] + list_num[2]] = v[j][n + 16 + 32 + 32]

            state_dict["fusion_m_scale4.main.weight"] = new_v

        if k == "fusion_m_scale4.main.bias":
            print("fusion_m_scale4.main.bias")
            new_v = torch.zeros(32)
            new_v = v
            state_dict["fusion_m_scale4.main.bias"] = new_v

        if k == "fusion_m_scale3.main.weight":
            print("fusion_m_scale3.main.weight")
            new_v = torch.zeros(list_num[7], list_num[0] + list_num[1] + list_num[2] + list_num[3], 3, 3,
                                3)  # v (64,224,3,3,3)

            last_index_list_0 = []
            for j in range(list_num[0]):
                location = index[0][j][1]
                last_index_list_0.append(location)

            last_index_list_1 = []
            for j in range(list_num[1]):
                location = index[1][j][1]
                last_index_list_1.append(location)

            last_index_list_2 = []
            for j in range(list_num[2]):
                location = index[2][j][1]
                last_index_list_2.append(location)

            last_index_list_3 = []
            for j in range(list_num[3]):
                location = index[3][j][1]
                last_index_list_3.append(location)

            index_list = []
            for j in range(list_num[7]):
                location = index[7][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_0):  # 9
                    # print(f"i: {i}, m: {m}, j: {j}, n: {n}")
                    new_v[i][m] = v[j][n]

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_1):  # 12
                    new_v[i][m + list_num[0]] = v[j][n + 16]

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_2):  # 2
                    new_v[i][m + list_num[0] + list_num[1]] = v[j][n + 16 + 32]

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_3):  # 1
                    new_v[i][m + list_num[0] + list_num[1] + list_num[2]] = v[j][n + 16 + 32 + 32]

            state_dict["fusion_m_scale3.main.weight"] = new_v

        if k == "fusion_m_scale3.main.bias":
            print("fusion_m_scale3.main.bias")
            new_v = torch.zeros(list_num[7])
            index_list = []
            for j in range(list_num[7]):
                location = index[7][j][1]
                index_list.append(location)
                # print(location)
            for i, j in enumerate(index_list):
                new_v[i] = v[j]
            state_dict["fusion_m_scale3.main.bias"] = new_v

        if k == "fusion_m_scale2.main.weight":
            print("fusion_m_scale2.main.weight")
            new_v = torch.zeros(list_num[9], list_num[0] + list_num[1] + list_num[2] + list_num[3], 3, 3,
                                3)  # v (64,224,3,3,3)

            last_index_list_0 = []
            for j in range(list_num[0]):
                location = index[0][j][1]
                last_index_list_0.append(location)

            last_index_list_1 = []
            for j in range(list_num[1]):
                location = index[1][j][1]
                last_index_list_1.append(location)

            last_index_list_2 = []
            for j in range(list_num[2]):
                location = index[2][j][1]
                last_index_list_2.append(location)

            last_index_list_3 = []
            for j in range(list_num[3]):
                location = index[3][j][1]
                last_index_list_3.append(location)

            index_list = []
            for j in range(list_num[9]):
                location = index[9][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_0):  # 9
                    # print(f"i: {i}, m: {m}, j: {j}, n: {n}")
                    new_v[i][m] = v[j][n]

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_1):  # 12
                    new_v[i][m + list_num[0]] = v[j][n + 16]

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_2):  # 2
                    new_v[i][m + list_num[0] + list_num[1]] = v[j][n + 16 + 32]

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_3):  # 1
                    new_v[i][m + list_num[0] + list_num[1] + list_num[2]] = v[j][n + 16 + 32 + 32]

            state_dict["fusion_m_scale2.main.weight"] = new_v

        if k == "fusion_m_scale2.main.bias":
            print("fusion_m_scale2.main.bias")
            new_v = torch.zeros(list_num[9])
            index_list = []
            for j in range(list_num[9]):
                location = index[9][j][1]
                index_list.append(location)
                # print(location)
            for i, j in enumerate(index_list):
                new_v[i] = v[j]
            state_dict["fusion_m_scale2.main.bias"] = new_v

        if k == "fusion_m_scale4_2.main.weight":
            print("fusion_m_scale4_2.main.weight")
            new_v = torch.zeros(32, 32, 3, 3, 3)

            last_index_list = []
            for j in range(list_num[5]):
                location = index[5][j][1]
                last_index_list.append(location)

            index_list = []
            for j in range(list_num[6]):
                location = index[6][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list):
                    # print(f"i: {i}, m: {m}, j: {j}, n: {n}")

                    new_v[i][m] = v[j][n]
            state_dict["fusion_m_scale4_2.main.weight"] = new_v

        if k == "fusion_m_scale4_2.main.bias":
            print("fusion_m_scale4_2.main.bias")
            new_v = torch.zeros(32)

            index_list = []
            for j in range(list_num[6]):
                location = index[6][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                new_v[i] = v[j]
            state_dict["fusion_m_scale4_2.main.bias"] = new_v

        if k == "fusion_m_scale3_2.main.weight":
            print("fusion_m_scale3_2.main.weight")
            new_v = torch.zeros(list_num[8], list_num[7], 3, 3, 3)

            last_index_list = []
            for j in range(list_num[7]):
                location = index[7][j][1]
                last_index_list.append(location)

            index_list = []
            for j in range(list_num[8]):
                location = index[8][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list):
                    # print(f"i: {i}, m: {m}, j: {j}, n: {n}")

                    new_v[i][m] = v[j][n]
            state_dict["fusion_m_scale3_2.main.weight"] = new_v

        if k == "fusion_m_scale3_2.main.bias":
            print("fusion_m_scale3_2.main.bias")
            new_v = torch.zeros(list_num[8])

            index_list = []
            for j in range(list_num[8]):
                location = index[8][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                new_v[i] = v[j]
            state_dict["fusion_m_scale3_2.main.bias"] = new_v

        if k == "fusion_m_scale2_2.main.weight":
            print("fusion_m_scale2_2.main.weight")
            new_v = torch.zeros(list_num[10], list_num[9], 3, 3, 3)

            last_index_list = []
            for j in range(list_num[9]):
                location = index[9][j][1]
                last_index_list.append(location)

            index_list = []
            for j in range(list_num[10]):
                location = index[10][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list):
                    # print(f"i: {i}, m: {m}, j: {j}, n: {n}")

                    new_v[i][m] = v[j][n]
            state_dict["fusion_m_scale2_2.main.weight"] = new_v

        if k == "fusion_m_scale2_2.main.bias":
            print("fusion_m_scale2_2.main.bias")
            new_v = torch.zeros(list_num[10])

            index_list = []
            for j in range(list_num[10]):
                location = index[10][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                new_v[i] = v[j]
            state_dict["fusion_m_scale2_2.main.bias"] = new_v

        if k == "output_block_1.0.main.weight":
            new_v = v
            state_dict["output_block_1.0.main.weight"] = new_v

        if k == "output_block_1.0.main.bias":
            new_v = v
            state_dict["output_block_1.0.main.bias"] = new_v

        if k == "output_block_1.1.main.weight":
            new_v = v
            state_dict["output_block_1.1.main.weight"] = new_v

        if k == "output_block_1.1.main.bias":
            new_v = v
            state_dict["output_block_1.1.main.bias"] = new_v

        if k == "output_block_2.0.main.weight":
            new_v = v
            state_dict["output_block_2.0.main.weight"] = new_v

        if k == "output_block_2.0.main.bias":
            new_v = v
            state_dict["output_block_2.0.main.bias"] = new_v

        if k == "output_block_2.1.main.weight":
            new_v = v
            state_dict["output_block_2.1.main.weight"] = new_v

        if k == "output_block_2.1.main.bias":
            new_v = v
            state_dict["output_block_2.1.main.bias"] = new_v

        if k == "output_block_3.0.main.weight":
            new_v = v
            state_dict["output_block_3.0.main.weight"] = new_v

        if k == "output_block_3.0.main.bias":
            new_v = v
            state_dict["output_block_3.0.main.bias"] = new_v

        if k == "output_block_3.1.main.weight":
            new_v = v
            state_dict["output_block_3.1.main.weight"] = new_v

        if k == "output_block_3.1.main.bias":
            new_v = v
            state_dict["output_block_3.1.main.bias"] = new_v

        if k == "output_block.0.main.weight":
            new_v = v
            state_dict["output_block.0.main.weight"] = new_v

        if k == "output_block.0.main.bias":
            new_v = v
            state_dict["output_block.0.main.bias"] = new_v

        if k == "output_block.1.main.weight":
            new_v = v
            state_dict["output_block.1.main.weight"] = new_v

        if k == "output_block.1.main.bias":
            new_v = v
            state_dict["output_block.1.main.bias"] = new_v

        if k == "flow.weight":
            new_v = v
            state_dict["flow.weight"] = new_v

        if k == "flow.bias":
            new_v = v
            state_dict["flow.bias"] = new_v

    # # # 加载参数
    # for k,v in checkpoint.items():
    #     # print(k)
    #     # print(v.shape)

    #     # ---------------- ENCODER ----------------
    #     if  k == "encoder1_m.main.weight":
    #         print("encoder1_m.main.weight")
    #         new_v = torch.zeros(list_num[0],1,3,3,3)
    #         for j in range(list_num[0]):
    #             location = index[0][j][1]
    #             alpha_val = gate_alpha[0].flatten()[location].item() # 提取当前通道的 alpha
    #             new_v[j] = v[location] * alpha_val # 权重注入 alpha
    #         state_dict["encoder1_m.main.weight"] = new_v

    #     if  k == "encoder1_m.main.bias":
    #         print("encoder1_m.main.bias")
    #         new_v = torch.zeros(list_num[0])
    #         for j in range(list_num[0]):
    #             location = index[0][j][1]
    #             alpha_val = gate_alpha[0].flatten()[location].item()
    #             new_v[j] = v[location] * alpha_val # 偏置同样注入 alpha
    #         state_dict["encoder1_m.main.bias"] = new_v

    #     if  k == "encoder2_m.main.weight":
    #         print("encoder2_m.main.weight")
    #         new_v = torch.zeros(list_num[1],list_num[0],3,3,3)
    #         last_index_list = [index[0][j][1] for j in range(list_num[0])]
    #         for i in range(list_num[1]):
    #             location = index[1][i][1]
    #             alpha_val = gate_alpha[1].flatten()[location].item()
    #             for m, n in enumerate(last_index_list):
    #                 new_v[i][m] = v[location][n] * alpha_val
    #         state_dict["encoder2_m.main.weight"] = new_v

    #     if  k == "encoder2_m.main.bias":
    #         print("encoder2_m.main.bias")
    #         new_v = torch.zeros(list_num[1])
    #         for i in range(list_num[1]):
    #             location = index[1][i][1]
    #             alpha_val = gate_alpha[1].flatten()[location].item()
    #             new_v[i] = v[location] * alpha_val
    #         state_dict["encoder2_m.main.bias"] = new_v

    #     if  k == "encoder3_m.main.weight":
    #         print("encoder3_m.main.weight")
    #         new_v = torch.zeros(list_num[2],list_num[1],3,3,3)
    #         last_index_list = [index[1][j][1] for j in range(list_num[1])]
    #         for i in range(list_num[2]):
    #             location = index[2][i][1]
    #             alpha_val = gate_alpha[2].flatten()[location].item()
    #             for m, n in enumerate(last_index_list):
    #                 new_v[i][m] = v[location][n] * alpha_val
    #         state_dict["encoder3_m.main.weight"] = new_v

    #     if  k == "encoder3_m.main.bias":
    #         print("encoder3_m.main.bias")
    #         new_v = torch.zeros(list_num[2])
    #         for i in range(list_num[2]):
    #             location = index[2][i][1]
    #             alpha_val = gate_alpha[2].flatten()[location].item()
    #             new_v[i] = v[location] * alpha_val
    #         state_dict["encoder3_m.main.bias"] = new_v

    #     if  k == "encoder4_m.main.weight":
    #         print("encoder4_m.main.weight")
    #         new_v = torch.zeros(list_num[3],list_num[2],3,3,3)
    #         last_index_list = [index[2][j][1] for j in range(list_num[2])]
    #         for i in range(list_num[3]):
    #             location = index[3][i][1]
    #             alpha_val = gate_alpha[3].flatten()[location].item()
    #             for m, n in enumerate(last_index_list):
    #                 new_v[i][m] = v[location][n] * alpha_val
    #         state_dict["encoder4_m.main.weight"] = new_v

    #     if  k == "encoder4_m.main.bias":
    #         print("encoder4_m.main.bias")
    #         new_v = torch.zeros(list_num[3])
    #         for i in range(list_num[3]):
    #             location = index[3][i][1]
    #             alpha_val = gate_alpha[3].flatten()[location].item()
    #             new_v[i] = v[location] * alpha_val
    #         state_dict["encoder4_m.main.bias"] = new_v

    #     if  k == "encoder5_m.main.weight":
    #         print("encoder5_m.main.weight")
    #         new_v = torch.zeros(list_num[4],list_num[3],3,3,3)
    #         last_index_list = [index[3][j][1] for j in range(list_num[3])]
    #         for i in range(list_num[4]):
    #             location = index[4][i][1]
    #             alpha_val = gate_alpha[4].flatten()[location].item()
    #             for m, n in enumerate(last_index_list):
    #                 new_v[i][m] = v[location][n] * alpha_val
    #         state_dict["encoder5_m.main.weight"] = new_v

    #     if  k == "encoder5_m.main.bias":
    #         print("encoder5_m.main.bias")
    #         new_v = torch.zeros(list_num[4])
    #         for i in range(list_num[4]):
    #             location = index[4][i][1]
    #             alpha_val = gate_alpha[4].flatten()[location].item()
    #             new_v[i] = v[location] * alpha_val
    #         state_dict["encoder5_m.main.bias"] = new_v

    #     if  k == "decoder1.main.weight":
    #         print("decoder1.main.weight")
    #         new_v = torch.zeros(32,64,3,3,3) # v(64,128,3,3,3)
    #         new_v = v
    #         state_dict["decoder1.main.weight"] = new_v

    #     if  k == "decoder1.main.bias":
    #         print("decoder1.main.bias")
    #         new_v = torch.zeros(32)
    #         new_v = v
    #         state_dict["decoder1.main.bias"] = new_v

    #     if  k == "decoder2.main.weight":
    #         print("decoder2.main.weight")
    #         new_v = torch.zeros(32,96,3,3,3) # v(64,128,3,3,3)
    #         new_v = v
    #         state_dict["decoder2.main.weight"] = new_v

    #     if  k == "decoder2.main.bias":
    #         print("decoder2.main.bias")
    #         new_v = torch.zeros(32)
    #         new_v = v
    #         state_dict["decoder2.main.bias"] = new_v

    #     if  k == "decoder3.main.weight":
    #         print("decoder3.main.weight")
    #         new_v = torch.zeros(32,32+list_num[8]+list_num[8],3,3,3) # v(64,128,3,3,3)

    #         last_index_list_0 = []
    #         for j in range(list_num[8]):
    #             location = index[8][j][1]
    #             last_index_list_0.append(location)

    #         for i in range(32):
    #             for j in range(32):
    #                 new_v[i][j] = v[i][j]

    #         for i in range(32):
    #             for m,n in enumerate(last_index_list_0): # 16
    #                 new_v[i][m+32] = v[i][n+32]

    #         for i in range(32):
    #             for m,n in enumerate(last_index_list_0): # 9
    #                 new_v[i][m+list_num[8]+32] = v[i][n+32+32]

    #         state_dict["decoder3.main.weight"] = new_v

    #     if  k == "decoder3.main.bias":
    #         print("decoder3.main.bias")
    #         new_v = torch.zeros(32)
    #         new_v = v
    #         state_dict["decoder3.main.bias"] = new_v

    #     if  k == "decoder4.main.weight":
    #         print("decoder4.main.weight")
    #         new_v = torch.zeros(16,list_num[10]+list_num[10]+32,3,3,3) # v(64,128,3,3,3)

    #         last_index_list_0 = []
    #         for j in range(list_num[10]):
    #             location = index[10][j][1]
    #             last_index_list_0.append(location)

    #         for i in range(16):
    #             for j in range(32):
    #                 new_v[i][j] = v[i][j]

    #         for i in range(16):
    #             for m,n in enumerate(last_index_list_0): # 16
    #                 new_v[i][m+32] = v[i][n+32]

    #         for i in range(16):
    #             for m,n in enumerate(last_index_list_0): # 9
    #                 new_v[i][m+list_num[10]+32] = v[i][n+32+32]

    #         state_dict["decoder4.main.weight"] = new_v

    #     if  k == "decoder4.main.bias":
    #         print("decoder4.main.bias")
    #         new_v = torch.zeros(16)
    #         new_v = v
    #         state_dict["decoder4.main.bias"] = new_v

    #     if  k == "decoder5.main.weight":
    #         print("decoder5.main.weight")
    #         new_v = torch.zeros(16,16+list_num[0]+list_num[0],3,3,3)  # v (32,96,3,3,3)

    #         last_index_list_0 = []
    #         for j in range(list_num[0]):
    #             location = index[0][j][1]
    #             last_index_list_0.append(location)

    #         for i in range(16):
    #             for j in range(16):
    #                 new_v[i][j] = v[i][j]

    #         for i in range(16):
    #             for m,n in enumerate(last_index_list_0): # 16
    #                 new_v[i][m+16] = v[i][n+16]

    #         for i in range(16):
    #             for m,n in enumerate(last_index_list_0): # 9
    #                 new_v[i][m+list_num[0]+16] = v[i][n+16+16]

    #         state_dict["decoder5.main.weight"] = new_v

    #     if  k == "decoder5.main.bias":
    #         print("decoder5.main.bias")
    #         new_v = v
    #         state_dict["decoder5.main.bias"] = new_v

    #     if  k == "fusion_m_scale4.main.weight":
    #         print("fusion_m_scale4.main.weight")
    #         new_v = torch.zeros(32,list_num[0]+list_num[1]+list_num[2]+list_num[3],3,3,3)  # v (64,224,3,3,3)

    #         last_index_list_0 = []
    #         for j in range(list_num[0]):
    #             location = index[0][j][1]
    #             last_index_list_0.append(location)

    #         last_index_list_1 = []
    #         for j in range(list_num[1]):
    #             location = index[1][j][1]
    #             last_index_list_1.append(location)

    #         last_index_list_2 = []
    #         for j in range(list_num[2]):
    #             location = index[2][j][1]
    #             last_index_list_2.append(location)

    #         last_index_list_3 = []
    #         for j in range(list_num[3]):
    #             location = index[3][j][1]
    #             last_index_list_3.append(location)

    #         index_list = []
    #         for j in range(list_num[5]):
    #             location = index[5][j][1]
    #             index_list.append(location)

    #         for i,j in enumerate(index_list):
    #             alpha_val = gate_alpha[5].flatten()[j].item() # 提取当前通道的 alpha
    #             for m,n in enumerate(last_index_list_0): # 9
    #                 # print(f"i: {i}, m: {m}, j: {j}, n: {n}")
    #                 new_v[i][m] = v[j][n] * alpha_val

    #         for i,j in enumerate(index_list):
    #             alpha_val = gate_alpha[5].flatten()[j].item()
    #             for m,n in enumerate(last_index_list_1): # 12
    #                 new_v[i][m+list_num[0]] = v[j][n+16] * alpha_val

    #         for i,j in enumerate(index_list):
    #             alpha_val = gate_alpha[5].flatten()[j].item()
    #             for m,n in enumerate(last_index_list_2): # 2
    #                 new_v[i][m+list_num[0]+list_num[1]] = v[j][n+16+32] * alpha_val

    #         for i,j in enumerate(index_list):
    #             alpha_val = gate_alpha[5].flatten()[j].item()
    #             for m,n in enumerate(last_index_list_3): # 1
    #                 new_v[i][m+list_num[0]+list_num[1]+list_num[2]] = v[j][n+16+32+32] * alpha_val

    #         state_dict["fusion_m_scale4.main.weight"] = new_v

    #     if  k == "fusion_m_scale4.main.bias":
    #         print("fusion_m_scale4.main.bias")
    #         new_v = torch.zeros(32)
    #         index_list = []
    #         for j in range(list_num[5]):
    #             location = index[5][j][1]
    #             index_list.append(location)

    #         for i,j in enumerate(index_list):
    #             alpha_val = gate_alpha[5].flatten()[j].item() # 提取当前通道的 alpha
    #             new_v[i] = v[j] * alpha_val
    #         state_dict["fusion_m_scale4.main.bias"] = new_v

    #     if  k == "fusion_m_scale3.main.weight":
    #         print("fusion_m_scale3.main.weight")
    #         new_v = torch.zeros(list_num[7],list_num[0]+list_num[1]+list_num[2]+list_num[3],3,3,3)  # v (64,224,3,3,3)

    #         last_index_list_0 = []
    #         for j in range(list_num[0]):
    #             location = index[0][j][1]
    #             last_index_list_0.append(location)

    #         last_index_list_1 = []
    #         for j in range(list_num[1]):
    #             location = index[1][j][1]
    #             last_index_list_1.append(location)

    #         last_index_list_2 = []
    #         for j in range(list_num[2]):
    #             location = index[2][j][1]
    #             last_index_list_2.append(location)

    #         last_index_list_3 = []
    #         for j in range(list_num[3]):
    #             location = index[3][j][1]
    #             last_index_list_3.append(location)

    #         index_list = []
    #         for j in range(list_num[7]):
    #             location = index[7][j][1]
    #             index_list.append(location)

    #         for i,j in enumerate(index_list):
    #             alpha_val = gate_alpha[7].flatten()[j].item() # 提取当前通道的 alpha
    #             for m,n in enumerate(last_index_list_0): # 9
    #                 new_v[i][m] = v[j][n] * alpha_val

    #         for i,j in enumerate(index_list):
    #             alpha_val = gate_alpha[7].flatten()[j].item()
    #             for m,n in enumerate(last_index_list_1): # 12
    #                 new_v[i][m+list_num[0]] = v[j][n+16] * alpha_val

    #         for i,j in enumerate(index_list):
    #             alpha_val = gate_alpha[7].flatten()[j].item()
    #             for m,n in enumerate(last_index_list_2): # 2
    #                 new_v[i][m+list_num[0]+list_num[1]] = v[j][n+16+32] * alpha_val

    #         for i,j in enumerate(index_list):
    #             alpha_val = gate_alpha[7].flatten()[j].item()
    #             for m,n in enumerate(last_index_list_3): # 1
    #                 new_v[i][m+list_num[0]+list_num[1]+list_num[2]] = v[j][n+16+32+32] * alpha_val

    #         state_dict["fusion_m_scale3.main.weight"] = new_v

    #     if  k == "fusion_m_scale3.main.bias":
    #         print("fusion_m_scale3.main.bias")
    #         new_v = torch.zeros(list_num[7])
    #         index_list = []
    #         for j in range(list_num[7]):
    #             location = index[7][j][1]
    #             index_list.append(location)
    #             # print(location)
    #         for i,j in enumerate(index_list):
    #             alpha_val = gate_alpha[7].flatten()[j].item()
    #             new_v[i] = v[j] * alpha_val
    #         state_dict["fusion_m_scale3.main.bias"] = new_v

    #     if  k == "fusion_m_scale2.main.weight":
    #         print("fusion_m_scale2.main.weight")
    #         new_v = torch.zeros(list_num[9],list_num[0]+list_num[1]+list_num[2]+list_num[3],3,3,3)  # v (64,224,3,3,3)

    #         last_index_list_0 = []
    #         for j in range(list_num[0]):
    #             location = index[0][j][1]
    #             last_index_list_0.append(location)

    #         last_index_list_1 = []
    #         for j in range(list_num[1]):
    #             location = index[1][j][1]
    #             last_index_list_1.append(location)

    #         last_index_list_2 = []
    #         for j in range(list_num[2]):
    #             location = index[2][j][1]
    #             last_index_list_2.append(location)

    #         last_index_list_3 = []
    #         for j in range(list_num[3]):
    #             location = index[3][j][1]
    #             last_index_list_3.append(location)

    #         index_list = []
    #         for j in range(list_num[9]):
    #             location = index[9][j][1]
    #             index_list.append(location)

    #         for i,j in enumerate(index_list):
    #             alpha_val = gate_alpha[9].flatten()[j].item()
    #             for m,n in enumerate(last_index_list_0): # 9
    #                 new_v[i][m] = v[j][n] * alpha_val

    #         for i,j in enumerate(index_list):
    #             alpha_val = gate_alpha[9].flatten()[j].item()
    #             for m,n in enumerate(last_index_list_1): # 12
    #                 new_v[i][m+list_num[0]] = v[j][n+16] * alpha_val

    #         for i,j in enumerate(index_list):
    #             alpha_val = gate_alpha[9].flatten()[j].item()
    #             for m,n in enumerate(last_index_list_2): # 2
    #                 new_v[i][m+list_num[0]+list_num[1]] = v[j][n+16+32] * alpha_val

    #         for i,j in enumerate(index_list):
    #             alpha_val = gate_alpha[9].flatten()[j].item()
    #             for m,n in enumerate(last_index_list_3): # 1
    #                 new_v[i][m+list_num[0]+list_num[1]+list_num[2]] = v[j][n+16+32+32] * alpha_val

    #         state_dict["fusion_m_scale2.main.weight"] = new_v

    #     if  k == "fusion_m_scale2.main.bias":
    #         print("fusion_m_scale2.main.bias")
    #         new_v = torch.zeros(list_num[9])
    #         index_list = []
    #         for j in range(list_num[9]):
    #             location = index[9][j][1]
    #             index_list.append(location)
    #             # print(location)
    #         for i,j in enumerate(index_list):
    #             alpha_val = gate_alpha[9].flatten()[j].item()
    #             new_v[i] = v[j] * alpha_val
    #         state_dict["fusion_m_scale2.main.bias"] = new_v

    #     if  k == "fusion_m_scale4_2.main.weight":
    #         print("fusion_m_scale4_2.main.weight")
    #         new_v = torch.zeros(32,32,3,3,3)

    #         last_index_list = []
    #         for j in range(list_num[5]):
    #             location = index[5][j][1]
    #             last_index_list.append(location)

    #         index_list = []
    #         for j in range(list_num[6]):
    #             location = index[6][j][1]
    #             index_list.append(location)

    #         for i,j in enumerate(index_list):
    #             alpha_val = gate_alpha[6].flatten()[j].item()
    #             for m,n in enumerate(last_index_list):
    #                 new_v[i][m] = v[j][n] * alpha_val
    #         state_dict["fusion_m_scale4_2.main.weight"] = new_v

    #     if  k == "fusion_m_scale4_2.main.bias":
    #         print("fusion_m_scale4_2.main.bias")
    #         new_v = torch.zeros(32)

    #         index_list = []
    #         for j in range(list_num[6]):
    #             location = index[6][j][1]
    #             index_list.append(location)

    #         for i,j in enumerate(index_list):
    #             alpha_val = gate_alpha[6].flatten()[j].item()
    #             new_v[i]= v[j] * alpha_val
    #         state_dict["fusion_m_scale4_2.main.bias"] = new_v

    #     if  k == "fusion_m_scale3_2.main.weight":
    #         print("fusion_m_scale3_2.main.weight")
    #         new_v = torch.zeros(list_num[8],list_num[7],3,3,3)

    #         last_index_list = []
    #         for j in range(list_num[7]):
    #             location = index[7][j][1]
    #             last_index_list.append(location)

    #         index_list = []
    #         for j in range(list_num[8]):
    #             location = index[8][j][1]
    #             index_list.append(location)

    #         for i,j in enumerate(index_list):
    #             alpha_val = gate_alpha[8].flatten()[j].item()
    #             for m,n in enumerate(last_index_list):
    #                 new_v[i][m] = v[j][n] * alpha_val
    #         state_dict["fusion_m_scale3_2.main.weight"] = new_v

    #     if  k == "fusion_m_scale3_2.main.bias":
    #         print("fusion_m_scale3_2.main.bias")
    #         new_v = torch.zeros(list_num[8])

    #         index_list = []
    #         for j in range(list_num[8]):
    #             location = index[8][j][1]
    #             index_list.append(location)

    #         for i,j in enumerate(index_list):
    #             alpha_val = gate_alpha[8].flatten()[j].item()
    #             new_v[i]= v[j] * alpha_val
    #         state_dict["fusion_m_scale3_2.main.bias"] = new_v

    #     if  k == "fusion_m_scale2_2.main.weight":
    #         print("fusion_m_scale2_2.main.weight")
    #         new_v = torch.zeros(list_num[10],list_num[9],3,3,3)

    #         last_index_list = []
    #         for j in range(list_num[9]):
    #             location = index[9][j][1]
    #             last_index_list.append(location)

    #         index_list = []
    #         for j in range(list_num[10]):
    #             location = index[10][j][1]
    #             index_list.append(location)

    #         for i,j in enumerate(index_list):
    #             alpha_val = gate_alpha[10].flatten()[j].item()
    #             for m,n in enumerate(last_index_list):
    #                 new_v[i][m] = v[j][n] * alpha_val
    #         state_dict["fusion_m_scale2_2.main.weight"] = new_v

    #     if  k == "fusion_m_scale2_2.main.bias":
    #         print("fusion_m_scale2_2.main.bias")
    #         new_v = torch.zeros(list_num[10])

    #         index_list = []
    #         for j in range(list_num[10]):
    #             location = index[10][j][1]
    #             index_list.append(location)

    #         for i,j in enumerate(index_list):
    #             alpha_val = gate_alpha[10].flatten()[j].item()
    #             new_v[i]= v[j] * alpha_val
    #         state_dict["fusion_m_scale2_2.main.bias"] = new_v

    #     if  k == "output_block_1.0.main.weight":
    #         new_v= v
    #         state_dict["output_block_1.0.main.weight"] = new_v

    #     if  k == "output_block_1.0.main.bias":
    #         new_v = v
    #         state_dict["output_block_1.0.main.bias"] = new_v

    #     if  k == "output_block_1.1.main.weight":
    #         new_v = v
    #         state_dict["output_block_1.1.main.weight"] = new_v

    #     if  k == "output_block_1.1.main.bias":
    #         new_v = v
    #         state_dict["output_block_1.1.main.bias"] = new_v

    #     if  k == "output_block_2.0.main.weight":
    #         new_v = v
    #         state_dict["output_block_2.0.main.weight"] = new_v

    #     if  k == "output_block_2.0.main.bias":
    #         new_v = v
    #         state_dict["output_block_2.0.main.bias"] = new_v

    #     if  k == "output_block_2.1.main.weight":
    #         new_v = v
    #         state_dict["output_block_2.1.main.weight"] = new_v

    #     if  k == "output_block_2.1.main.bias":
    #         new_v = v
    #         state_dict["output_block_2.1.main.bias"] = new_v

    #     if  k == "output_block_3.0.main.weight":
    #         new_v = v
    #         state_dict["output_block_3.0.main.weight"] = new_v

    #     if  k == "output_block_3.0.main.bias":
    #         new_v = v
    #         state_dict["output_block_3.0.main.bias"] = new_v

    #     if  k == "output_block_3.1.main.weight":
    #         new_v = v
    #         state_dict["output_block_3.1.main.weight"] = new_v

    #     if  k == "output_block_3.1.main.bias":
    #         new_v = v
    #         state_dict["output_block_3.1.main.bias"] = new_v

    #     if  k == "output_block.0.main.weight":
    #         new_v = v
    #         state_dict["output_block.0.main.weight"] = new_v

    #     if  k == "output_block.0.main.bias":
    #         new_v = v
    #         state_dict["output_block.0.main.bias"] = new_v

    #     if  k == "output_block.1.main.weight":
    #         new_v = v
    #         state_dict["output_block.1.main.weight"] = new_v

    #     if  k == "output_block.1.main.bias":
    #         new_v = v
    #         state_dict["output_block.1.main.bias"] = new_v

    #     if  k == "flow.weight":
    #         new_v = v
    #         state_dict["flow.weight"] = new_v

    #     if  k == "flow.bias":
    #         new_v = v
    #         state_dict["flow.bias"] = new_v

    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

    # # 统计参数量
    dummy_input = torch.randn(1, 1, 192, 160, 192).to(device)
    flops, params = profile(model, (dummy_input, dummy_input,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))

    # # 对验证集进行测试
    Dice = 0
    Dice_1 = 0
    Dice_2 = 0
    Dice_3 = 0
    mean_dice = 0
    num = 0
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = torch.zeros(90)
    model.eval()
    print("---------验证集的Dice----------")
    SPT = vxm.torch.layers.SpatialTransformer(inshape, mode="nearest").to(device)
    for step in range(len(val_t1_files)):
        # 提取t1数据
        inputs_fixed, y_true = next(val_t1_generator)
        inputs_fixed = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs_fixed]
        # 提取对应t1的seg数据
        seg_inputs_fixed, seg_y_true = next(val_seg_generator)
        seg_inputs_fixed = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in seg_inputs_fixed]
        for step in range(len(val_t1_files)):
            # 提取t1数据
            inputs_moving, y_true = next(val_t1_generator)
            inputs_moving = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs_moving]
            # 提取对应t1的seg数据
            seg_inputs_moving, seg_y_true = next(val_seg_generator)
            seg_inputs_moving = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in
                                 seg_inputs_moving]
            if step < len(val_t1_files) - 1:
                with torch.no_grad():
                    # 生成flow
                    starter.record()

                    flow_1, flow_2, flow_3, flow, delta_flow_2, delta_flow_3, _ = model(inputs_moving[0],
                                                                                        inputs_fixed[0])

                    # flow_1,flow_2,flow_3,flow,delta_flow_2,delta_flow_3,_,_,_ = model_pretrain(inputs_moving[0],inputs_fixed[0])

                    ender.record()
                    # 同步GPU时间
                    torch.cuda.synchronize()
                    curr_time = starter.elapsed_time(ender)  # 计算时间
                    times[num] = curr_time
                    # num = num + 1
                    print(curr_time)

                    # x_in = torch.cat((inputs_moving[0],inputs_fixed[0]), dim=1)
                    # flow_1,flow_2,flow_3,flow,delta_flow_2,delta_flow_3,_ = model(inputs_moving[0],inputs_fixed[0])

                    # 上采样
                    flow_1_up = nn.functional.interpolate(flow_1, scale_factor=8, mode="trilinear") * 8
                    flow_2_up = nn.functional.interpolate(flow_2, scale_factor=4, mode="trilinear") * 4
                    flow_3_up = nn.functional.interpolate(flow_3, scale_factor=2, mode="trilinear") * 2

                    seg_pre_1 = SPT(seg_inputs_moving[0], flow_1_up)
                    seg_pre_2 = SPT(seg_inputs_moving[0], flow_2_up)
                    seg_pre_3 = SPT(seg_inputs_moving[0], flow_3_up)
                    dice_1, _ = vxm.torch.losses.compute_dice(seg_pre_1, seg_inputs_fixed[0])
                    dice_2, _ = vxm.torch.losses.compute_dice(seg_pre_2, seg_inputs_fixed[0])
                    dice_3, _ = vxm.torch.losses.compute_dice(seg_pre_3, seg_inputs_fixed[0])

                    # 得到变形场2
                    seg_pre = SPT(seg_inputs_moving[0], flow)
                    dice, dice_region = vxm.torch.losses.compute_dice(seg_pre, seg_inputs_fixed[0])
                    if dice < 1:
                        print("dice", dice)
                        Dice += dice
                        num += 1
        print("----------换fixed image---------")

    metric = Dice / num
    print("mean_dice", metric)
    mean_time = times.mean().item()
    print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000 / mean_time))

    torch.save(model.state_dict(),
               "/home/boys/project/voxelmorph/voxelmorph_code/models/comparison_methods/abdomenCT/daul_pyramid_PFNet_FFM_normal_GDP_adaptive_nopretrain_diff.pth")


# 将大模型的参数赋值给小模型并保存 normal_diff
def test_dual_pyramid_vxm_FFM_adaptive_normal_trans(config=config):
    print("Training dual_pyramid_vxm_plus")

    # 读取huge model
    model_pretrain = vxm.networks.dual_pyramid_VxmDense_Trans_FFM_normal_GDP()

    checkpoint = torch.load(
        "./daul_pyramid_vxm_FFM_4layer_4_trans_normal_GDP_Lambda0.0036_theta4_decay0.5_decayrate0.98_nopretrain_abdomenct_839.pth")

    model_pretrain.to(device)
    model_dict = model_pretrain.state_dict()
    state_dict = {k: v for k, v in checkpoint.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model_pretrain.load_state_dict(model_dict)

    # model_pretrain.train()
    decay = 0.5 * (0.98 ** 839)
    # decay = 1
    model_pretrain.set_decay(decay)

    inputs, y_true = next(seg_generator)
    inputs = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs]
    y_true = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in y_true]

    # 生成flow
    with torch.no_grad():
        # 得到非零索引
        flow_1, flow_2, flow_3, flow_final, delta_flow_2, delta_flow_3, _, ori_alpha, gate_alpha = model_pretrain(
            inputs[0], inputs[1])

        # print("gate_alpha",gate_alpha)
    # print("ori_aplha",ori_alpha)
    # print(ori_alpha)

    index = []
    for tensor in gate_alpha:
        indices = torch.nonzero(tensor).squeeze()  # 获取非零元素的索引
        index.append(indices)
    print(index[0])
    print(index[1])
    print(index[2])
    print(index[3])
    print(len(index[4]))
    print(len(index[5]))
    print(len(index[6]))
    print(len(index[7]))
    print(len(index[8]))
    print(len(index[9]))
    print(len(index[10]))

    list_num = [len(index[0]), len(index[1]), len(index[2]), len(index[3]), len(index[4]), len(index[5]), len(index[6]),
                len(index[7]), len(index[8]), len(index[9]), len(index[10])]
    new_alpha = []
    for i in range(len(list_num)):
        ori_alpha_i = ori_alpha[i]
        new_alpha_i = torch.ones(1, list_num[i])
        for j in range(list_num[i]):
            # if i == 2:
            #     index[2] = torch.unsqueeze(index[2], 0)
            #     print(index[2])
            # if i == 3:
            #     index[3] = torch.unsqueeze(index[3], 0)
            #     print(index[3])
            # print(i)
            # print(j)
            location = index[i][j][1]
            new_alpha_i[0][j] = ori_alpha_i[0][location]
        new_alpha.append(new_alpha_i)

    # # 初始化一个轻量的小模型
    # model = vxm.networks.dual_pyramid_VxmDense_FFM_4layer_4_huge_adaptive_val_2(alpha1=new_alpha[0], alpha2=new_alpha[1],
    #         alpha3=new_alpha[2], alpha4=new_alpha[3],alpha5=new_alpha[4], alpha6_1=new_alpha[5],alpha6_2=new_alpha[6],
    #         alpha7_1=new_alpha[7], alpha7_2=new_alpha[8],alpha8_1=new_alpha[9],alpha8_2=new_alpha[10],list_num=list_num)

    # model.set_decay(decay)

    model = vxm.networks.dual_pyramid_VxmDense_Trans_FFM_normal_adaptive_val(list_num=list_num)

    model.to(device)
    model_dict = model.state_dict()

    # # 验证回传是否正确
    # with torch.no_grad():
    #     # 得到非零索引
    #     flow_1,flow_2,flow_3,flow_final,delta_flow_2,delta_flow_3,_,ori_alpha,gate_alpha_new = model(inputs[0],inputs[1])
    #     print(gate_alpha_new)

    # index = index1

    state_dict = {}
    # # # 加载参数
    for k, v in checkpoint.items():
        # print(k)
        # print(v.shape)

        if k == "encoder1_m.main.weight":
            print("encoder1_m.main.weight")
            new_v = torch.zeros(list_num[0], 1, 3, 3, 3)
            index_list = []
            for j in range(list_num[0]):
                location = index[0][j][1]
                index_list.append(location)
                # print(location)
            for i, j in enumerate(index_list):
                new_v[i] = v[j]
            state_dict["encoder1_m.main.weight"] = new_v

        if k == "encoder1_m.main.bias":
            print("encoder1_m.main.bias")

            new_v = torch.zeros(list_num[0])
            index_list = []
            for j in range(list_num[0]):
                location = index[0][j][1]
                index_list.append(location)
                # print(location)
            for i, j in enumerate(index_list):
                new_v[i] = v[j]
            state_dict["encoder1_m.main.bias"] = new_v

        if k == "encoder2_m.main.weight":
            print("encoder2_m.main.weight")
            new_v = torch.zeros(list_num[1], list_num[0], 3, 3, 3)

            last_index_list = []
            for j in range(list_num[0]):
                location = index[0][j][1]
                last_index_list.append(location)

            index_list = []
            for j in range(list_num[1]):
                location = index[1][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list):
                    # print(f"i: {i}, m: {m}, j: {j}, n: {n}")

                    new_v[i][m] = v[j][n]
            state_dict["encoder2_m.main.weight"] = new_v

        if k == "encoder2_m.main.bias":
            print("encoder2_m.main.bias")
            # print(v.shape)
            new_v = torch.zeros(list_num[1])

            index_list = []
            for j in range(list_num[1]):
                location = index[1][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                new_v[i] = v[j]
            state_dict["encoder2_m.main.bias"] = new_v

        if k == "encoder3_m.main.weight":
            print("encoder3_m.main.weight")
            new_v = torch.zeros(list_num[2], list_num[1], 3, 3, 3)

            last_index_list = []
            for j in range(list_num[1]):
                location = index[1][j][1]
                last_index_list.append(location)

            index_list = []
            for j in range(list_num[2]):
                location = index[2][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list):
                    # print(f"i: {i}, m: {m}, j: {j}, n: {n}")

                    new_v[i][m] = v[j][n]
            state_dict["encoder3_m.main.weight"] = new_v

        if k == "encoder3_m.main.bias":
            print("encoder3_m.main.bias")
            new_v = torch.zeros(list_num[2])

            index_list = []
            for j in range(list_num[2]):
                location = index[2][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                new_v[i] = v[j]
            state_dict["encoder3_m.main.bias"] = new_v

        if k == "encoder4_m.main.weight":
            print("encoder4_m.main.weight")
            new_v = torch.zeros(list_num[3], list_num[2], 3, 3, 3)

            last_index_list = []
            for j in range(list_num[2]):
                location = index[2][j][1]
                last_index_list.append(location)

            index_list = []
            for j in range(list_num[3]):
                location = index[3][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list):
                    # print(f"i: {i}, m: {m}, j: {j}, n: {n}")

                    new_v[i][m] = v[j][n]
            state_dict["encoder4_m.main.weight"] = new_v

        if k == "encoder4_m.main.bias":
            print("encoder4_m.main.bias")
            new_v = torch.zeros(list_num[3])

            index_list = []
            for j in range(list_num[3]):
                location = index[3][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                new_v[i] = v[j]
            state_dict["encoder4_m.main.bias"] = new_v

        if k == "encoder5_m.main.weight":
            print("encoder5_m.main.weight")
            new_v = torch.zeros(list_num[4], list_num[3], 3, 3, 3)

            last_index_list = []
            for j in range(list_num[3]):
                location = index[3][j][1]
                last_index_list.append(location)

            index_list = []
            for j in range(list_num[4]):
                location = index[4][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list):
                    # print(f"i: {i}, m: {m}, j: {j}, n: {n}")

                    new_v[i][m] = v[j][n]
            state_dict["encoder5_m.main.weight"] = new_v

        if k == "encoder5_m.main.bias":
            print("encoder5_m.main.bias")
            new_v = torch.zeros(list_num[4])

            index_list = []
            for j in range(list_num[4]):
                location = index[4][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                new_v[i] = v[j]
            state_dict["encoder5_m.main.bias"] = new_v

        if k == "decoder1.main.weight":
            print("decoder1.main.weight")
            new_v = torch.zeros(32, 64, 3, 3, 3)  # v(64,128,3,3,3)
            new_v = v
            state_dict["decoder1.main.weight"] = new_v

        if k == "decoder1.main.bias":
            print("decoder1.main.bias")
            new_v = torch.zeros(32)
            new_v = v
            state_dict["decoder1.main.bias"] = new_v

        if k == "decoder2.main.weight":
            print("decoder2.main.weight")
            new_v = torch.zeros(32, 96, 3, 3, 3)  # v(64,128,3,3,3)
            new_v = v
            state_dict["decoder2.main.weight"] = new_v

        if k == "decoder2.main.bias":
            print("decoder2.main.bias")
            new_v = torch.zeros(32)
            new_v = v
            state_dict["decoder2.main.bias"] = new_v

        if k == "decoder3.main.weight":
            print("decoder3.main.weight")
            new_v = torch.zeros(32, 32 + list_num[8] + list_num[8], 3, 3, 3)  # v(64,128,3,3,3)

            last_index_list_0 = []
            for j in range(list_num[8]):
                location = index[8][j][1]
                last_index_list_0.append(location)

            for i in range(32):
                for j in range(32):
                    new_v[i][j] = v[i][j]

            for i in range(32):
                for m, n in enumerate(last_index_list_0):  # 16
                    new_v[i][m + 32] = v[i][n + 32]

            for i in range(32):
                for m, n in enumerate(last_index_list_0):  # 9
                    new_v[i][m + list_num[8] + 32] = v[i][n + 32 + 32]

            state_dict["decoder3.main.weight"] = new_v

        if k == "decoder3.main.bias":
            print("decoder3.main.bias")
            new_v = torch.zeros(32)
            new_v = v
            state_dict["decoder3.main.bias"] = new_v

        if k == "decoder4.main.weight":
            print("decoder4.main.weight")
            new_v = torch.zeros(16, list_num[10] + list_num[10] + 32, 3, 3, 3)  # v(64,128,3,3,3)

            last_index_list_0 = []
            for j in range(list_num[10]):
                location = index[10][j][1]
                last_index_list_0.append(location)

            for i in range(16):
                for j in range(32):
                    new_v[i][j] = v[i][j]

            for i in range(16):
                for m, n in enumerate(last_index_list_0):  # 16
                    new_v[i][m + 32] = v[i][n + 32]

            for i in range(16):
                for m, n in enumerate(last_index_list_0):  # 9
                    new_v[i][m + list_num[10] + 32] = v[i][n + 32 + 32]

            state_dict["decoder4.main.weight"] = new_v

        if k == "decoder4.main.bias":
            print("decoder4.main.bias")
            new_v = torch.zeros(16)
            new_v = v
            state_dict["decoder4.main.bias"] = new_v

        if k == "decoder5.main.weight":
            print("decoder5.main.weight")
            new_v = torch.zeros(16, 16 + list_num[0] + list_num[0], 3, 3, 3)  # v (32,96,3,3,3)

            last_index_list_0 = []
            for j in range(list_num[0]):
                location = index[0][j][1]
                last_index_list_0.append(location)

            for i in range(16):
                for j in range(16):
                    new_v[i][j] = v[i][j]

            for i in range(16):
                for m, n in enumerate(last_index_list_0):  # 16
                    new_v[i][m + 16] = v[i][n + 16]

            for i in range(16):
                for m, n in enumerate(last_index_list_0):  # 9
                    new_v[i][m + list_num[0] + 16] = v[i][n + 16 + 16]

            state_dict["decoder5.main.weight"] = new_v

        if k == "decoder5.main.bias":
            print("decoder5.main.bias")
            new_v = v
            state_dict["decoder5.main.bias"] = new_v

        if k == "fusion_m_scale4.main.weight":
            print("fusion_m_scale4.main.weight")
            new_v = torch.zeros(32, list_num[0] + list_num[1] + list_num[2] + list_num[3], 3, 3, 3)  # v (64,224,3,3,3)

            last_index_list_0 = []
            for j in range(list_num[0]):
                location = index[0][j][1]
                last_index_list_0.append(location)

            last_index_list_1 = []
            for j in range(list_num[1]):
                location = index[1][j][1]
                last_index_list_1.append(location)

            last_index_list_2 = []
            for j in range(list_num[2]):
                location = index[2][j][1]
                last_index_list_2.append(location)

            last_index_list_3 = []
            for j in range(list_num[3]):
                location = index[3][j][1]
                last_index_list_3.append(location)

            index_list = []
            for j in range(list_num[5]):
                location = index[5][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_0):  # 9
                    # print(f"i: {i}, m: {m}, j: {j}, n: {n}")
                    new_v[i][m] = v[j][n]

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_1):  # 12
                    new_v[i][m + list_num[0]] = v[j][n + 16]

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_2):  # 2
                    new_v[i][m + list_num[0] + list_num[1]] = v[j][n + 16 + 32]

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_3):  # 1
                    new_v[i][m + list_num[0] + list_num[1] + list_num[2]] = v[j][n + 16 + 32 + 32]

            state_dict["fusion_m_scale4.main.weight"] = new_v

        if k == "fusion_m_scale4.main.bias":
            print("fusion_m_scale4.main.bias")
            new_v = torch.zeros(32)
            new_v = v
            state_dict["fusion_m_scale4.main.bias"] = new_v

        if k == "fusion_m_scale3.main.weight":
            print("fusion_m_scale3.main.weight")
            new_v = torch.zeros(list_num[7], list_num[0] + list_num[1] + list_num[2] + list_num[3], 3, 3,
                                3)  # v (64,224,3,3,3)

            last_index_list_0 = []
            for j in range(list_num[0]):
                location = index[0][j][1]
                last_index_list_0.append(location)

            last_index_list_1 = []
            for j in range(list_num[1]):
                location = index[1][j][1]
                last_index_list_1.append(location)

            last_index_list_2 = []
            for j in range(list_num[2]):
                location = index[2][j][1]
                last_index_list_2.append(location)

            last_index_list_3 = []
            for j in range(list_num[3]):
                location = index[3][j][1]
                last_index_list_3.append(location)

            index_list = []
            for j in range(list_num[7]):
                location = index[7][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_0):  # 9
                    # print(f"i: {i}, m: {m}, j: {j}, n: {n}")
                    new_v[i][m] = v[j][n]

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_1):  # 12
                    new_v[i][m + list_num[0]] = v[j][n + 16]

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_2):  # 2
                    new_v[i][m + list_num[0] + list_num[1]] = v[j][n + 16 + 32]

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_3):  # 1
                    new_v[i][m + list_num[0] + list_num[1] + list_num[2]] = v[j][n + 16 + 32 + 32]

            state_dict["fusion_m_scale3.main.weight"] = new_v

        if k == "fusion_m_scale3.main.bias":
            print("fusion_m_scale3.main.bias")
            new_v = torch.zeros(list_num[7])
            index_list = []
            for j in range(list_num[7]):
                location = index[7][j][1]
                index_list.append(location)
                # print(location)
            for i, j in enumerate(index_list):
                new_v[i] = v[j]
            state_dict["fusion_m_scale3.main.bias"] = new_v

        if k == "fusion_m_scale2.main.weight":
            print("fusion_m_scale2.main.weight")
            new_v = torch.zeros(list_num[9], list_num[0] + list_num[1] + list_num[2] + list_num[3], 3, 3,
                                3)  # v (64,224,3,3,3)

            last_index_list_0 = []
            for j in range(list_num[0]):
                location = index[0][j][1]
                last_index_list_0.append(location)

            last_index_list_1 = []
            for j in range(list_num[1]):
                location = index[1][j][1]
                last_index_list_1.append(location)

            last_index_list_2 = []
            for j in range(list_num[2]):
                location = index[2][j][1]
                last_index_list_2.append(location)

            last_index_list_3 = []
            for j in range(list_num[3]):
                location = index[3][j][1]
                last_index_list_3.append(location)

            index_list = []
            for j in range(list_num[9]):
                location = index[9][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_0):  # 9
                    # print(f"i: {i}, m: {m}, j: {j}, n: {n}")
                    new_v[i][m] = v[j][n]

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_1):  # 12
                    new_v[i][m + list_num[0]] = v[j][n + 16]

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_2):  # 2
                    new_v[i][m + list_num[0] + list_num[1]] = v[j][n + 16 + 32]

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list_3):  # 1
                    new_v[i][m + list_num[0] + list_num[1] + list_num[2]] = v[j][n + 16 + 32 + 32]

            state_dict["fusion_m_scale2.main.weight"] = new_v

        if k == "fusion_m_scale2.main.bias":
            print("fusion_m_scale2.main.bias")
            new_v = torch.zeros(list_num[9])
            index_list = []
            for j in range(list_num[9]):
                location = index[9][j][1]
                index_list.append(location)
                # print(location)
            for i, j in enumerate(index_list):
                new_v[i] = v[j]
            state_dict["fusion_m_scale2.main.bias"] = new_v

        if k == "fusion_m_scale4_2.main.weight":
            print("fusion_m_scale4_2.main.weight")
            new_v = torch.zeros(32, 32, 3, 3, 3)

            last_index_list = []
            for j in range(list_num[5]):
                location = index[5][j][1]
                last_index_list.append(location)

            index_list = []
            for j in range(list_num[6]):
                location = index[6][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list):
                    # print(f"i: {i}, m: {m}, j: {j}, n: {n}")

                    new_v[i][m] = v[j][n]
            state_dict["fusion_m_scale4_2.main.weight"] = new_v

        if k == "fusion_m_scale4_2.main.bias":
            print("fusion_m_scale4_2.main.bias")
            new_v = torch.zeros(32)

            index_list = []
            for j in range(list_num[6]):
                location = index[6][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                new_v[i] = v[j]
            state_dict["fusion_m_scale4_2.main.bias"] = new_v

        if k == "fusion_m_scale3_2.main.weight":
            print("fusion_m_scale3_2.main.weight")
            new_v = torch.zeros(list_num[8], list_num[7], 3, 3, 3)

            last_index_list = []
            for j in range(list_num[7]):
                location = index[7][j][1]
                last_index_list.append(location)

            index_list = []
            for j in range(list_num[8]):
                location = index[8][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list):
                    # print(f"i: {i}, m: {m}, j: {j}, n: {n}")

                    new_v[i][m] = v[j][n]
            state_dict["fusion_m_scale3_2.main.weight"] = new_v

        if k == "fusion_m_scale3_2.main.bias":
            print("fusion_m_scale3_2.main.bias")
            new_v = torch.zeros(list_num[8])

            index_list = []
            for j in range(list_num[8]):
                location = index[8][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                new_v[i] = v[j]
            state_dict["fusion_m_scale3_2.main.bias"] = new_v

        if k == "fusion_m_scale2_2.main.weight":
            print("fusion_m_scale2_2.main.weight")
            new_v = torch.zeros(list_num[10], list_num[9], 3, 3, 3)

            last_index_list = []
            for j in range(list_num[9]):
                location = index[9][j][1]
                last_index_list.append(location)

            index_list = []
            for j in range(list_num[10]):
                location = index[10][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                for m, n in enumerate(last_index_list):
                    # print(f"i: {i}, m: {m}, j: {j}, n: {n}")

                    new_v[i][m] = v[j][n]
            state_dict["fusion_m_scale2_2.main.weight"] = new_v

        if k == "fusion_m_scale2_2.main.bias":
            print("fusion_m_scale2_2.main.bias")
            new_v = torch.zeros(list_num[10])

            index_list = []
            for j in range(list_num[10]):
                location = index[10][j][1]
                index_list.append(location)

            for i, j in enumerate(index_list):
                new_v[i] = v[j]
            state_dict["fusion_m_scale2_2.main.bias"] = new_v

        if k == "output_block_1.0.main.weight":
            new_v = v
            state_dict["output_block_1.0.main.weight"] = new_v

        if k == "output_block_1.0.main.bias":
            new_v = v
            state_dict["output_block_1.0.main.bias"] = new_v

        if k == "output_block_1.1.main.weight":
            new_v = v
            state_dict["output_block_1.1.main.weight"] = new_v

        if k == "output_block_1.1.main.bias":
            new_v = v
            state_dict["output_block_1.1.main.bias"] = new_v

        if k == "output_block_2.0.main.weight":
            new_v = v
            state_dict["output_block_2.0.main.weight"] = new_v

        if k == "output_block_2.0.main.bias":
            new_v = v
            state_dict["output_block_2.0.main.bias"] = new_v

        if k == "output_block_2.1.main.weight":
            new_v = v
            state_dict["output_block_2.1.main.weight"] = new_v

        if k == "output_block_2.1.main.bias":
            new_v = v
            state_dict["output_block_2.1.main.bias"] = new_v

        if k == "output_block_3.0.main.weight":
            new_v = v
            state_dict["output_block_3.0.main.weight"] = new_v

        if k == "output_block_3.0.main.bias":
            new_v = v
            state_dict["output_block_3.0.main.bias"] = new_v

        if k == "output_block_3.1.main.weight":
            new_v = v
            state_dict["output_block_3.1.main.weight"] = new_v

        if k == "output_block_3.1.main.bias":
            new_v = v
            state_dict["output_block_3.1.main.bias"] = new_v

        if k == "output_block.0.main.weight":
            new_v = v
            state_dict["output_block.0.main.weight"] = new_v

        if k == "output_block.0.main.bias":
            new_v = v
            state_dict["output_block.0.main.bias"] = new_v

        if k == "output_block.1.main.weight":
            new_v = v
            state_dict["output_block.1.main.weight"] = new_v

        if k == "output_block.1.main.bias":
            new_v = v
            state_dict["output_block.1.main.bias"] = new_v

        if k == "flow.weight":
            new_v = v
            state_dict["flow.weight"] = new_v

        if k == "flow.bias":
            new_v = v
            state_dict["flow.bias"] = new_v

    # # 加载参数
    # for k,v in checkpoint.items():
    #     print(k)
    #     print(v.shape)

    #     # ---------------- ENCODER ----------------
    #     if  k == "encoder1_m.main.weight":
    #         print("encoder1_m.main.weight")
    #         new_v = torch.zeros(list_num[0],1,3,3,3)
    #         for j in range(list_num[0]):
    #             location = index[0][j][1]
    #             alpha_val = gate_alpha[0].flatten()[location].item() # 提取当前通道的 alpha
    #             new_v[j] = v[location] * alpha_val # 权重注入 alpha
    #         state_dict["encoder1_m.main.weight"] = new_v

    #     if  k == "encoder1_m.main.bias":
    #         print("encoder1_m.main.bias")
    #         new_v = torch.zeros(list_num[0])
    #         for j in range(list_num[0]):
    #             location = index[0][j][1]
    #             alpha_val = gate_alpha[0].flatten()[location].item()
    #             new_v[j] = v[location] * alpha_val # 偏置同样注入 alpha
    #         state_dict["encoder1_m.main.bias"] = new_v

    #     if  k == "encoder2_m.main.weight":
    #         print("encoder2_m.main.weight")
    #         new_v = torch.zeros(list_num[1],list_num[0],3,3,3)
    #         last_index_list = [index[0][j][1] for j in range(list_num[0])]
    #         for i in range(list_num[1]):
    #             location = index[1][i][1]
    #             alpha_val = gate_alpha[1].flatten()[location].item()
    #             for m, n in enumerate(last_index_list):
    #                 new_v[i][m] = v[location][n] * alpha_val
    #         state_dict["encoder2_m.main.weight"] = new_v

    #     if  k == "encoder2_m.main.bias":
    #         print("encoder2_m.main.bias")
    #         new_v = torch.zeros(list_num[1])
    #         for i in range(list_num[1]):
    #             location = index[1][i][1]
    #             alpha_val = gate_alpha[1].flatten()[location].item()
    #             new_v[i] = v[location] * alpha_val
    #         state_dict["encoder2_m.main.bias"] = new_v

    #     if  k == "encoder3_m.main.weight":
    #         print("encoder3_m.main.weight")
    #         new_v = torch.zeros(list_num[2],list_num[1],3,3,3)
    #         last_index_list = [index[1][j][1] for j in range(list_num[1])]
    #         for i in range(list_num[2]):
    #             location = index[2][i][1]
    #             alpha_val = gate_alpha[2].flatten()[location].item()
    #             for m, n in enumerate(last_index_list):
    #                 new_v[i][m] = v[location][n] * alpha_val
    #         state_dict["encoder3_m.main.weight"] = new_v

    #     if  k == "encoder3_m.main.bias":
    #         print("encoder3_m.main.bias")
    #         new_v = torch.zeros(list_num[2])
    #         for i in range(list_num[2]):
    #             location = index[2][i][1]
    #             alpha_val = gate_alpha[2].flatten()[location].item()
    #             new_v[i] = v[location] * alpha_val
    #         state_dict["encoder3_m.main.bias"] = new_v

    #     if  k == "encoder4_m.main.weight":
    #         print("encoder4_m.main.weight")
    #         new_v = torch.zeros(list_num[3],list_num[2],3,3,3)
    #         last_index_list = [index[2][j][1] for j in range(list_num[2])]
    #         for i in range(list_num[3]):
    #             location = index[3][i][1]
    #             alpha_val = gate_alpha[3].flatten()[location].item()
    #             for m, n in enumerate(last_index_list):
    #                 new_v[i][m] = v[location][n] * alpha_val
    #         state_dict["encoder4_m.main.weight"] = new_v

    #     if  k == "encoder4_m.main.bias":
    #         print("encoder4_m.main.bias")
    #         new_v = torch.zeros(list_num[3])
    #         for i in range(list_num[3]):
    #             location = index[3][i][1]
    #             alpha_val = gate_alpha[3].flatten()[location].item()
    #             new_v[i] = v[location] * alpha_val
    #         state_dict["encoder4_m.main.bias"] = new_v

    #     if  k == "encoder5_m.main.weight":
    #         print("encoder5_m.main.weight")
    #         new_v = torch.zeros(list_num[4],list_num[3],3,3,3)
    #         last_index_list = [index[3][j][1] for j in range(list_num[3])]
    #         for i in range(list_num[4]):
    #             location = index[4][i][1]
    #             alpha_val = gate_alpha[4].flatten()[location].item()
    #             for m, n in enumerate(last_index_list):
    #                 new_v[i][m] = v[location][n] * alpha_val
    #         state_dict["encoder5_m.main.weight"] = new_v

    #     if  k == "encoder5_m.main.bias":
    #         print("encoder5_m.main.bias")
    #         new_v = torch.zeros(list_num[4])
    #         for i in range(list_num[4]):
    #             location = index[4][i][1]
    #             alpha_val = gate_alpha[4].flatten()[location].item()
    #             new_v[i] = v[location] * alpha_val
    #         state_dict["encoder5_m.main.bias"] = new_v

    #     if  k == "decoder1.main.weight":
    #         print("decoder1.main.weight")
    #         new_v = torch.zeros(32,64,3,3,3) # v(64,128,3,3,3)
    #         new_v = v
    #         state_dict["decoder1.main.weight"] = new_v

    #     if  k == "decoder1.main.bias":
    #         print("decoder1.main.bias")
    #         new_v = torch.zeros(32)
    #         new_v = v
    #         state_dict["decoder1.main.bias"] = new_v

    #     if  k == "decoder2.main.weight":
    #         print("decoder2.main.weight")
    #         new_v = torch.zeros(32,96,3,3,3) # v(64,128,3,3,3)
    #         new_v = v
    #         state_dict["decoder2.main.weight"] = new_v

    #     if  k == "decoder2.main.bias":
    #         print("decoder2.main.bias")
    #         new_v = torch.zeros(32)
    #         new_v = v
    #         state_dict["decoder2.main.bias"] = new_v

    #     if  k == "decoder3.main.weight":
    #         print("decoder3.main.weight")
    #         new_v = torch.zeros(32,32+list_num[8]+list_num[8],3,3,3) # v(64,128,3,3,3)

    #         last_index_list_0 = []
    #         for j in range(list_num[8]):
    #             location = index[8][j][1]
    #             last_index_list_0.append(location)

    #         for i in range(32):
    #             for j in range(32):
    #                 new_v[i][j] = v[i][j]

    #         for i in range(32):
    #             for m,n in enumerate(last_index_list_0): # 16
    #                 new_v[i][m+32] = v[i][n+32]

    #         for i in range(32):
    #             for m,n in enumerate(last_index_list_0): # 9
    #                 new_v[i][m+list_num[8]+32] = v[i][n+32+32]

    #         state_dict["decoder3.main.weight"] = new_v

    #     if  k == "decoder3.main.bias":
    #         print("decoder3.main.bias")
    #         new_v = torch.zeros(32)
    #         new_v = v
    #         state_dict["decoder3.main.bias"] = new_v

    #     if  k == "decoder4.main.weight":
    #         print("decoder4.main.weight")
    #         new_v = torch.zeros(16,list_num[10]+list_num[10]+32,3,3,3) # v(64,128,3,3,3)

    #         last_index_list_0 = []
    #         for j in range(list_num[10]):
    #             location = index[10][j][1]
    #             last_index_list_0.append(location)

    #         for i in range(16):
    #             for j in range(32):
    #                 new_v[i][j] = v[i][j]

    #         for i in range(16):
    #             for m,n in enumerate(last_index_list_0): # 16
    #                 new_v[i][m+32] = v[i][n+32]

    #         for i in range(16):
    #             for m,n in enumerate(last_index_list_0): # 9
    #                 new_v[i][m+list_num[10]+32] = v[i][n+32+32]

    #         state_dict["decoder4.main.weight"] = new_v

    #     if  k == "decoder4.main.bias":
    #         print("decoder4.main.bias")
    #         new_v = torch.zeros(16)
    #         new_v = v
    #         state_dict["decoder4.main.bias"] = new_v

    #     if  k == "decoder5.main.weight":
    #         print("decoder5.main.weight")
    #         new_v = torch.zeros(16,16+list_num[0]+list_num[0],3,3,3)  # v (32,96,3,3,3)

    #         last_index_list_0 = []
    #         for j in range(list_num[0]):
    #             location = index[0][j][1]
    #             last_index_list_0.append(location)

    #         for i in range(16):
    #             for j in range(16):
    #                 new_v[i][j] = v[i][j]

    #         for i in range(16):
    #             for m,n in enumerate(last_index_list_0): # 16
    #                 new_v[i][m+16] = v[i][n+16]

    #         for i in range(16):
    #             for m,n in enumerate(last_index_list_0): # 9
    #                 new_v[i][m+list_num[0]+16] = v[i][n+16+16]

    #         state_dict["decoder5.main.weight"] = new_v

    #     if  k == "decoder5.main.bias":
    #         print("decoder5.main.bias")
    #         new_v = v
    #         state_dict["decoder5.main.bias"] = new_v

    #     if  k == "fusion_m_scale4.main.weight":
    #         print("fusion_m_scale4.main.weight")
    #         new_v = torch.zeros(32,list_num[0]+list_num[1]+list_num[2]+list_num[3],3,3,3)  # v (64,224,3,3,3)

    #         last_index_list_0 = []
    #         for j in range(list_num[0]):
    #             location = index[0][j][1]
    #             last_index_list_0.append(location)

    #         last_index_list_1 = []
    #         for j in range(list_num[1]):
    #             location = index[1][j][1]
    #             last_index_list_1.append(location)

    #         last_index_list_2 = []
    #         for j in range(list_num[2]):
    #             location = index[2][j][1]
    #             last_index_list_2.append(location)

    #         last_index_list_3 = []
    #         for j in range(list_num[3]):
    #             location = index[3][j][1]
    #             last_index_list_3.append(location)

    #         index_list = []
    #         for j in range(list_num[5]):
    #             location = index[5][j][1]
    #             index_list.append(location)

    #         for i,j in enumerate(index_list):
    #             alpha_val = gate_alpha[5].flatten()[j].item() # 提取当前通道的 alpha
    #             for m,n in enumerate(last_index_list_0): # 9
    #                 # print(f"i: {i}, m: {m}, j: {j}, n: {n}")
    #                 new_v[i][m] = v[j][n] * alpha_val

    #         for i,j in enumerate(index_list):
    #             alpha_val = gate_alpha[5].flatten()[j].item()
    #             for m,n in enumerate(last_index_list_1): # 12
    #                 new_v[i][m+list_num[0]] = v[j][n+16] * alpha_val

    #         for i,j in enumerate(index_list):
    #             alpha_val = gate_alpha[5].flatten()[j].item()
    #             for m,n in enumerate(last_index_list_2): # 2
    #                 new_v[i][m+list_num[0]+list_num[1]] = v[j][n+16+32] * alpha_val

    #         for i,j in enumerate(index_list):
    #             alpha_val = gate_alpha[5].flatten()[j].item()
    #             for m,n in enumerate(last_index_list_3): # 1
    #                 new_v[i][m+list_num[0]+list_num[1]+list_num[2]] = v[j][n+16+32+32] * alpha_val

    #         state_dict["fusion_m_scale4.main.weight"] = new_v

    #     if  k == "fusion_m_scale4.main.bias":
    #         print("fusion_m_scale4.main.bias")
    #         new_v = torch.zeros(32)
    #         index_list = []
    #         for j in range(list_num[5]):
    #             location = index[5][j][1]
    #             index_list.append(location)

    #         for i,j in enumerate(index_list):
    #             alpha_val = gate_alpha[5].flatten()[j].item() # 提取当前通道的 alpha
    #             new_v[i] = v[j] * alpha_val
    #         state_dict["fusion_m_scale4.main.bias"] = new_v

    #     if  k == "fusion_m_scale3.main.weight":
    #         print("fusion_m_scale3.main.weight")
    #         new_v = torch.zeros(list_num[7],list_num[0]+list_num[1]+list_num[2]+list_num[3],3,3,3)  # v (64,224,3,3,3)

    #         last_index_list_0 = []
    #         for j in range(list_num[0]):
    #             location = index[0][j][1]
    #             last_index_list_0.append(location)

    #         last_index_list_1 = []
    #         for j in range(list_num[1]):
    #             location = index[1][j][1]
    #             last_index_list_1.append(location)

    #         last_index_list_2 = []
    #         for j in range(list_num[2]):
    #             location = index[2][j][1]
    #             last_index_list_2.append(location)

    #         last_index_list_3 = []
    #         for j in range(list_num[3]):
    #             location = index[3][j][1]
    #             last_index_list_3.append(location)

    #         index_list = []
    #         for j in range(list_num[7]):
    #             location = index[7][j][1]
    #             index_list.append(location)

    #         for i,j in enumerate(index_list):
    #             alpha_val = gate_alpha[7].flatten()[j].item() # 提取当前通道的 alpha
    #             for m,n in enumerate(last_index_list_0): # 9
    #                 new_v[i][m] = v[j][n] * alpha_val

    #         for i,j in enumerate(index_list):
    #             alpha_val = gate_alpha[7].flatten()[j].item()
    #             for m,n in enumerate(last_index_list_1): # 12
    #                 new_v[i][m+list_num[0]] = v[j][n+16] * alpha_val

    #         for i,j in enumerate(index_list):
    #             alpha_val = gate_alpha[7].flatten()[j].item()
    #             for m,n in enumerate(last_index_list_2): # 2
    #                 new_v[i][m+list_num[0]+list_num[1]] = v[j][n+16+32] * alpha_val

    #         for i,j in enumerate(index_list):
    #             alpha_val = gate_alpha[7].flatten()[j].item()
    #             for m,n in enumerate(last_index_list_3): # 1
    #                 new_v[i][m+list_num[0]+list_num[1]+list_num[2]] = v[j][n+16+32+32] * alpha_val

    #         state_dict["fusion_m_scale3.main.weight"] = new_v

    #     if  k == "fusion_m_scale3.main.bias":
    #         print("fusion_m_scale3.main.bias")
    #         new_v = torch.zeros(list_num[7])
    #         index_list = []
    #         for j in range(list_num[7]):
    #             location = index[7][j][1]
    #             index_list.append(location)
    #             # print(location)
    #         for i,j in enumerate(index_list):
    #             alpha_val = gate_alpha[7].flatten()[j].item()
    #             new_v[i] = v[j] * alpha_val
    #         state_dict["fusion_m_scale3.main.bias"] = new_v

    #     if  k == "fusion_m_scale2.main.weight":
    #         print("fusion_m_scale2.main.weight")
    #         new_v = torch.zeros(list_num[9],list_num[0]+list_num[1]+list_num[2]+list_num[3],3,3,3)  # v (64,224,3,3,3)

    #         last_index_list_0 = []
    #         for j in range(list_num[0]):
    #             location = index[0][j][1]
    #             last_index_list_0.append(location)

    #         last_index_list_1 = []
    #         for j in range(list_num[1]):
    #             location = index[1][j][1]
    #             last_index_list_1.append(location)

    #         last_index_list_2 = []
    #         for j in range(list_num[2]):
    #             location = index[2][j][1]
    #             last_index_list_2.append(location)

    #         last_index_list_3 = []
    #         for j in range(list_num[3]):
    #             location = index[3][j][1]
    #             last_index_list_3.append(location)

    #         index_list = []
    #         for j in range(list_num[9]):
    #             location = index[9][j][1]
    #             index_list.append(location)

    #         for i,j in enumerate(index_list):
    #             alpha_val = gate_alpha[9].flatten()[j].item()
    #             for m,n in enumerate(last_index_list_0): # 9
    #                 new_v[i][m] = v[j][n] * alpha_val

    #         for i,j in enumerate(index_list):
    #             alpha_val = gate_alpha[9].flatten()[j].item()
    #             for m,n in enumerate(last_index_list_1): # 12
    #                 new_v[i][m+list_num[0]] = v[j][n+16] * alpha_val

    #         for i,j in enumerate(index_list):
    #             alpha_val = gate_alpha[9].flatten()[j].item()
    #             for m,n in enumerate(last_index_list_2): # 2
    #                 new_v[i][m+list_num[0]+list_num[1]] = v[j][n+16+32] * alpha_val

    #         for i,j in enumerate(index_list):
    #             alpha_val = gate_alpha[9].flatten()[j].item()
    #             for m,n in enumerate(last_index_list_3): # 1
    #                 new_v[i][m+list_num[0]+list_num[1]+list_num[2]] = v[j][n+16+32+32] * alpha_val

    #         state_dict["fusion_m_scale2.main.weight"] = new_v

    #     if  k == "fusion_m_scale2.main.bias":
    #         print("fusion_m_scale2.main.bias")
    #         new_v = torch.zeros(list_num[9])
    #         index_list = []
    #         for j in range(list_num[9]):
    #             location = index[9][j][1]
    #             index_list.append(location)
    #             # print(location)
    #         for i,j in enumerate(index_list):
    #             alpha_val = gate_alpha[9].flatten()[j].item()
    #             new_v[i] = v[j] * alpha_val
    #         state_dict["fusion_m_scale2.main.bias"] = new_v

    #     if  k == "fusion_m_scale4_2.main.weight":
    #         print("fusion_m_scale4_2.main.weight")
    #         new_v = torch.zeros(32,32,3,3,3)

    #         last_index_list = []
    #         for j in range(list_num[5]):
    #             location = index[5][j][1]
    #             last_index_list.append(location)

    #         index_list = []
    #         for j in range(list_num[6]):
    #             location = index[6][j][1]
    #             index_list.append(location)

    #         for i,j in enumerate(index_list):
    #             alpha_val = gate_alpha[6].flatten()[j].item()
    #             for m,n in enumerate(last_index_list):
    #                 new_v[i][m] = v[j][n] * alpha_val
    #         state_dict["fusion_m_scale4_2.main.weight"] = new_v

    #     if  k == "fusion_m_scale4_2.main.bias":
    #         print("fusion_m_scale4_2.main.bias")
    #         new_v = torch.zeros(32)

    #         index_list = []
    #         for j in range(list_num[6]):
    #             location = index[6][j][1]
    #             index_list.append(location)

    #         for i,j in enumerate(index_list):
    #             alpha_val = gate_alpha[6].flatten()[j].item()
    #             new_v[i]= v[j] * alpha_val
    #         state_dict["fusion_m_scale4_2.main.bias"] = new_v

    #     if  k == "fusion_m_scale3_2.main.weight":
    #         print("fusion_m_scale3_2.main.weight")
    #         new_v = torch.zeros(list_num[8],list_num[7],3,3,3)

    #         last_index_list = []
    #         for j in range(list_num[7]):
    #             location = index[7][j][1]
    #             last_index_list.append(location)

    #         index_list = []
    #         for j in range(list_num[8]):
    #             location = index[8][j][1]
    #             index_list.append(location)

    #         for i,j in enumerate(index_list):
    #             alpha_val = gate_alpha[8].flatten()[j].item()
    #             for m,n in enumerate(last_index_list):
    #                 new_v[i][m] = v[j][n] * alpha_val
    #         state_dict["fusion_m_scale3_2.main.weight"] = new_v

    #     if  k == "fusion_m_scale3_2.main.bias":
    #         print("fusion_m_scale3_2.main.bias")
    #         new_v = torch.zeros(list_num[8])

    #         index_list = []
    #         for j in range(list_num[8]):
    #             location = index[8][j][1]
    #             index_list.append(location)

    #         for i,j in enumerate(index_list):
    #             alpha_val = gate_alpha[8].flatten()[j].item()
    #             new_v[i]= v[j] * alpha_val
    #         state_dict["fusion_m_scale3_2.main.bias"] = new_v

    #     if  k == "fusion_m_scale2_2.main.weight":
    #         print("fusion_m_scale2_2.main.weight")
    #         new_v = torch.zeros(list_num[10],list_num[9],3,3,3)

    #         last_index_list = []
    #         for j in range(list_num[9]):
    #             location = index[9][j][1]
    #             last_index_list.append(location)

    #         index_list = []
    #         for j in range(list_num[10]):
    #             location = index[10][j][1]
    #             index_list.append(location)

    #         for i,j in enumerate(index_list):
    #             alpha_val = gate_alpha[10].flatten()[j].item()
    #             for m,n in enumerate(last_index_list):
    #                 new_v[i][m] = v[j][n] * alpha_val
    #         state_dict["fusion_m_scale2_2.main.weight"] = new_v

    #     if  k == "fusion_m_scale2_2.main.bias":
    #         print("fusion_m_scale2_2.main.bias")
    #         new_v = torch.zeros(list_num[10])

    #         index_list = []
    #         for j in range(list_num[10]):
    #             location = index[10][j][1]
    #             index_list.append(location)

    #         for i,j in enumerate(index_list):
    #             alpha_val = gate_alpha[10].flatten()[j].item()
    #             new_v[i]= v[j] * alpha_val
    #         state_dict["fusion_m_scale2_2.main.bias"] = new_v

    #     if  k == "output_block_1.0.main.weight":
    #         new_v= v
    #         state_dict["output_block_1.0.main.weight"] = new_v

    #     if  k == "output_block_1.0.main.bias":
    #         new_v = v
    #         state_dict["output_block_1.0.main.bias"] = new_v

    #     if  k == "output_block_1.1.main.weight":
    #         new_v = v
    #         state_dict["output_block_1.1.main.weight"] = new_v

    #     if  k == "output_block_1.1.main.bias":
    #         new_v = v
    #         state_dict["output_block_1.1.main.bias"] = new_v

    #     if  k == "output_block_2.0.main.weight":
    #         new_v = v
    #         state_dict["output_block_2.0.main.weight"] = new_v

    #     if  k == "output_block_2.0.main.bias":
    #         new_v = v
    #         state_dict["output_block_2.0.main.bias"] = new_v

    #     if  k == "output_block_2.1.main.weight":
    #         new_v = v
    #         state_dict["output_block_2.1.main.weight"] = new_v

    #     if  k == "output_block_2.1.main.bias":
    #         new_v = v
    #         state_dict["output_block_2.1.main.bias"] = new_v

    #     if  k == "output_block_3.0.main.weight":
    #         new_v = v
    #         state_dict["output_block_3.0.main.weight"] = new_v

    #     if  k == "output_block_3.0.main.bias":
    #         new_v = v
    #         state_dict["output_block_3.0.main.bias"] = new_v

    #     if  k == "output_block_3.1.main.weight":
    #         new_v = v
    #         state_dict["output_block_3.1.main.weight"] = new_v

    #     if  k == "output_block_3.1.main.bias":
    #         new_v = v
    #         state_dict["output_block_3.1.main.bias"] = new_v

    #     if  k == "output_block.0.main.weight":
    #         new_v = v
    #         state_dict["output_block.0.main.weight"] = new_v

    #     if  k == "output_block.0.main.bias":
    #         new_v = v
    #         state_dict["output_block.0.main.bias"] = new_v

    #     if  k == "output_block.1.main.weight":
    #         new_v = v
    #         state_dict["output_block.1.main.weight"] = new_v

    #     if  k == "output_block.1.main.bias":
    #         new_v = v
    #         state_dict["output_block.1.main.bias"] = new_v

    #     if  k == "flow.weight":
    #         new_v = v
    #         state_dict["flow.weight"] = new_v

    #     if  k == "flow.bias":
    #         new_v = v
    #         state_dict["flow.bias"] = new_v

    #     # Define the target block prefixes as a tuple

    target_prefixes = ("trans_2.", "trans_3.", "trans_4.", "trans_5.")

    # Assuming 'checkpoint_dict' is the dictionary loaded from your saved file,
    # and 'model_state_dict' is the current state dictionary of your model.
    for k, v in checkpoint.items():
        # .startswith() efficiently checks if the key 'k' starts with any of the prefixes
        if k.startswith(target_prefixes):
            # This will correctly capture and assign all sub-module weights,
            # such as 'trans_2.norm.weight' or 'trans_4.attn.qkv.bias'
            state_dict[k] = v

    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

    # # 统计参数量
    dummy_input = torch.randn(1, 1, 192, 160, 192).to(device)
    flops, params = profile(model, (dummy_input, dummy_input,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))

    # # 对验证集进行测试
    Dice = 0
    Dice_1 = 0
    Dice_2 = 0
    Dice_3 = 0
    mean_dice = 0
    num = 0
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = torch.zeros(90)
    model.eval()
    print("---------验证集的Dice----------")
    SPT = vxm.torch.layers.SpatialTransformer(inshape, mode="nearest").to(device)
    for step in range(len(val_t1_files)):
        # 提取t1数据
        inputs_fixed, y_true = next(val_t1_generator)
        inputs_fixed = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs_fixed]
        # 提取对应t1的seg数据
        seg_inputs_fixed, seg_y_true = next(val_seg_generator)
        seg_inputs_fixed = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in seg_inputs_fixed]
        for step in range(len(val_t1_files)):
            # 提取t1数据
            inputs_moving, y_true = next(val_t1_generator)
            inputs_moving = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs_moving]
            # 提取对应t1的seg数据
            seg_inputs_moving, seg_y_true = next(val_seg_generator)
            seg_inputs_moving = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in
                                 seg_inputs_moving]
            if step < len(val_t1_files) - 1:
                with torch.no_grad():
                    # 生成flow
                    starter.record()

                    flow_1, flow_2, flow_3, flow, delta_flow_2, delta_flow_3, _ = model(inputs_moving[0],
                                                                                        inputs_fixed[0])

                    # flow_1,flow_2,flow_3,flow,delta_flow_2,delta_flow_3,_,_,_ = model_pretrain(inputs_moving[0],inputs_fixed[0])

                    ender.record()
                    # 同步GPU时间
                    torch.cuda.synchronize()
                    curr_time = starter.elapsed_time(ender)  # 计算时间
                    times[num] = curr_time
                    # num = num + 1
                    print(curr_time)

                    # x_in = torch.cat((inputs_moving[0],inputs_fixed[0]), dim=1)
                    # flow_1,flow_2,flow_3,flow,delta_flow_2,delta_flow_3,_ = model(inputs_moving[0],inputs_fixed[0])

                    # 上采样
                    flow_1_up = nn.functional.interpolate(flow_1, scale_factor=8, mode="trilinear") * 8
                    flow_2_up = nn.functional.interpolate(flow_2, scale_factor=4, mode="trilinear") * 4
                    flow_3_up = nn.functional.interpolate(flow_3, scale_factor=2, mode="trilinear") * 2

                    seg_pre_1 = SPT(seg_inputs_moving[0], flow_1_up)
                    seg_pre_2 = SPT(seg_inputs_moving[0], flow_2_up)
                    seg_pre_3 = SPT(seg_inputs_moving[0], flow_3_up)
                    dice_1, _ = vxm.torch.losses.compute_dice(seg_pre_1, seg_inputs_fixed[0])
                    dice_2, _ = vxm.torch.losses.compute_dice(seg_pre_2, seg_inputs_fixed[0])
                    dice_3, _ = vxm.torch.losses.compute_dice(seg_pre_3, seg_inputs_fixed[0])

                    # 得到变形场2
                    seg_pre = SPT(seg_inputs_moving[0], flow)
                    dice, dice_region = vxm.torch.losses.compute_dice(seg_pre, seg_inputs_fixed[0])
                    if dice < 1:
                        print("dice", dice)
                        Dice += dice
                        num += 1
        print("----------换fixed image---------")

    metric = Dice / num
    print("mean_dice", metric)
    mean_time = times.mean().item()
    print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000 / mean_time))

    torch.save(model.state_dict(),
               "/home/boys/project/voxelmorph/voxelmorph_code/models/comparison_methods/abdomenCT/daul_pyramid_PFNet_FFM_normal_GDP_adaptive_nopretrain_trans.pth")


if __name__ == "__main__":
    # train_dual_pyramid_vxm_trans_FFM_4layer_normal_GDP()
    # train_dual_pyramid_vxm_FFM_4layer_normal_GDP()
    # train_dual_pyramid_vxm_FFM_4layer_large_GDP()
    # train_dual_pyramid_vxm_PDFNet_normal_FFM_diff_GDP()
    train_dual_pyramid_vxm_FFM_4layer_huge_GDP()



    





    