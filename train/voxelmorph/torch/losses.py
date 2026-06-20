import torch
import torch.nn.functional as F
import numpy as np
import math
import surface_distance as surfdist
import torchio as tio
import torch.nn as nn
from torch.autograd import Variable


class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)


class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)


class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=2, reduction='mean'):
        """
        focal_loss , -αt(1-pt)**gamma * log(pt)
        :param alpha:   balance class. list(alpha = alpha) or constant([alpha, 1-alpha, 1-alpha, ...]), default 0.25
        :param gamma:   gamma, default 2
        :param num_classes:  num classes
        :param reduction: mean or sum, default mean
        """

        super().__init__()
        self.reduction = reduction
        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1   # background decay
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha) #  [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss forward
        :param preds:   size:[N, T, C] or [T, C]    N: batch size T: video length C: num classes
        :param labels:  size:[N, T] or [T]
        :return:
        """
        preds = preds.view(-1, preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_softmax = F.softmax(preds, dim=1)
        preds_logsoft = torch.log(preds_softmax)
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1).type(torch.long))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1).type(torch.long))
        self.alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)
        loss = torch.mul(self.alpha, loss.t())
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, _, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad

class Grad2d(torch.nn.Module):
    """
    2D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        super(Grad2d, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred, y_true):
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx

        d = torch.mean(dx) + torch.mean(dy)
        grad = d / 2.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad


class Grad3d(torch.nn.Module):
    """
    3D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        super(Grad3d, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad


def JacboianDet(flow):
    '''
    计算输入变形场flow的雅克比行列式,要求输入变形场的shape=[B,H,W,D,C]
    但是网络输出的预测flow shape=[B,C,W,H,D]
    所以需要permute(0,3,2,4,1)
    Calculate the Jacobian value at each point of the displacement map having
    size of b*h*w*d*3 and in the cubic volumn of [-1, 1]^3
    '''
    D_y = (flow[:, 1:, :-1, :-1, :] - flow[:, :-1, :-1, :-1, :])
    D_x = (flow[:, :-1, 1:, :-1, :] - flow[:, :-1, :-1, :-1, :])
    D_z = (flow[:, :-1, :-1, 1:, :] - flow[:, :-1, :-1, :-1, :])
    D1 = (D_x[..., 0] + 1) * ((D_y[..., 1] + 1) * (D_z[..., 2] + 1) - D_z[..., 1] * D_y[..., 2])
    D2 = (D_x[..., 1]) * (D_y[..., 0] * (D_z[..., 2] + 1) - D_y[..., 2] * D_x[..., 0])
    D3 = (D_x[..., 2]) * (D_y[..., 0] * D_z[..., 1] - (D_y[..., 1] + 1) * D_z[..., 0])
    return D1 - D2 + D3

def neg_Jbet_num(ypred):
    '''
    输入预测的变形场ypred, 输出所有负行列式的个数
    Penalizing locations where Jacobian has negative determinants
    '''
    Neg_Jac = 0.5 * (torch.abs(JacboianDet(ypred)) - JacboianDet(ypred))  # 只有行列式为负, 该体素才会被保留下来
    Neg_Jac_value = torch.sum(Neg_Jac)  # 所有行列式为负的体素求和, 求是所有负行列式值之和, 不是负值体素个数【这个据说是voxelmorph定义的雅克比loss】
    Neg_Jac_num = torch.nonzero(Neg_Jac).shape[0]  # 所有行列式为负的体素个数之和
    return Neg_Jac_value,Neg_Jac_num

def neg_Jbet_loss(ypred):
    '''
    基于LapIRN的公式, 给出雅克比行列式对应的抗折叠损失函数, 其中涉及relu
    '''
    neg_Jdet = -1.0 * JacboianDet(ypred)
    selected_neg_Jdet = F.relu(neg_Jdet)

    return torch.mean(selected_neg_Jdet)


'''
互信息(MI)损失函数, 一般用于多模态配准
'''
class MutualInformation(torch.nn.Module):
    """
    Mutual Information
    """

    def __init__(self, sigma_ratio=1, minval=0., maxval=1., num_bin=32):
        super(MutualInformation, self).__init__()

        """Create bin centers"""
        bin_centers = np.linspace(minval, maxval, num=num_bin)
        vol_bin_centers = Variable(torch.linspace(minval, maxval, num_bin), requires_grad=False).cuda()
        num_bins = len(bin_centers)

        """Sigma for Gaussian approx."""
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio
        # print(sigma)

        self.preterm = 1 / (2 * sigma ** 2)
        self.bin_centers = bin_centers
        self.max_clip = maxval
        self.num_bins = num_bins
        self.vol_bin_centers = vol_bin_centers

    def mi(self, y_true, y_pred):
        y_pred = torch.clamp(y_pred, 0., self.max_clip)
        y_true = torch.clamp(y_true, 0, self.max_clip)

        y_true = y_true.contiguous()
        y_pred = y_pred.contiguous()
        y_true = y_true.view(y_true.shape[0], -1)
        y_true = torch.unsqueeze(y_true, 2)
        y_pred = y_pred.view(y_pred.shape[0], -1)
        y_pred = torch.unsqueeze(y_pred, 2)

        nb_voxels = y_pred.shape[1]  # total num of voxels

        """Reshape bin centers"""
        o = [1, 1, np.prod(self.vol_bin_centers.shape)]
        vbc = torch.reshape(self.vol_bin_centers, o).cuda()

        """compute image terms by approx. Gaussian dist."""
        I_a = torch.exp(- self.preterm * torch.square(y_true - vbc))
        I_a = I_a / torch.sum(I_a, dim=-1, keepdim=True)

        I_b = torch.exp(- self.preterm * torch.square(y_pred - vbc))
        I_b = I_b / torch.sum(I_b, dim=-1, keepdim=True)

        # compute probabilities
        pab = torch.bmm(I_a.permute(0, 2, 1), I_b)
        pab = pab / nb_voxels
        pa = torch.mean(I_a, dim=1, keepdim=True)
        pb = torch.mean(I_b, dim=1, keepdim=True)

        papb = torch.bmm(pa.permute(0, 2, 1), pb) + 1e-6
        mi = torch.sum(torch.sum(pab * torch.log(pab / papb + 1e-6), dim=1), dim=1)
        return mi.mean()  # average across batch

    def forward(self, y_true, y_pred):
        return -self.mi(y_true, y_pred)


'''
局部互信息(LMI)损失函数, 一般用于多模态配准
'''
class localMutualInformation(torch.nn.Module):
    """
    Local Mutual Information for non-overlapping patches
    """

    def __init__(self, sigma_ratio=1, minval=0., maxval=1., num_bin=64, patch_size=9):
        super(localMutualInformation, self).__init__()

        """Create bin centers"""
        bin_centers = np.linspace(minval, maxval, num=num_bin)
        vol_bin_centers = Variable(torch.linspace(minval, maxval, num_bin), requires_grad=False).cuda()
        num_bins = len(bin_centers)

        """Sigma for Gaussian approx."""
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio

        self.preterm = 1 / (2 * sigma ** 2)
        self.bin_centers = bin_centers
        self.max_clip = maxval
        self.num_bins = num_bins
        self.vol_bin_centers = vol_bin_centers
        self.patch_size = patch_size

    def local_mi(self, y_true, y_pred):
        y_pred = torch.clamp(y_pred, 0., self.max_clip)
        y_true = torch.clamp(y_true, 0, self.max_clip)

        """Reshape bin centers"""
        o = [1, 1, np.prod(self.vol_bin_centers.shape)]
        vbc = torch.reshape(self.vol_bin_centers, o).cuda()

        """Making image paddings"""
        if len(list(y_pred.size())[2:]) == 3:
            ndim = 3
            x, y, z = list(y_pred.size())[2:]
            # compute padding sizes
            x_r = -x % self.patch_size
            y_r = -y % self.patch_size
            z_r = -z % self.patch_size
            padding = (z_r // 2, z_r - z_r // 2, y_r // 2, y_r - y_r // 2, x_r // 2, x_r - x_r // 2, 0, 0, 0, 0)
        elif len(list(y_pred.size())[2:]) == 2:
            ndim = 2
            x, y = list(y_pred.size())[2:]
            # compute padding sizes
            x_r = -x % self.patch_size
            y_r = -y % self.patch_size
            padding = (y_r // 2, y_r - y_r // 2, x_r // 2, x_r - x_r // 2, 0, 0, 0, 0)
        else:
            raise Exception('Supports 2D and 3D but not {}'.format(list(y_pred.size())))
        y_true = F.pad(y_true, padding, "constant", 0)
        y_pred = F.pad(y_pred, padding, "constant", 0)

        """Reshaping images into non-overlapping patches"""
        if ndim == 3:
            y_true_patch = torch.reshape(y_true, (y_true.shape[0], y_true.shape[1],
                                                  (x + x_r) // self.patch_size, self.patch_size,
                                                  (y + y_r) // self.patch_size, self.patch_size,
                                                  (z + z_r) // self.patch_size, self.patch_size))
            y_true_patch = y_true_patch.permute(0, 1, 2, 4, 6, 3, 5, 7)
            y_true_patch = torch.reshape(y_true_patch, (-1, self.patch_size ** 3, 1))

            y_pred_patch = torch.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1],
                                                  (x + x_r) // self.patch_size, self.patch_size,
                                                  (y + y_r) // self.patch_size, self.patch_size,
                                                  (z + z_r) // self.patch_size, self.patch_size))
            y_pred_patch = y_pred_patch.permute(0, 1, 2, 4, 6, 3, 5, 7)
            y_pred_patch = torch.reshape(y_pred_patch, (-1, self.patch_size ** 3, 1))
        else:
            y_true_patch = torch.reshape(y_true, (y_true.shape[0], y_true.shape[1],
                                                  (x + x_r) // self.patch_size, self.patch_size,
                                                  (y + y_r) // self.patch_size, self.patch_size))
            y_true_patch = y_true_patch.permute(0, 1, 2, 4, 3, 5)
            y_true_patch = torch.reshape(y_true_patch, (-1, self.patch_size ** 2, 1))

            y_pred_patch = torch.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1],
                                                  (x + x_r) // self.patch_size, self.patch_size,
                                                  (y + y_r) // self.patch_size, self.patch_size))
            y_pred_patch = y_pred_patch.permute(0, 1, 2, 4, 3, 5)
            y_pred_patch = torch.reshape(y_pred_patch, (-1, self.patch_size ** 2, 1))

        """Compute MI"""
        I_a_patch = torch.exp(- self.preterm * torch.square(y_true_patch - vbc))
        I_a_patch = I_a_patch / torch.sum(I_a_patch, dim=-1, keepdim=True)

        I_b_patch = torch.exp(- self.preterm * torch.square(y_pred_patch - vbc))
        I_b_patch = I_b_patch / torch.sum(I_b_patch, dim=-1, keepdim=True)

        pab = torch.bmm(I_a_patch.permute(0, 2, 1), I_b_patch)
        pab = pab / self.patch_size ** ndim
        pa = torch.mean(I_a_patch, dim=1, keepdim=True)
        pb = torch.mean(I_b_patch, dim=1, keepdim=True)

        papb = torch.bmm(pa.permute(0, 2, 1), pb) + 1e-6
        mi = torch.sum(torch.sum(pab * torch.log(pab / papb + 1e-6), dim=1), dim=1)
        return mi.mean()

    def forward(self, y_true, y_pred):
        return -self.local_mi(y_true, y_pred)

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

def compute_dice(s1, s2,dim=5):
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




def compute_patch_dice_coefficient(mask_gt, mask_pred):
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
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2*volume_intersect, volume_sum


def compute_patch_dice(patches_s1,patcdes_s2,patch_num):
    count = 0
    volume_intersect = [0]*35
    volume_sum = [0]*35
    Dice = 0
    for i in range(patch_num):
        s1 = patches_s1[i]
        s2 = patcdes_s2[i]
        for label in range(1, 36):
            dice = compute_patch_dice_coefficient((s1==label), (s2==label))
            volume_intersect[label-1] += dice[0]
            volume_sum[label-1] += dice[1]
    for i in range(35):
        if volume_sum[i] != 0:
            Dice = Dice + volume_intersect[i]/volume_sum[i]
            count = count + 1
    Dice /= count
    return Dice

def compute_average_surface_distance(surface_distances):
  """Returns the average surface distance.
  Computes the average surface distances by correctly taking the area of each
  surface element into account. Call compute_surface_distances(...) before, to
  obtain the `surface_distances` dict.
  Args:
    surface_distances: dict with "distances_gt_to_pred", "distances_pred_to_gt"
    "surfel_areas_gt", "surfel_areas_pred" created by
    compute_surface_distances()
  Returns:
    A tuple with two float values:
      - the average distance (in mm) from the ground truth surface to the
        predicted surface
      - the average distance from the predicted surface to the ground truth
        surface.
  """
  distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
  distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
  surfel_areas_gt = surface_distances["surfel_areas_gt"]
  surfel_areas_pred = surface_distances["surfel_areas_pred"]
  average_distance = (np.sum(distances_gt_to_pred * surfel_areas_gt) + np.sum(distances_pred_to_gt * surfel_areas_pred))/ ((np.sum(surfel_areas_gt))+np.sum(surfel_areas_pred))
  return average_distance


def compute_ASSD(mask_gt, mask_pred,dim=2):
    mask_pred = mask_pred[0,0,:,:,:]
    mask_gt = mask_gt[0,0,:,:,:]
    mask_pred=mask_pred.numpy()
    mask_gt = mask_gt.numpy()
    surface_distances_all =[]
    avg_surf_dist_all = []
    ASSD = 0
    num = 0

    # liver
    # value = [1.0]

    # IXI
    # value = [2.0, 3.0, 4.0, 5.0, 7.0, 8.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 24.0, 26.0, 28.0, 30.0, 31.0, 41.0, 42.0, 43.0, 44.0, 46.0, 47.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 58.0, 60.0, 62.0, 63.0, 77.0, 85.0, 251.0, 252.0, 253.0, 254.0, 255.0]

    # Mind101
    # value = [2.0, 4.0, 5.0, 7.0, 8.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 24.0, 26.0, 28.0, 30.0, 31.0, 41.0, 43.0, 44.0, 46.0, 47.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 58.0, 60.0, 62.0, 63.0, 77.0, 80.0, 85.0, 251.0, 252.0, 253.0, 254.0, 255.0]

    # IBSR18
    # value = [2.0, 8.0, 9.0, 24.0, 30.0]
    
    # CUMC12
    # value = [97,53,35,57,67,25,15,29,12]
    # LPBA40
    # value = [21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 101.0, 102.0, 121.0, 122.0, 161.0, 162.0, 163.0, 164.0, 165.0, 166.0, 181.0, 182.0]
    # MGH10
    # value = [4.0, 6.0, 7.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 26.0, 28.0, 29.0, 30.0, 31.0, 33.0, 35.0, 36.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 55.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 104.0, 106.0, 107.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0, 122.0, 126.0, 128.0, 129.0, 130.0, 131.0, 133.0, 135.0, 136.0, 138.0, 139.0, 140.0, 141.0, 142.0, 143.0, 144.0, 145.0, 146.0, 147.0, 148.0, 149.0, 150.0, 151.0, 152.0, 155.0, 158.0, 159.0, 160.0, 161.0, 162.0, 163.0, 164.0, 165.0, 166.0, 167.0, 168.0, 169.0]
    # value = [4,5,6,10,35]
    # for label in value:
    for label in range(1,dim):
        mask_pred_singel = np.where(mask_pred==label,True,False)
        mask_gt_singel = np.where(mask_gt==label,True,False)
        # mask_pred_singel = np.where(mask_pred!=label,False)
        # mask_gt_singel = np.where(mask_gt!=label,False)
        # print(mask_gt_singel)
    # print(mask_gt.dtype)
        surface_distances = surfdist.compute_surface_distances(mask_gt_singel, mask_pred_singel, spacing_mm=(1.0, 1.0, 1.0))
        avg_surf_dist = compute_average_surface_distance(surface_distances)
        surface_distances_all.append(avg_surf_dist)
        avg_surf_dist_all.append(avg_surf_dist)
        # print(avg_surf_dist)      
        if avg_surf_dist < 100000:
            ASSD += avg_surf_dist
            num += 1
    ASSD = ASSD/num
    return ASSD

def BCE_loss(y_hat, y):
    y_hat = y_hat.view(-1,1)
    # y_hat:预测标签，已经过sigmoid/softmax处理 shape is (batch_size, 1)
    # y：真实标签（一般为0或1） shape is (batch_size)
    y_hat = torch.cat((1 - y_hat, y_hat), 1)  # 将二种情况的概率都列出，y_hat形状变为(batch_size, 2)
    # 按照y标定的真实标签，取出预测的概率，来计算损失
    return - torch.log(y_hat.gather(1, y.view(-1, 1).type(torch.long))).mean()

class Grad3d(torch.nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        super(Grad3d, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred, y_true):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad

def jacobian_determinant(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.
    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims
    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    # disp = disp.transpose(1, 2, 3, 0)
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    import pystrum.pynd.ndutils as nd
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # must be 2

        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]
    

class InfoNCELoss(torch.nn.Module):
    def __init__(self, temperature=0.1):
        """
        初始化 InfoNCE 对比损失函数。
        参数：
        - temperature: 控制相似性分布的温度参数 τ。
        """
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, q, n, x):
        """
        计算 InfoNCE 对比损失。
        参数：
        - q: Tensor，大小为 [1, 512]，查询向量。
        - n: Tensor，大小为 [80, 1, 512]，候选样本向量。
        - x: int，正样本的索引。

        返回：
        - loss: 计算出的 InfoNCE 损失。
        """
        # 归一化向量以计算余弦相似度
        x = x - 1
        q = F.normalize(q, dim=-1)  # [1, 512]
        n = F.normalize(n.squeeze(1), dim=-1)  # [80, 512]

        # 计算 q 和所有候选样本 n 的相似性分数
        logits = torch.matmul(q, n.T) / self.temperature  # [1, 80]

        # 创建标签，正样本位置为 x
        labels = torch.tensor([x], dtype=torch.long).to(logits.device)

        # 使用交叉熵损失计算 InfoNCE 损失
        loss = F.cross_entropy(logits, labels)

        return loss
    

class KoLeoLossModified(nn.Module):
    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def pairwise_NNs_inner(self, x):
        """
        Pairwise nearest neighbors for L2-normalized vectors.
        Uses Torch rather than Faiss to remain on GPU.
        """
        # 计算所有向量之间的点积
        dots = torch.mm(x, x.t())  # [80, 80]
        n = x.shape[0]
        # 将对角线元素设置为 -1，以避免自匹配
        dots.view(-1)[:: (n + 1)].fill_(-1)
        # 查找最大点积对应的索引（即最近邻）
        _, I = torch.max(dots, dim=1)
        return I

    def forward(self, m, n, x, eps=1e-8):
        """
        计算第 x 个向量与其最近邻之间的 L2 距离。
        参数：
        - m: Tensor，大小为 [1, 512]，待替换的向量。
        - n: Tensor，大小为 [80, 1, 512]，候选向量集合。
        - x: int，要替换的索引（0 <= x < 80）。
        """
        with torch.cuda.amp.autocast(enabled=False):
            # x = x - 1
            # 将 n[x] 替换为 m，并得到新的向量集合
            n_modified = n.clone()  # 复制 n，避免原始数据被修改
            n_modified[x] = m.unsqueeze(0)  # 替换第 x 个向量

            # 对 n_modified 进行 L2 归一化
            n_modified = n_modified.squeeze(1)  # [80, 512]
            n_modified = F.normalize(n_modified, eps=eps, p=2, dim=-1)

            # 查找最近邻索引
            I = self.pairwise_NNs_inner(n_modified)  # [80]

            # 计算第 x 个向量与其最近邻的 L2 距离
            distance = self.pdist(n_modified[x].unsqueeze(0), n_modified[I[x]].unsqueeze(0))

            # 计算损失：-log(距离 + eps)
            loss = -torch.log(distance + eps).mean()

        return loss
    

    
class KoLeoLoss(nn.Module):
    """Kozachenko-Leonenko entropic loss regularizer."""
    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def pairwise_NNs_inner(self, x):
        """
        计算每个向量的最近邻索引（排除自身）。
        """
        # 计算所有向量之间的点积
        dots = torch.mm(x, x.t())  # [80, 80]
        n = x.shape[0]

        # 将对角线元素设置为 -1，避免选择自身
        dots.view(-1)[:: (n + 1)].fill_(-1)

        # 找到每行的最大点积（即最近邻）对应的索引
        _, I = torch.max(dots, dim=1)
        return I

    def forward(self, n, eps=1e-8):
        """
        计算替换后的向量集合的 KoLeo 损失。
        参数：
        - m: Tensor，大小为 [1, 512]，待替换的向量。
        - n: Tensor，大小为 [80, 1, 512]，候选向量集合。
        - x: int，替换的索引（0 <= x < 80）。
        """
        # with torch.cuda.amp.autocast(enabled=False):
            # 将 n 的第 x 个向量替换为 m
        n_modified = n.clone()  # 复制 n，避免修改原始数据

        # 去掉中间的维度，使其变为 [80, 512]
        n_modified = n_modified.squeeze(1)

        # 对所有向量进行 L2 归一化
        n_modified = F.normalize(n_modified, eps=eps, p=2, dim=-1)

        # 找到每个向量的最近邻索引
        I = self.pairwise_NNs_inner(n_modified)  # [80]

        # 计算所有向量与其最近邻的 L2 距离
        distances = self.pdist(n_modified, n_modified[I])

        # 计算 KoLeo 损失：-log(距离 + eps) 的均值
        loss = -torch.log(distances + eps).mean()

        return loss



class KoLeoLossModified_CWD(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def pairwise_NNs_inner_cwd(self, x, eps=1e-8):
        """
        Find the nearest neighbor indices based on CWD (Channel-Wise Distillation) loss.
        Uses KL divergence to find the closest match for each vector.
        """
        batch_size, dim = x.size()
        kl_divergences = torch.zeros(batch_size, batch_size, device=x.device)  # KL matrix [batch_size, batch_size]

        # Normalize input vectors
        x = F.normalize(x, p=2, dim=-1)

        for i in range(batch_size):
            # Compute softmax probabilities for the i-th vector
            teacher_probs = F.softmax(x[i].unsqueeze(0) / self.temperature, dim=1)  # [1, dim]

            # Compute softmax probabilities for all other vectors
            student_probs = F.softmax(x / self.temperature, dim=1)  # [batch_size, dim]

            # Compute KL divergence between the i-th vector and all others
            kl_div = teacher_probs * (torch.log(teacher_probs + eps) - torch.log(student_probs + eps))
            kl_divergences[i] = kl_div.sum(dim=1)  # Sum over features for each pair

        # Find the index of the vector with the smallest KL divergence (excluding self-match)
        _, nearest_neighbors = torch.topk(-kl_divergences, k=2, dim=1)  # Negative to sort by minimum
        nearest_neighbors = nearest_neighbors[:, 1]  # Skip self-match (1st closest is always itself)

        return nearest_neighbors

    def forward(self, m, n, x, eps=1e-8):
        """
        Compute the KoLeo loss with CWD distance for the x-th vector.
        Parameters:
        - m: Tensor, shape [1, 512], the replacement vector.
        - n: Tensor, shape [80, 1, 512], the candidate vector set.
        - x: int, the index to replace (0 <= x < 80).
        """
        with torch.cuda.amp.autocast(enabled=False):
            # Replace the x-th vector in n with m
            n_modified = n.clone()
            n_modified[x] = m.unsqueeze(0)

            # Remove middle dimension and normalize vectors
            n_modified = n_modified.squeeze(1)  # [80, 512]
            n_modified = F.normalize(n_modified, eps=eps, p=2, dim=-1)

            # Find nearest neighbor indices using CWD
            I = self.pairwise_NNs_inner_cwd(n_modified, eps=eps)  # [80]

            # Compute the CWD loss between the x-th vector and its nearest neighbor
            teacher_vector = n_modified[x].unsqueeze(0)  # [1, 512]
            student_vector = n_modified[I[x]].unsqueeze(0)  # [1, 512]

            # Compute probabilities using softmax
            teacher_probs = F.softmax(teacher_vector / self.temperature, dim=1)
            student_probs = F.softmax(student_vector / self.temperature, dim=1)

            # Compute KL divergence
            kl_div = teacher_probs * (torch.log(teacher_probs + eps) - torch.log(student_probs + eps))
            loss = (self.temperature ** 2) * kl_div.sum(dim=1).mean()  # Final CWD loss

        return loss


import torch
import torch.nn as nn
import torch.nn.functional as F

class KoLeoLoss_CWD(nn.Module):
    """Kozachenko-Leonenko entropic loss regularizer with modified CWD."""
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def pairwise_NNs_inner_cwd(self, x, eps=1e-8):
        """
        Compute the nearest neighbor indices based on CWD (Channel-Wise Distillation) loss.
        Uses KL divergence to find the closest match for each vector.
        """
        batch_size, dim = x.size()
        kl_divergences = torch.zeros(batch_size, batch_size, device=x.device)  # KL matrix [batch_size, batch_size]

        # Normalize input vectors
        x = F.normalize(x, p=2, dim=-1)

        for i in range(batch_size):
            # Compute softmax probabilities for the i-th vector
            teacher_probs = F.softmax(x[i].unsqueeze(0) / self.temperature, dim=1)  # [1, dim]

            # Compute softmax probabilities for all other vectors
            student_probs = F.softmax(x / self.temperature, dim=1)  # [batch_size, dim]

            # Compute KL divergence between the i-th vector and all others
            kl_div = teacher_probs * (torch.log(teacher_probs + eps) - torch.log(student_probs + eps))
            kl_divergences[i] = kl_div.sum(dim=1)  # Sum over features for each pair

        # Find the index of the vector with the smallest KL divergence (excluding self-match)
        _, nearest_neighbors = torch.topk(-kl_divergences, k=2, dim=1)  # Negative to sort by minimum
        nearest_neighbors = nearest_neighbors[:, 1]  # Skip self-match (1st closest is always itself)

        return nearest_neighbors

    def forward(self, n, eps=1e-8):
        """
        Compute the KoLeo loss with CWD for the entire modified vector set.
        Parameters:
        - n: Tensor, shape [80, 1, 512], the candidate vector set.
        """
        # Clone and normalize vectors
        n_modified = n.clone()
        n_modified = n_modified.squeeze(1)  # [80, 512]
        n_modified = F.normalize(n_modified, eps=eps, p=2, dim=-1)

        # Find nearest neighbor indices using CWD
        I = self.pairwise_NNs_inner_cwd(n_modified, eps=eps)  # [80]

        # Compute CWD loss for all vectors
        losses = []
        for idx in range(n_modified.shape[0]):
            teacher_vector = n_modified[idx].unsqueeze(0)  # [1, 512]
            student_vector = n_modified[I[idx]].unsqueeze(0)  # [1, 512]

            # Compute probabilities using softmax
            teacher_probs = F.softmax(teacher_vector / self.temperature, dim=1)
            student_probs = F.softmax(student_vector / self.temperature, dim=1)

            # Compute KL divergence (CWD distance)
            kl_div = teacher_probs * (torch.log(teacher_probs + eps) - torch.log(student_probs + eps))
            distance = kl_div.sum(dim=1)  # Sum over features

            # Apply -log(distance) to enforce non-zero loss when distance is 0
            losses.append(-torch.log(distance + eps))

        # Compute mean CWD loss
        return torch.stack(losses).mean()

    
def top3_cosine_similarity_indices(m, n):
    """
    计算向量 m 和 n 的每一维的余弦相似度，并输出排名前三的索引。
    参数:
    - m: Tensor，大小为 [1, 512]。
    - n: Tensor，大小为 [80, 1, 512]。

    返回:
    - top3_indices: 排名前三的索引列表。
    """
    # 对 m 和 n 进行 L2 归一化，以确保计算余弦相似度
    m_normalized = F.normalize(m, p=2, dim=-1)  # [1, 512]
    n_normalized = F.normalize(n.squeeze(1), p=2, dim=-1)  # [80, 512]

    # 计算 m 和 n 的每一维的余弦相似度
    cosine_similarities = torch.matmul(m_normalized, n_normalized.T).squeeze(0)  # [80]

    # 对余弦相似度进行归一化
    normalized_similarities = (cosine_similarities - cosine_similarities.min()) / \
                              (cosine_similarities.max() - cosine_similarities.min())

    # 对归一化的余弦相似度进行排序，并获取排名前三的索引
    top3_indices = torch.topk(normalized_similarities, 3).indices.tolist()

    return top3_indices

def cwd_loss_1d(student_vector, teacher_vector, temperature=1.0):
    """
    Compute Channel-Wise Distillation Loss (CWD Loss) for 1D tensors.
    :param student_vector: Tensor of shape [1, 768] from student model
    :param teacher_vector: Tensor of shape [1, 768] from teacher model
    :param temperature: Temperature scaling factor
    :return: Scalar CWD loss value
    """
    # Ensure input tensors have the correct shape
    assert student_vector.shape == teacher_vector.shape, "Student and teacher vectors must have the same shape"
    assert len(student_vector.shape) == 2 and student_vector.shape[0] == 1, "Input must have shape [1, 768]"
    
    # Apply temperature scaling
    student_vector = student_vector / temperature
    teacher_vector = teacher_vector / temperature

    # Normalize the vectors to probabilities using softmax
    student_probs = F.softmax(student_vector, dim=1)  # Shape: [1, 768]
    teacher_probs = F.softmax(teacher_vector, dim=1)  # Shape: [1, 768]

    # Compute the KL divergence for each element
    kl_div = teacher_probs * (torch.log(teacher_probs + 1e-8) - torch.log(student_probs + 1e-8))
    loss = kl_div.sum()  # Sum over all elements in the vector

    # Scale the loss by temperature^2
    return (temperature ** 2) * loss


class MINDLoss:
    def __init__(self, patch_size=3, neigh_size=6, sigma=0.5, eps=1e-6):
        self.patch_size = patch_size
        self.neigh_size = neigh_size
        self.sigma = sigma
        self.eps = eps


    def gaussian_kernel(self, sigma, sz):
        xpos_vec = np.arange(sz)
        ypos_vec = np.arange(sz)
        output = np.ones([1, 1, sz, sz], dtype=np.single)
        midpos = sz // 2
        for xpos in xpos_vec:
            for ypos in ypos_vec:
                output[:,:,xpos,ypos] = np.exp(-((xpos-midpos)**2 + (ypos-midpos)**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
        return torch.tensor(output)

    def torch_image_translate(self, input_, tx, ty, interpolation='nearest'):
        translation_matrix = torch.zeros([1, 3, 3], dtype=torch.float)
        translation_matrix[:, 0, 0] = 1.0
        translation_matrix[:, 1, 1] = 1.0
        translation_matrix[:, 0, 2] = -2 * tx / (input_.size()[2] - 1)
        translation_matrix[:, 1, 2] = -2 * ty / (input_.size()[3] - 1)
        translation_matrix[:, 2, 2] = 1.0
        grid = F.affine_grid(translation_matrix[:, 0:2, :], input_.size())
        wrp = F.grid_sample(input_, grid, mode=interpolation)
        return wrp

    def Dp(self, image, xshift, yshift):
        shift_image = self.torch_image_translate(image, xshift, yshift, interpolation='nearest')
        diff = torch.sub(image, shift_image)
        diff_square = torch.mul(diff, diff)
        kernel = self.gaussian_kernel(self.sigma, self.patch_size)
        res = torch.conv2d(diff_square, weight=kernel, stride=1, padding=3)
        return res

    def MIND(self, image, image_size0, image_size1):
        reduce_size = int((self.patch_size + self.neigh_size - 2) / 2)

        # Estimate local variance of each pixel within the input image
        Vimg = torch.add(self.Dp(image, -1, 0), self.Dp(image, 1, 0))
        Vimg = torch.add(Vimg, self.Dp(image, 0, -1))
        Vimg = torch.add(Vimg, self.Dp(image, 0, 1))
        Vimg = torch.div(Vimg, 4) + torch.mul(torch.ones_like(Vimg), self.eps)

        # Estimate the (R*R)-length MIND feature by shifting the input image by R*R times
        xshift_vec = np.arange(-(self.neigh_size // 2), self.neigh_size - (self.neigh_size // 2))
        yshift_vec = np.arange(-(self.neigh_size // 2), self.neigh_size - (self.neigh_size // 2))
        iter_pos = 0
        for xshift in xshift_vec:
            for yshift in yshift_vec:
                if (xshift, yshift) == (0, 0):
                    continue
                MIND_tmp = torch.exp(torch.mul(torch.div(self.Dp(image, xshift, yshift), Vimg), -1))
                tmp = MIND_tmp[:, :, reduce_size:(image_size0 - reduce_size), reduce_size:(image_size1 - reduce_size)]
                if iter_pos == 0:
                    output = tmp
                else:
                    output = torch.cat([output, tmp], 1)
                iter_pos += 1

        # Normalize output
        input_max, input_indexes = torch.max(output, dim=1)
        output = torch.div(output, input_max)

        return output

    def loss(self, input_image, target_image):
        # Get the image size
        image_size0, image_size1 = input_image.size(2), input_image.size(3)

        # Compute MIND descriptors for both images
        mind_input = self.MIND(input_image, image_size0, image_size1)
        mind_target = self.MIND(target_image, image_size0, image_size1)

        # Calculate the MIND loss as the mean squared error between the MIND descriptors of both images
        loss = torch.mean((mind_input - mind_target) ** 2)

        return loss