import torch
import torch.nn.functional as F
import numpy as np
import math
import surface_distance as surfdist
import torchio as tio


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



def compute_dice_coefficient(mask_gt, mask_pred):
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

def compute_dice(s1, s2):
    dice = 0
    count = 0
    for label in range(1, 36):
        dice += compute_dice_coefficient((s1==label), (s2==label))
        count += 1
    dice /= count
    return dice


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


def compute_ASSD(mask_gt, mask_pred,dim=36):
    mask_pred = mask_pred[0,0,:,:,:]
    mask_gt = mask_gt[0,0,:,:,:]
    mask_pred=mask_pred.numpy()
    mask_gt = mask_gt.numpy()
    surface_distances_all =[]
    avg_surf_dist_all = []
    ASSD = 0
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
        ASSD += avg_surf_dist
    ASSD = ASSD/(dim-1)
    return ASSD


# mask_gt_path = "/home/boys/Datasets/OASIS_OAS1_0176_MR1/seg35.nii.gz"
# # mask_pred_path = "/home/boys/Datasets/OASIS_OAS1_0176_MR1/seg35.nii.gz"
# mask_pred_path = "/home/boys/Datasets/test/OASIS_OAS1_0026_MR1/seg35.nii.gz"
# mask_gt = tio.ScalarImage(mask_gt_path)
# mask_gt = mask_gt.data
# mask_gt = torch.unsqueeze(mask_gt,0)
# mask_pred = tio.ScalarImage(mask_pred_path)
# mask_pred = mask_pred.data
# mask_pred = torch.unsqueeze(mask_pred,0)
# # mask_pred = mask_pred[0,0,:,:,:]
# # mask_pred=mask_pred.numpy()
# # print(mask_pred)
# surfaces = compute_ASSD(mask_gt,mask_pred)
# print(surfaces)
# Flow = torch.rand(1,3,256,256,256)
# print(Flow.size())
# D = JacboianDet(Flow.permute(0, 3, 2, 4, 1))
# print(D.size())