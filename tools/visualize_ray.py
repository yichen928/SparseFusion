import torch
import numpy as np
import os
import pickle
from vis_util import project_pts_on_img, draw_lidar_bbox3d_on_img, draw_bboxes_2d
import cv2
import matplotlib

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def bev_to_lidar(bev_xy):
    assert bev_xy.shape[-1] == 2
    bev_xy[..., 0] = bev_xy[..., 0] / grid_x * (pc_range[3] - pc_range[0]) + pc_range[0]
    bev_xy[..., 1] = bev_xy[..., 1] / grid_y * (pc_range[4] - pc_range[1]) + pc_range[1]
    return bev_xy

grid_x = 180
grid_y = 180
pc_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]

img_tensor = torch.load("vis/img.pt")
point_tensor = torch.load("vis/points.pt")
gt_bbox_2d_tensor = torch.load("vis/gt_bboxes_2d.pt")
gt_label_2d_tensor = torch.load("vis/gt_labels_2d.pt")
gt_bbox_3d_tensor = torch.load("vis/gt_bboxes_3d.pt")
gt_label_3d_tensor = torch.load("vis/gt_labels_3d.pt")
img_query_view_3d_tensor = torch.load("vis/img_query_view_3d.pt")
img_query_pos_3d_tensor = torch.load("vis/img_query_pos_3d.pt")

with open("vis/img_metas.pkl", "rb") as file:
    img_metas = pickle.load(file, encoding="bytes")

bs = img_tensor.shape[0]
num_view = img_tensor.shape[1]
unnormal = UnNormalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
for i in range(bs):
    for j in range(num_view):
        img_tensor[i, j] = unnormal(img_tensor[i,j])

img_tensor = img_tensor.permute(0, 1, 3, 4, 2)
img_np = img_tensor.detach().cpu().numpy()
img_query_pos_3d_np = img_query_pos_3d_tensor.detach().cpu().numpy()
img_query_view_3d_np = img_query_view_3d_tensor.detach().cpu().numpy()
gt_bbox_2d_np = []
gt_label_2d_np = []
for i in range(bs):
    gt_bbox_2d_np.append(gt_bbox_2d_tensor[i].detach().cpu().numpy())
    gt_label_2d_np.append(gt_label_2d_tensor[i].detach().cpu().numpy())

pos_inds = []
for i in range(bs):
    sample_pos_inds_2d_tensor = torch.load("vis/pos_inds_%d.pt"%i)
    sample_pos_inds_2d_np = sample_pos_inds_2d_tensor.detach().cpu().numpy()
    pos_inds.append(sample_pos_inds_2d_np)

for sample_idx in range(bs):
    lidar_bbox_3d = gt_bbox_3d_tensor[sample_idx]
    img_labels_2d = gt_label_2d_np[sample_idx]
    for view_idx in range(num_view):
        sample_pos_ind = pos_inds[sample_idx]
        pos_mask = np.zeros(img_query_view_3d_np.shape[1], dtype=np.bool)
        pos_mask[sample_pos_ind] = True
        on_the_image = np.logical_and(img_query_view_3d_np[sample_idx] == view_idx, pos_mask)
        img_view = img_np[sample_idx, view_idx]  # [H, W, 3]
        pos_num_view = np.sum(on_the_image)
        img_scale_factor = img_metas[sample_idx]['scale_factor'][:2]

        img_view = cv2.cvtColor(img_view, cv2.COLOR_RGB2BGR)
        img_view = cv2.resize(img_view, (0, 0), fx=1/img_scale_factor[0], fy=1/img_scale_factor[1])

        img_view = cv2.putText(img_view, '%d'%pos_num_view, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 3, cv2.LINE_AA)
        query_pos_3d_view = img_query_pos_3d_np[sample_idx, on_the_image]  # [on_the_image, 96]
        query_pos_3d_view = query_pos_3d_view.reshape(-1, 32, 3)
        query_pos_3d_view[..., :2] = bev_to_lidar(query_pos_3d_view[..., :2])
        all_query_pos_3d_view = query_pos_3d_view.reshape(-1, 3)

        new_img = project_pts_on_img(all_query_pos_3d_view, img_view, img_metas[sample_idx]['lidar2img'][view_idx])

        new_img = draw_lidar_bbox3d_on_img(lidar_bbox_3d, new_img, img_metas[sample_idx]['lidar2img'][view_idx], img_metas)

        inds_2d_view = img_labels_2d[:, 1] == view_idx
        print(sample_idx, view_idx, img_labels_2d[inds_2d_view])
        gt_bbox_2d_view = gt_bbox_2d_np[sample_idx][inds_2d_view]

        img_scale = np.array([1/img_scale_factor[0], 1/img_scale_factor[1], 1/img_scale_factor[0], 1/img_scale_factor[1]])
        if gt_bbox_2d_view.shape[0] > 0:
            gt_bbox_2d_view = gt_bbox_2d_view * img_scale
            new_img = draw_bboxes_2d(gt_bbox_2d_view, new_img)

        cv2.imwrite("vis/img_ray_b%d_v%d.png"%(sample_idx, view_idx), new_img)
