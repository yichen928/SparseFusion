import numpy as np
import torch
import cv2
import pickle

from vis_util import project_pts_on_img, our_draw_lidar_bbox3d_on_img, draw_bboxes_2d, draw_camera_bbox3d_on_img

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("agg")


depth_mean = 14.41
depth_var = 156.89

scales = [4]

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

# img_tensor = torch.load("vis/img.pt")
# point_tensor = torch.load("vis/points.pt")
# gt_depth_tensor = torch.load("vis/sparse_depth.pt")

img_tensor = torch.load("vis/img.pt")
gt_depth_tensor = torch.load("vis/sparse_depth.pt")

bs = img_tensor.shape[0]
view_num = img_tensor.shape[1]
unnormal = UnNormalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])

for i in range(bs):
    for j in range(view_num):
        img_tensor[i, j] = unnormal(img_tensor[i,j])

img_tensor = img_tensor.permute(0, 1, 3, 4, 2)
imgs = img_tensor.detach().cpu().numpy()

gt_depth = gt_depth_tensor.detach().cpu().numpy()
gt_depth[:, :, :, 0] = gt_depth[:, :, :, 0] * np.sqrt(depth_var) + depth_mean
gt_depth[:, :, :, 0] = gt_depth[:, :, :, 1] * gt_depth[:, :, :, 0]

for bid in range(bs):
    for view_id in range(view_num):
        gt_depth_view = gt_depth[bid, view_id]
        view_img = imgs[bid, view_id]
        view_img = cv2.cvtColor(view_img, cv2.COLOR_RGB2BGR)

        # img_scale_factor = img_metas[bid]['scale_factor'][:2]
        # lidar2cam_r = img_metas[bid]['lidar2cam_r'][view_id]
        # lidar2cam_t = img_metas[bid]['lidar2cam_t'][view_id]
        # lidar2cam_rt = np.eye(4)
        # lidar2cam_rt[:3, :3] = lidar2cam_r
        # lidar2cam_rt[:3, 3] = lidar2cam_t
        # intrinsic = img_metas[bid]['cam_intrinsic'][view_id]
        # viewpad = np.eye(4)
        # viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
        # lidar2img_rt = viewpad @ lidar2cam_rt

        cv2.imwrite("vis/images/aug_b%d_v%d_img.png" % (bid, view_id), view_img)
        for scale_id, scale in enumerate(scales):
            plt.imshow(gt_depth_view[scale_id, 0], cmap='YlGnBu', interpolation='nearest')
            plt.savefig("vis/images/aug_b%d_v%d_s%d.png"%(bid, view_id, scale_id), bbox='tight', pad_inches = 0)
            plt.show()
            plt.close()

