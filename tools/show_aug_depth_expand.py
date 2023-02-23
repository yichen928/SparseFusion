import numpy as np
import torch
import cv2
import pickle
import torch.nn.functional as F

from vis_util import project_pts_on_img, our_draw_lidar_bbox3d_on_img, draw_bboxes_2d, draw_camera_bbox3d_on_img

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("agg")


depth_mean = 14.41
depth_var = 156.89

scales = [4]
expand_size = 3

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
        gt_depth_view = gt_depth[bid, view_id, 0, 0]
        view_img = imgs[bid, view_id]
        view_img = cv2.cvtColor(view_img, cv2.COLOR_RGB2BGR)

        gt_depth_view = torch.from_numpy(gt_depth_view)
        zero_inds = gt_depth[bid, view_id, 0, 1] == 0
        gt_depth_view[zero_inds] = 9999

        # zero_inds = torch.from_numpy(zero_inds)
        #
        # gt_depth_view[zero_inds] = 9999
        #
        # gt_depth_view_min = -F.max_pool2d(-gt_depth_view[None], kernel_size=3, stride=1, padding=1)[0]
        # zero_inds_new = gt_depth_view_min == 9999
        # gt_depth_view_min[zero_inds_new] = 0
        # gt_depth_view = torch.where(zero_inds, gt_depth_view_min, gt_depth_view)
        # gt_depth_view = gt_depth_view.numpy()

        for i in range(3):
            gt_depth_view_new = np.zeros_like(gt_depth_view) + 9999
            gt_depth_view_new[1:] = np.minimum(gt_depth_view_new[1:], gt_depth_view[:-1])
            gt_depth_view_new[:-1] = np.minimum(gt_depth_view_new[:-1], gt_depth_view[1:])
            gt_depth_view_new[:, 1:] = np.minimum(gt_depth_view_new[:, 1:], gt_depth_view[:, :-1])
            gt_depth_view_new[:, :-1] = np.minimum(gt_depth_view_new[:, :-1], gt_depth_view[:, 1:])

            gt_depth_view = np.where(zero_inds, gt_depth_view_new, gt_depth_view)
            zero_inds = gt_depth_view == 9999

        gt_depth_view[zero_inds] = 0

        valid_depth = np.zeros_like(gt_depth_view)
        valid_depth[zero_inds] = 1
        cv2.imwrite("vis/depth_images/aug_b%d_v%d_img.png" % (bid, view_id), view_img)
        plt.imshow(gt_depth_view, cmap='YlGnBu', interpolation='nearest')
        plt.savefig("vis/depth_images/aug_b%d_v%d_exp3.png"%(bid, view_id), bbox='tight', pad_inches=0)
        plt.show()
        plt.close()

        plt.imshow(valid_depth, cmap='YlGnBu', interpolation='nearest')
        plt.savefig("vis/depth_images/aug_b%d_v%d_valid.png"%(bid, view_id), bbox='tight', pad_inches=0)
        plt.show()
        plt.close()