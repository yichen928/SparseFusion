import numpy as np
import torch
import cv2
import pickle

from vis_util import project_pts_on_img, our_draw_lidar_bbox3d_on_img, draw_bboxes_2d


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

img_tensor = torch.load("vis/img.pt")
point_tensor = torch.load("vis/points.pt")
gt_bbox_2d_tensor = torch.load("vis/gt_bboxes_2d.pt")
gt_label_2d_tensor = torch.load("vis/gt_labels_2d.pt")
gt_bbox_3d_tensor = torch.load("vis/gt_bboxes_3d.pt")
gt_label_3d_tensor = torch.load("vis/gt_labels_3d.pt")
gt_img_centers_tensor = torch.load("vis/gt_img_centers_2d.pt")

img_query_center_2d = torch.load("vis/img_query_center_2d.pt")
img_query_center_depth = torch.load("vis/img_query_center_depth.pt")
img_query_offset = torch.load("vis/img_query_offset.pt")
img_query_view = torch.load("vis/img_query_view_3d.pt")

pos_inds = []
for i in range(img_query_offset.shape[0]):
    pos_ind = torch.load("vis/pos_inds_%d.pt"%i)
    pos_inds.append(pos_ind)

with open("vis/img_metas.pkl", "rb") as file:
    img_metas = pickle.load(file, encoding="bytes")


bs = img_tensor.shape[0]
view_num = img_tensor.shape[1]
unnormal = UnNormalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])

for i in range(bs):
    for j in range(view_num):
        img_tensor[i, j] = unnormal(img_tensor[i,j])

img_tensor = img_tensor.permute(0, 1, 3, 4, 2)
imgs = img_tensor.detach().cpu().numpy()
gt_bbox_2d_np = []
gt_label_2d_np = []
gt_centers_2d_np = []
for i in range(bs):
    gt_bbox_2d_np.append(gt_bbox_2d_tensor[i].detach().cpu().numpy())
    gt_label_2d_np.append(gt_label_2d_tensor[i].detach().cpu().numpy())
    gt_centers_2d_np.append(gt_img_centers_tensor[i].detach().cpu().numpy())


img_query_center_2d_np = []
img_query_offset_np = []
img_query_center_depth_np = []
img_query_view_np = []
pos_inds_np = []
for i in range(len(pos_inds)):
    img_query_center_2d_np.append(img_query_center_2d[i].detach().cpu().numpy())
    img_query_center_depth_np.append(img_query_center_depth[i].detach().cpu().numpy())
    img_query_offset_np.append(img_query_offset[i].detach().cpu().numpy())
    img_query_view_np.append(img_query_view[i].detach().cpu().numpy())
    pos_inds_np.append(pos_inds[i].detach().cpu().numpy())

img_h = imgs.shape[2]
img_w = imgs.shape[3]

for bid in range(bs):
    lidar_bbox_3d = gt_bbox_3d_tensor[bid]
    img_labels_2d = gt_label_2d_np[bid]
    img_bbox_2d = gt_bbox_2d_np[bid]
    img_centers_2d = gt_centers_2d_np[bid]

    pred_pos_inds = pos_inds_np[bid]
    pred_offset = img_query_offset_np[bid].transpose(1, 0)
    pred_center = img_query_center_2d_np[bid].transpose(1, 0)
    pred_center_depth = img_query_center_depth_np[bid].transpose(1, 0)
    pred_view = img_query_view_np[bid]

    pred_offset = pred_offset[pred_pos_inds]
    pred_center = pred_center[pred_pos_inds]
    pred_center_depth = pred_center_depth[pred_pos_inds]
    pred_view = pred_view[pred_pos_inds]

    pred_center[:, 0] = pred_center[:, 0] * img_w
    pred_center[:, 1] = pred_center[:, 1] * img_h
    pred_offset[:, 0] = pred_offset[:, 0] * img_w
    pred_offset[:, 1] = pred_offset[:, 1] * img_h
    pred_offset[:, 2] = pred_offset[:, 2] * img_w
    pred_offset[:, 3] = pred_offset[:, 3] * img_h
    pred_center_depth[:, 0] = pred_center_depth[:, 0] * img_w
    pred_center_depth[:, 1] = pred_center_depth[:, 1] * img_h

    x1 = pred_center[:, 0] - pred_offset[:, 0]
    y1 = pred_center[:, 1] - pred_offset[:, 1]
    x2 = pred_center[:, 0] + pred_offset[:, 2]
    y2 = pred_center[:, 1] + pred_offset[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    pred_bbox = np.vstack([cx, cy, width, height]).transpose(1, 0)

    for view_id in range(view_num):
        view_img = imgs[bid, view_id]
        view_img = cv2.cvtColor(view_img, cv2.COLOR_RGB2BGR)

        inds_2d_view = img_labels_2d[:, 1] == view_id
        gt_bbox_2d_view = img_bbox_2d[inds_2d_view]
        gt_center_2d_view = img_centers_2d[inds_2d_view]

        view_mask = pred_view == view_id
        pred_bbox_view = pred_bbox[view_mask]
        pred_center_view = pred_center[view_mask]
        pred_center_depth_view = pred_center_depth[view_mask]

        img_scale_factor = img_metas[bid]['scale_factor'][:2]

        if gt_bbox_2d_view.shape[0] > 0:
            gt_bbox_2d_view = gt_bbox_2d_view
            view_img = draw_bboxes_2d(gt_bbox_2d_view, view_img)
            view_img = draw_bboxes_2d(pred_bbox_view, view_img, color=(0, 255, 255))


            for i in range(gt_bbox_2d_view.shape[0]):
                cv2.circle(
                    view_img,
                    center=(int(np.round(gt_center_2d_view[i, 0])), int(np.round(gt_center_2d_view[i, 1]))),
                    radius=1,
                    color=(0, 255, 0),
                    thickness=3,
                )

                cv2.circle(
                    view_img,
                    center=(int(np.round(pred_center_view[i, 0])), int(np.round(pred_center_view[i, 1]))),
                    radius=1,
                    color=(255, 255, 0),
                    thickness=3,
                )

                cv2.circle(
                    view_img,
                    center=(int(np.round(pred_center_depth_view[i, 0])), int(np.round(pred_center_depth_view[i, 1]))),
                    radius=1,
                    color=(0, 0, 0),
                    thickness=3,
                )

        cv2.imwrite("vis/images/img_b%d_v%d.png"%(bid, view_id), view_img)