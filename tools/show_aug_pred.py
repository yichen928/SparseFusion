import numpy as np
import torch
import cv2
import pickle

from vis_util import project_pts_on_img, our_draw_lidar_bbox3d_on_img, draw_bboxes_2d, draw_camera_bbox3d_on_img
from mmdet3d.core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes, CameraInstance3DBoxes


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
gt_bbox_cam_3d_tensor = torch.load("vis/gt_bboxes_cam_3d.pt")

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

for bid in range(bs):
    lidar_bbox_3d = gt_bbox_3d_tensor[bid]
    img_labels_2d = gt_label_2d_np[bid]
    img_bbox_2d = gt_bbox_2d_np[bid]
    img_centers_2d = gt_centers_2d_np[bid]
    cam_bbox_3d = gt_bbox_cam_3d_tensor[bid]

    pos_inds_vt_i = torch.load("vis/pos_view_inds_%d.pt"%bid, map_location='cpu')
    pred_bbox_vt_i = torch.load("vis/bboxes_view_tensor_%d.pt"%bid, map_location='cpu')

    pos_inds_i = torch.load("vis/pos_inds_%d.pt"%bid, map_location='cpu')
    pred_bbox_i = torch.load("vis/bboxes_tensor_%d.pt"%bid, map_location='cpu')

    pts_pos_inds_i = pos_inds_i[pos_inds_i < 200]
    fuse_pos_inds_i = pos_inds_i[pos_inds_i >= 200] - 200

    pts_pred_bbox_i = pred_bbox_i[:200]
    fuse_pred_bbox_i = pred_bbox_i[200:]

    # pts_pred_bbox_pos = pts_pred_bbox_i[pts_pos_inds_i]
    pts_pred_bbox_pos = pts_pred_bbox_i[:100 ]
    pts_pred_bbox_pos = LiDARInstance3DBoxes(pts_pred_bbox_pos, box_dim=pts_pred_bbox_pos.shape[-1], origin=(0.5, 0.5, 0.0)).convert_to(Box3DMode.LIDAR)

    fuse_pred_bbox_pos = fuse_pred_bbox_i[fuse_pos_inds_i]
    fuse_pred_bbox_pos = LiDARInstance3DBoxes(fuse_pred_bbox_pos, box_dim=fuse_pred_bbox_pos.shape[-1], origin=(0.5, 0.5, 0.0)).convert_to(Box3DMode.LIDAR)

    pred_bbox_vt_pos = pred_bbox_vt_i[pos_inds_vt_i]
    pred_bbox_vt_pos = LiDARInstance3DBoxes(pred_bbox_vt_pos, box_dim=pred_bbox_vt_pos.shape[-1], origin=(0.5, 0.5, 0.0)).convert_to(Box3DMode.LIDAR)

    for view_id in range(view_num):

        view_img = imgs[bid, view_id]
        view_img = cv2.cvtColor(view_img, cv2.COLOR_RGB2BGR)

        inds_2d_view = img_labels_2d[:, 1] == view_id
        gt_bbox_2d_view = img_bbox_2d[inds_2d_view]
        gt_center_2d_view = img_centers_2d[inds_2d_view]
        cam_bbox_3d_view = cam_bbox_3d[inds_2d_view]

        img_scale_factor = img_metas[bid]['scale_factor'][:2]
        # img_scale = np.array([img_scale_factor[0], img_scale_factor[1], img_scale_factor[0], img_scale_factor[1]])
        lidar2cam_r = img_metas[bid]['lidar2cam_r'][view_id]
        lidar2cam_t = img_metas[bid]['lidar2cam_t'][view_id]
        lidar2cam_rt = np.eye(4)
        lidar2cam_rt[:3, :3] = lidar2cam_r
        lidar2cam_rt[:3, 3] = lidar2cam_t
        intrinsic = img_metas[bid]['cam_intrinsic'][view_id]
        viewpad = np.eye(4)
        viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
        lidar2img_rt = viewpad @ lidar2cam_rt

        # if pred_bbox_vt_pos.tensor.shape[0] > 0:
        #     view_img = our_draw_lidar_bbox3d_on_img(pred_bbox_vt_pos, view_img, lidar2img_rt, img_metas[bid], view_id=view_id, color=(0, 255, 255))

        # if cam_bbox_3d_view.tensor.shape[0]>0:
        #     view_img = draw_camera_bbox3d_on_img(cam_bbox_3d_view, view_img, intrinsic, color=(255, 0, 0))

        # if lidar_bbox_3d.tensor.shape[0] > 0:
        #     view_img = our_draw_lidar_bbox3d_on_img(lidar_bbox_3d, view_img, lidar2img_rt, img_metas[bid], view_id=view_id, color=(255, 0, 0))

        # if fuse_pred_bbox_pos.tensor.shape[0] > 0:
        #     view_img = our_draw_lidar_bbox3d_on_img(fuse_pred_bbox_pos, view_img, lidar2img_rt, img_metas[bid], view_id=view_id, color=(0, 0, 255))

        if pts_pred_bbox_pos.tensor.shape[0] > 0:
            view_img = our_draw_lidar_bbox3d_on_img(pts_pred_bbox_pos, view_img, lidar2img_rt, img_metas[bid], view_id=view_id, color=(255, 255, 0))

        if gt_bbox_2d_view.shape[0] > 0:
            gt_bbox_2d_view = gt_bbox_2d_view
            for i in range(gt_bbox_2d_view.shape[0]):
                cv2.circle(
                    view_img,
                    center=(int(np.round(gt_center_2d_view[i, 0])), int(np.round(gt_center_2d_view[i, 1]))),
                    radius=1,
                    color=(0, 255, 0),
                    thickness=3,
                )

        cv2.imwrite("vis/images/img_b%d_v%d.png"%(bid, view_id), view_img)

