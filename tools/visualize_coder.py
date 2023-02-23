import numpy as np
import torch
import cv2
import pickle

from vis_util import project_pts_on_img, our_draw_lidar_bbox3d_on_img, draw_bboxes_2d, draw_camera_bbox3d_on_img
from mmdet3d.core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes, CameraInstance3DBoxes
from mmdet3d.core.bbox.coders import CameraBBoxCoder

cam_intrinsic = np.array([
    [1.27259795e+03, 0.00000000e+00, 8.26615493e+02],
    [0.00000000e+00, 1.27259795e+03, 4.79751654e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])

angle = 60
pred_score = torch.Tensor([0.9]).reshape(1, 1, 1)
pred_loc = torch.Tensor([0, 0, 4]).reshape(1, 3, 1)
pred_dims = torch.Tensor([np.log(2.0), np.log(1.0), np.log(3.0)]).reshape(1, 3, 1)
pred_rot = torch.Tensor([np.sin(angle), np.cos(angle)]).reshape(1, 2, 1)
pred_vel = torch.Tensor([5, 5]).reshape(1, 2, 1)

bbox_coder = CameraBBoxCoder(code_size=10)

pred_dims_exp = torch.exp(pred_dims)
pred_rot_angle = torch.atan2(pred_rot[:, 0:1, :], pred_rot[:, 1:2, :])

cam_bbox = torch.cat([pred_loc, pred_dims_exp, pred_rot_angle, pred_vel], dim=1)
cam_bbox = CameraInstance3DBoxes(cam_bbox[..., 0], box_dim=9, origin=(0.5,0.5,0.5))

decode_bbox = bbox_coder.decode(pred_score, pred_rot, pred_dims, pred_loc, pred_vel)

image = np.zeros((800, 1600, 3), dtype=np.uint8) + 255

center = cam_intrinsic @ pred_loc[0].numpy()[:, 0]
center[:2] = center[:2] / center[2]

image = cv2.circle(image, center=(int(center[0]), int(center[1])), radius=3, color=(0, 255, 0), thickness=3)
image = draw_camera_bbox3d_on_img(cam_bbox, image, cam_intrinsic, color=(255, 0, 0))

cv2.imwrite("vis/images/test_image.png", image)



