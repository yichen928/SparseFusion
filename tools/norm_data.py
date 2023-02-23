import json
import pickle
import cv2
import os
import numpy as np
from collections import Counter

from vis_util import draw_camera_bbox3d_on_img, draw_bboxes_2d
from mmdet3d.core.bbox import CameraInstance3DBoxes

import random

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

data_dir = "data/nuscenes"
cam_orders = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT']

with open(os.path.join(data_dir, "nuscenes_infos_train_with_proj2d_wbox3d_woRange.pkl"), "rb") as file:
    mono_data = pickle.load(file, encoding="bytes")


infos = mono_data['infos']

random.seed(0)
random.shuffle(infos)

means = []
vars = []

for i in range(len(infos)):
    if i % 100 == 1:
        print("%d/%d"%(i, len(infos)))

    cam_items = infos[i]['cams']

    view_2d = infos[i]['gt_views_2d']
    cam_box_3d = infos[i]['gt_bboxes_cam_3d']
    cam_velo = infos[i]['gt_bboxes_cam_vel']
    box_2d = infos[i]['gt_bboxes_2d']
    center_2d = infos[i]['gt_img_centers_2d']

    lidar_path = infos[i]['lidar_path']
    lidar_data = np.fromfile(lidar_path, dtype=np.float32)

    lidar_data = lidar_data.reshape(-1, 5)

    lidar_data = lidar_data[:, :3]
    lidar_data_4d = np.concatenate([lidar_data, np.ones_like(lidar_data[:, :1])], axis=1)

    for view_id, cam_name in enumerate(cam_orders):
        cam_item = cam_items[cam_name]
        filepath = cam_item['data_path']
        image = cv2.imread(filepath)

        lidar2cam_r = np.linalg.inv(cam_item['sensor2lidar_rotation'])
        lidar2cam_t = cam_item['sensor2lidar_translation'] @ lidar2cam_r.T
        lidar2cam_rt = np.eye(4)
        lidar2cam_rt[:3, :3] = lidar2cam_r
        lidar2cam_rt[:3, 3] = -lidar2cam_t

        cam_intrinsic = cam_item['cam_intrinsic']

        lidar_2d = lidar_data_4d @ lidar2cam_rt.T
        lidar_2d = lidar_2d[:, :3]
        lidar_2d = lidar_2d @ cam_intrinsic.T
        lidar_2d[:, 0] = lidar_2d[:, 0] / lidar_2d[:, 2]
        lidar_2d[:, 1] = lidar_2d[:, 1] / lidar_2d[:, 2]

        height = image.shape[0]
        width = image.shape[1]

        view_mask = (lidar_2d[:, 0] > 0) & (lidar_2d[:, 0] < width) & (lidar_2d[:, 1] > 0) & (lidar_2d[:, 1] < height) & (lidar_2d[:, 2] > 0)
        lidar_2d_view = lidar_2d[view_mask]

        means.append(np.mean(lidar_2d_view[:, 2]))
        vars.append(np.var(lidar_2d_view[:, 2]))

    if i % 100 == 0:
        print(np.mean(means), np.mean(vars))

