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

image_num = 10
data_dir = "data/nuscenes"
cam_orders = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT']

with open(os.path.join(data_dir, "nuscenes_infos_train_with_proj2d_wbox3d_woRange.pkl"), "rb") as file:
    mono_data = pickle.load(file, encoding="bytes")


infos = mono_data['infos']

random.seed(0)
random.shuffle(infos)

stride = 4

for i in range(image_num):
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

        view_mask = (lidar_2d[:, 0] > 0) & (lidar_2d[:, 0] < width-stride) & (lidar_2d[:, 1] > 0) & (lidar_2d[:, 1] < height-stride) & (lidar_2d[:, 2] > 0)
        lidar_2d_view = lidar_2d[view_mask]

        lidar_2d_view[:, :2] = lidar_2d_view[:, :2] / stride
        sort_id = np.argsort(-lidar_2d_view[:, 2])
        lidar_2d_view = lidar_2d_view[sort_id]

        image = cv2.resize(image, None, fx=1.0/stride, fy=1.0/stride)

        max_depth = np.percentile(lidar_2d_view[:, 2], 99)
        depth_map = np.full((image.shape[0], image.shape[1]), max_depth)

        cx = lidar_2d_view[:, 0].astype(np.int32)
        cy = lidar_2d_view[:, 1].astype(np.int32)

        depth_map[cy, cx] = lidar_2d_view[:, 2]
        plt.imshow(depth_map, cmap='YlGnBu', interpolation='nearest')
        plt.savefig("vis/images/b%d_v%d_depth.png"%(i, view_id))
        plt.show()
        plt.close()

        pos = cx * 1000 + cy
        # # print(np.max(np.bincount(pos)))
        # count = np.bincount(pos)
        # print(np.sum(count>1), np.sum(count>0))

        for id in range(lidar_2d_view.shape[0]):
            cx = lidar_2d_view[id, 0]
            cy = lidar_2d_view[id, 1]
            image = cv2.circle(image, (int(cx), int(cy)), 1, (255, 0, 0))
        cv2.imwrite("vis/images/b%d_v%d.png"%(i, view_id), image)

        plt.hist(lidar_2d_view[:, 2], 15)
        plt.savefig("vis/images/b%d_v%d_dist.png"%(i, view_id))
        plt.show()
        plt.close()
