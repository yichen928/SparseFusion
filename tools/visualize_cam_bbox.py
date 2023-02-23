import json
import pickle
import cv2
import os
import numpy as np
from collections import Counter

from vis_util import draw_camera_bbox3d_on_img, draw_bboxes_2d
from mmdet3d.core.bbox import CameraInstance3DBoxes

import random

image_num = 20
data_dir = "data/nuscenes"
cam_orders = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT']

with open(os.path.join(data_dir, "nuscenes_infos_train_with_proj2d_wbox3d_woRange.pkl"), "rb") as file:
    mono_data = pickle.load(file, encoding="bytes")


infos = mono_data['infos']

random.seed(0)
random.shuffle(infos)

for i in range(image_num):
    cam_items = infos[i]['cams']

    view_2d = infos[i]['gt_views_2d']
    cam_box_3d = infos[i]['gt_bboxes_cam_3d']
    cam_velo = infos[i]['gt_bboxes_cam_vel']
    box_2d = infos[i]['gt_bboxes_2d']
    center_2d = infos[i]['gt_img_centers_2d']

    loc = cam_box_3d[:, :3]
    cam_velo_3d = np.concatenate([cam_velo[:, :1], np.zeros_like(cam_velo[:, :1]), cam_velo[:, 1:2]], axis=1) * 2

    target_loc = loc + cam_velo_3d

    assert view_2d.shape[0] == cam_box_3d.shape[0] == cam_velo.shape[0] == box_2d.shape[0] == center_2d.shape[0]

    for view_id, cam_name in enumerate(cam_orders):
        cam_item = cam_items[cam_name]
        filepath = cam_item['data_path']
        image = cv2.imread(filepath)

        view_mask = view_2d == view_id
        cam_box_3d_view = cam_box_3d[view_mask]
        box_2d_view = box_2d[view_mask]
        center_2d_view = center_2d[view_mask]
        target_loc_view = target_loc[view_mask]
        loc_view = loc[view_mask]
        cam_velo_3d_view = cam_velo_3d[view_mask]

        if cam_box_3d_view.shape[0] == 0:
            cv2.imwrite("vis/images/%d_%d.png" % (i, view_id), image)
            continue

        cam_intrinsic = cam_item['cam_intrinsic']

        cam_box_3d_view = CameraInstance3DBoxes(cam_box_3d_view, box_dim=7, origin=(0.5, 0.5, 0.5))

        image = draw_camera_bbox3d_on_img(cam_box_3d_view, image, cam_intrinsic)

        target_loc_2d_view = target_loc_view @ cam_intrinsic.T
        loc_2d_view = loc_view @ cam_intrinsic.T

        target_loc_2d_view[:, :2] = target_loc_2d_view[:, :2] / target_loc_2d_view[:, 2:3]
        loc_2d_view[:, :2] = loc_2d_view[:, :2] / loc_2d_view[:, 2:3]

        for j in range(loc_2d_view.shape[0]):
            if target_loc_2d_view[j, 2] > 0:
                image = cv2.arrowedLine(image, loc_2d_view[j, :2].astype(np.int32), target_loc_2d_view[j, :2].astype(np.int32), (255, 0, 0), 5)

        box_2d_view[:, 0] = box_2d_view[:, 0] + box_2d_view[:, 2] / 2
        box_2d_view[:, 1] = box_2d_view[:, 1] + box_2d_view[:, 3] / 2

        image = draw_bboxes_2d(box_2d_view, image)

        cv2.imwrite("vis/images/%d_%d.png" %(i, view_id), image)
