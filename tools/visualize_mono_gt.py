import json
import cv2
import os
import numpy as np
from collections import Counter

from vis_util import draw_camera_bbox3d_on_img, draw_bboxes_2d
from mmdet3d.core.bbox import CameraInstance3DBoxes

import random

image_num = 100
data_dir = "data/nuscenes"

with open(os.path.join(data_dir, "nuscenes_infos_val_mono3d.coco.json"), "r") as file:
    mono_data = json.load(file)

img2ann = {}
for ann_item in mono_data['annotations']:
    img_id = ann_item['image_id']
    if img_id in img2ann:
        img2ann[img_id].append(ann_item)
    else:
        img2ann[img_id] = [ann_item]


random.seed(0)
random.shuffle(mono_data['images'])

for i in range(image_num):
    image_item = mono_data['images'][i]
    img_id = image_item['id']
    filename = image_item['file_name']
    filepath = os.path.join(data_dir, filename)
    image = cv2.imread(filepath)
    if img_id not in img2ann:
        cv2.imwrite("vis/images/%d.png" % i, image)
        continue
    img_anns = img2ann[img_id]

    cam_intrinsic = image_item['cam_intrinsic']

    new_image = image
    all_bbox_cam3d = []
    all_bbox_2d = []
    for ann_item in img_anns:
        bbox_cam3d = ann_item['bbox_cam3d']
        # bbox_cam3d[6] = -np.arctan2(bbox_cam3d[0], bbox_cam3d[2]) + bbox_cam3d[6]
        all_bbox_cam3d.append(bbox_cam3d)

        bbox_2d = ann_item['bbox']
        bbox_2d[0] = bbox_2d[0] + bbox_2d[2] / 2
        bbox_2d[1] = bbox_2d[1] + bbox_2d[3] / 2

        all_bbox_2d.append(bbox_2d)

    if len(all_bbox_cam3d) == 0:
        continue

    all_bbox_2d = np.array(all_bbox_2d)
    all_bbox_cam3d = np.array(all_bbox_cam3d)

    new_image = draw_bboxes_2d(all_bbox_2d, new_image)

    bbox_cam3d = CameraInstance3DBoxes(all_bbox_cam3d, box_dim=7, origin=(0.5, 0.5, 0.5))

    new_image = draw_camera_bbox3d_on_img(bbox_cam3d, new_image, cam_intrinsic)

    cv2.imwrite("vis/images/%d.png"%i, new_image)
