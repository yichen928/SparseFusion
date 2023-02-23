import os
import pickle
import json
import numpy as np
from mmdet3d.core import Box3DMode, LiDARInstance3DBoxes

cam_orders = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT']

data_root = "data/nuscenes/"

info_file = "nuscenes_infos_val.pkl"
coco_file = "nuscenes_infos_val.coco.json"
# info_gt2d_file = "nuscenes_infos_val_with_gt2d.pkl"

output_file = "nuscenes_infos_val_with_proj2d.pkl"

pc_range = [-65, -65.0, -5.0, 65.0, 65.0, 3.0]

info_path = os.path.join(data_root, info_file)
# info_gt2d_path = os.path.join(data_root, info_gt2d_file)

coco_file_path = os.path.join(data_root, coco_file)


def project_to_image(points, lidar2img_rt, img_h, img_w):
    num_points = points.shape[0]
    pts_4d = np.concatenate([points[:, :3], np.ones((num_points, 1))], axis=-1)
    pts_2d = pts_4d @ lidar2img_rt.T
    # cam_points is Tensor of Nx4 whose last column is 1
    # transform camera coordinate to image coordinate
    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=99999)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]

    fov_inds = ((pts_2d[:, 0] < img_w)
                & (pts_2d[:, 0] >= 0)
                & (pts_2d[:, 1] < img_h)
                & (pts_2d[:, 1] >= 0))
    return pts_2d, fov_inds


with open(info_path, "rb") as file:
    info = pickle.load(file, encoding="bytes")

# with open(info_gt2d_path, "rb") as file:
#     info_gt2d = pickle.load(file, encoding="bytes")

with open(coco_file_path, "r") as file:
    coco = json.load(file)

id2image = {}
for image in coco["images"]:
    image_id = image['id']
    if image_id not in id2image:
        id2image[image_id] = image
print("total %d images"%len(id2image))

print("Generate new info file")

for info_id, info_dict in enumerate(info['infos']):
    # info_gt2d_dict = info_gt2d['infos'][info_id]
    if info_id % 200 == 1:
        print(info_id, "/", len(info['infos']))

    valid_flag = info_dict['valid_flag']
    gt_boxes_3d = info_dict['gt_boxes'][valid_flag]
    gt_names_3d = info_dict['gt_names'][valid_flag]
    if gt_boxes_3d.shape[0] == 0:
        info_dict['gt_bboxes_2d'] = np.zeros((0, 4))
        info_dict['gt_names_2d'] = []
        info_dict['gt_views_2d'] = np.zeros(0)
        continue
    # gt_boxes_3d = LiDARInstance3DBoxes(gt_boxes_3d, box_dim=7)
    gt_boxes_3d = LiDARInstance3DBoxes(gt_boxes_3d, box_dim=7, origin=(0.5, 0.5, 0.5)).convert_to(Box3DMode.LIDAR)

    corners = gt_boxes_3d.corners
    centers = gt_boxes_3d.gravity_center

    view_ids = []
    bboxes = []
    gt_names = []
    pt_centers = []
    for view_id, cam in enumerate(cam_orders):
        cam_info = info_dict['cams'][cam]
        lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
        lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
        lidar2cam_rt = np.eye(4)
        lidar2cam_rt[:3, :3] = lidar2cam_r.T
        lidar2cam_rt[3, :3] = -lidar2cam_t
        intrinsic = cam_info['cam_intrinsic']
        viewpad = np.eye(4)
        viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
        lidar2img_rt = (viewpad @ lidar2cam_rt.T)

        image_token = cam_info['sample_data_token']
        image_info = id2image[image_token]
        img_h = image_info['height']
        img_w = image_info['width']

        _, view_mask = project_to_image(centers, lidar2img_rt, img_h, img_w)

        corners_view = corners[view_mask].cpu().numpy()
        names_view = gt_names_3d[view_mask]
        centers_view = centers[view_mask].cpu().numpy()

        ann_num = corners_view.shape[0]
        if ann_num == 0:
            continue

        for ann_id in range(ann_num):
            corner_2d, _ = project_to_image(corners_view[ann_id], lidar2img_rt, img_h, img_w)  # (8, 2)
            coord_min = np.min(corner_2d, axis=0)
            coord_max = np.max(corner_2d, axis=0)

            x1, y1 = coord_min[0], coord_min[1]
            x2, y2 = coord_max[0], coord_max[1]

            x1 = max(x1, 0)
            y1 = max(y1, 0)
            x2 = min(x2, img_w)
            y2 = min(y2, img_h)
            w = x2 - x1
            h = y2 - y1

            center_ann = centers_view[ann_id]
            if center_ann[0] > pc_range[0] and center_ann[0] < pc_range[3] and center_ann[1] > pc_range[1] and center_ann[1] < pc_range[4]:
                bboxes.append([x1, y1, w, h])
                view_ids.append(view_id)
                gt_names.append(names_view[ann_id])

    view_ids = np.array(view_ids)
    bboxes = np.array(bboxes).reshape(-1, 4)
    pt_centers = np.array(pt_centers).reshape(-1, 3)

    info_dict['gt_bboxes_2d'] = bboxes
    info_dict['gt_names_2d'] = gt_names
    info_dict['gt_views_2d'] = view_ids
    info_dict['gt_pt_centers_2d'] = pt_centers

output_path = os.path.join(data_root, output_file)
with open(output_path, "wb") as file:
    pickle.dump(info, file)
