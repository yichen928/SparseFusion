import os
import pickle
import json
import numpy as np
import torch
import argparse
from mmdet3d.core import Box3DMode, LiDARInstance3DBoxes

from mmdet3d.core.bbox.structures.utils import limit_period

cam_orders = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT']

def project_to_image(points, cam_int, cam_ext, img_h, img_w):
    num_points = points.shape[0]
    pts_4d = np.concatenate([points[:, :3], np.ones((num_points, 1))], axis=-1)
    pts_cam_4d = pts_4d @ cam_ext.T
    pts_2d = pts_cam_4d @ cam_int.T
    # cam_points is Tensor of Nx4 whose last column is 1
    # transform camera coordinate to image coordinate

    depth_mask = pts_2d[:, 2] > 0
    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=99999)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]

    fov_inds = ((pts_2d[:, 0] < img_w)
                & (pts_2d[:, 0] >= 0)
                & (pts_2d[:, 1] < img_h)
                & (pts_2d[:, 1] >= 0)
                & depth_mask)
    return pts_2d, pts_cam_4d, fov_inds


def combine_data(data_root, info_file, coco_file, output_file):
    info_path = os.path.join(data_root, info_file)
    coco_file_path = os.path.join(data_root, coco_file)

    with open(info_path, "rb") as file:
        info = pickle.load(file, encoding="bytes")

    with open(coco_file_path, "r") as file:
        coco = json.load(file)

    id2image = {}
    for image in coco["images"]:
        image_id = image['id']
        if image_id not in id2image:
            id2image[image_id] = image
    print("Generate new info file")
    all_depths = []

    for info_id, info_dict in enumerate(info['infos']):
        if info_id % 1000 == 1:
            print(info_id, "/", len(info['infos']))
        valid_flag = info_dict['num_lidar_pts'] > 0

        gt_visible_3d = np.zeros((info_dict['gt_boxes'].shape[0], ), dtype=np.int32)

        gt_boxes_3d_tensor = info_dict['gt_boxes'][valid_flag]

        gt_names_3d = info_dict['gt_names'][valid_flag]
        gt_vel_2d = info_dict['gt_velocity'][valid_flag]
        gt_vel_3d = np.concatenate([gt_vel_2d, np.zeros_like(gt_vel_2d[:,:1])], axis=1)
        gt_visible_3d_valid = np.zeros((gt_boxes_3d_tensor.shape[0], ), dtype=np.int32)

        if gt_boxes_3d_tensor.shape[0] == 0:
            info_dict['gt_bboxes2d_view'] = np.zeros((0, 4))
            info_dict['gt_names2d_view'] = []
            info_dict['gt_viewsIDs'] = np.zeros(0)
            info_dict['gt_pts_centers_view'] = np.zeros((0, 3))
            info_dict['gt_img_centers_view'] = np.zeros((0, 3))
            info_dict['gt_bboxes_cam_view'] = np.zeros((0, 7))
            info_dict['gt_velocity_cam_view'] = np.zeros((0, 2))
            info_dict['gt_visible'] = gt_visible_3d
            info_dict['gt_bboxes_lidar_view'] = np.zeros((0, 7))
            info_dict['gt_velocity_lidar_view'] = np.zeros((0, 2))
            continue

        gt_boxes_3d = LiDARInstance3DBoxes(gt_boxes_3d_tensor, box_dim=7, origin=(0.5, 0.5, 0.5)).convert_to(Box3DMode.LIDAR)

        corners = gt_boxes_3d.corners
        centers = gt_boxes_3d.gravity_center
        dims = gt_boxes_3d.dims
        yaws = gt_boxes_3d.yaw[:,None]

        view_ids = []
        bboxes = []
        gt_names = []
        pts_centers = []
        img_centers = []

        bboxes_cam_3d = []
        vel_cam = []

        bboxes_lidar = []
        vels_lidar = []

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

            centers_2d, center_cam_3d, view_mask = project_to_image(centers, viewpad, lidar2cam_rt.T, img_h, img_w)
            centers_2d = centers_2d[..., :3]
            center_cam_3d = center_cam_3d[..., :3]

            corners_view = corners[view_mask].cpu().numpy()
            names_view = gt_names_3d[view_mask]
            centers_view = centers[view_mask].cpu().numpy()
            centers_2d_view = centers_2d[view_mask]
            dims_view = dims[view_mask].cpu().numpy()
            yaw_view = -yaws[view_mask] - np.pi / 2
            center_cam_3d_view = center_cam_3d[view_mask]
            vel_3d_view = gt_vel_3d[view_mask]

            boxes_lidar_view = gt_boxes_3d_tensor[view_mask]
            vels_lidar_view = gt_vel_2d[view_mask]

            vel_cam_3d_view = vel_3d_view @ lidar2cam_r.T
            vel_cam_view = vel_cam_3d_view[:, [0, 2]]

            dims_view = dims_view[:, [1, 2, 0]]

            rot_dir_view = torch.cat([torch.cos(yaw_view), torch.sin(yaw_view), torch.zeros_like(yaw_view)], dim=1)
            rot_dir_view = rot_dir_view @ lidar2cam_r.T
            rot_dir_view = rot_dir_view[:, [0, 2]]

            yaw_view = -torch.atan2(rot_dir_view[:, 1:2], rot_dir_view[:, 0:1])
            yaw_view = limit_period(yaw_view, period=2*np.pi).cpu().numpy()

            bboxes_cam_3d_view = np.concatenate([center_cam_3d_view, dims_view, yaw_view], axis=1)

            gt_visible_3d_valid_view = np.zeros((view_mask.sum(),), dtype=np.int32)

            ann_num = corners_view.shape[0]
            if ann_num == 0:
                continue

            for ann_id in range(ann_num):
                corner_2d, _, _ = project_to_image(corners_view[ann_id], viewpad, lidar2cam_rt.T, img_h, img_w)  # (8, 2)
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

                bboxes.append([x1, y1, w, h])
                view_ids.append(view_id)
                gt_names.append(names_view[ann_id])
                pts_centers.append(centers_view[ann_id])
                img_centers.append(centers_2d_view[ann_id])
                all_depths.append(centers_2d_view[ann_id][2])
                bboxes_cam_3d.append(bboxes_cam_3d_view[ann_id])
                vel_cam.append(vel_cam_view[ann_id])
                bboxes_lidar.append(boxes_lidar_view[ann_id])
                vels_lidar.append(vels_lidar_view[ann_id])
                gt_visible_3d_valid_view[ann_id] = 1

            gt_visible_3d_valid[view_mask] = gt_visible_3d_valid_view


        view_ids = np.array(view_ids)
        bboxes = np.array(bboxes).reshape(-1, 4)
        pts_centers = np.array(pts_centers).reshape(-1, 3)
        img_centers = np.array(img_centers).reshape(-1, 3)
        bboxes_cam_3d = np.vstack(bboxes_cam_3d)
        vel_cam = np.vstack(vel_cam)

        bboxes_lidar = np.vstack(bboxes_lidar)
        vels_lidar = np.vstack(vels_lidar)

        gt_visible_3d[valid_flag] = gt_visible_3d_valid

        info_dict['gt_bboxes2d_view'] = bboxes
        info_dict['gt_names2d_view'] = gt_names
        info_dict['gt_viewsIDs'] = view_ids
        info_dict['gt_pts_centers_view'] = pts_centers
        info_dict['gt_img_centers_view'] = img_centers

        info_dict['gt_bboxes_cam_view'] = bboxes_cam_3d
        info_dict['gt_velocity_cam_view'] = vel_cam
        info_dict['gt_visible'] = gt_visible_3d

        info_dict['gt_bboxes_lidar_view'] = bboxes_lidar
        info_dict['gt_velocity_lidar_view'] = vels_lidar

    output_path = os.path.join(data_root, output_file)
    with open(output_path, "wb") as file:
        pickle.dump(info, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combine the 3D bboxes in the camera coordinates into the ann file')
    parser.add_argument('--data_root', type=str, default='./data/nuscenes/', help='root path of dataset')
    parser.add_argument('--info_tag', type=str, default='nuscenes_infos', help='data info filename prefix')
    parser.add_argument('--output_tag', type=str, default='nuscenes_infos_w_views', help='output filename prefix')
    parser.add_argument('--output_file', type=str, default='./data/nuscenes/', help='root path of dataset')
    args = parser.parse_args()

    for split in ["train", "val"]:
        info_file = args.info_tag + "_%s.pkl"%split
        coco_file = args.info_tag + "_%s.coco.json"%split
        output_file = args.output_tag + "_%s.pkl"%split

        print("Processing %s data"%split)
        combine_data(args.data_root, info_file, coco_file, output_file)
