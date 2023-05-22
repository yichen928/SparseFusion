import mmcv
import numpy as np
import pyquaternion
import tempfile
from nuscenes.utils.data_classes import Box as NuScenesBox
from os import path as osp

from mmdet.datasets import DATASETS
from ..core import show_result
from ..core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes, CameraInstance3DBoxes
from .nuscenes_dataset import NuScenesDataset


@DATASETS.register_module()
class NuScenesDataset_ViewInfo(NuScenesDataset):
    """
        Compared with NuScenesDataset, we also load 2d annotations
    """

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:
        """
        info = self.data_infos[index]
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0

        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]
        gt_visible_3d = info['gt_visible'][mask]

        # .copy() cannot be missed!
        gt_bboxes2d_view = info['gt_bboxes2d_view'].copy()
        gt_bboxes2d_view[..., :2] = gt_bboxes2d_view[..., :2] + gt_bboxes2d_view[..., 2:4] / 2

        gt_bboxes_lidar_view = info['gt_bboxes_lidar_view'].copy()

        gt_names2d_view = info['gt_names2d_view']
        gt_viewsIDs = info['gt_viewsIDs']
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        gt_labels2d_view = []
        for cat in gt_names2d_view:
            if cat in self.CLASSES:
                gt_labels2d_view.append(self.CLASSES.index(cat))
            else:
                gt_labels2d_view.append(-1)
        gt_labels2d_view = np.array(gt_labels2d_view)
        gt_labels2d_view = np.stack([gt_labels2d_view, gt_viewsIDs], axis=-1)

        gt_bboxes_cam_view = info['gt_bboxes_cam_view'].copy()

        if self.with_velocity:
            gt_velocity = info['gt_velocity'][mask].copy()
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

            gt_cam_vel = info['gt_velocity_cam_view'].copy()
            nan_mask_cam = np.isnan(gt_cam_vel[:, 0])
            gt_cam_vel[nan_mask_cam] = [0.0, 0.0]
            gt_bboxes_cam_view = np.concatenate([gt_bboxes_cam_view, gt_cam_vel], axis=-1)

            gt_lidar_vel =info['gt_velocity_lidar_view'].copy()
            nan_mask_lidar = np.isnan(gt_lidar_vel[:, 0])
            gt_lidar_vel[nan_mask_lidar] = [0.0, 0.0]

            gt_bboxes_lidar_view = np.concatenate([gt_bboxes_lidar_view, gt_lidar_vel], axis=-1)

        gt_bboxes_cam_view = CameraInstance3DBoxes(
            gt_bboxes_cam_view,
            box_dim=gt_bboxes_cam_view.shape[-1],
            origin=(0.5, 0.5, 0.5)
        )

        gt_bboxes_lidar_view = LiDARInstance3DBoxes(
            gt_bboxes_lidar_view,
            box_dim=gt_bboxes_lidar_view.shape[-1],
            origin=(0.5, 0.5, 0.5)
        ).convert_to(self.box_mode_3d)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        if "gt_pts_centers_view" in info:
            gt_pts_centers_view = info['gt_pts_centers_view'].copy()
            gt_img_centers_view = info['gt_img_centers_view'].copy()

            anns_results = dict(
                gt_bboxes_3d=gt_bboxes_3d,
                gt_labels_3d=gt_labels_3d,
                gt_visible_3d=gt_visible_3d,
                gt_names=gt_names_3d,
                bboxes=gt_bboxes2d_view,
                labels=gt_labels2d_view,
                pts_centers_view=gt_pts_centers_view,
                img_centers_view=gt_img_centers_view,
                bboxes_cam_view=gt_bboxes_cam_view,
                bboxes_lidar_view=gt_bboxes_lidar_view,
            )

        else:
            anns_results = dict(
                gt_bboxes_3d=gt_bboxes_3d,
                gt_labels_3d=gt_labels_3d,
                gt_visible_3d=gt_visible_3d,
                gt_names=gt_names_3d,
                bboxes=gt_bboxes2d_view,
                labels=gt_labels2d_view,
            )

        return anns_results

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch

        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            timestamp=info['timestamp'] / 1e6,
        )

        cam_orders = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT']
        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            # for cam_type, cam_info in info['cams'].items():
            intrinsics = []
            lidar2cam_rs = []
            lidar2cam_ts = []

            for cam_type in cam_orders:
                cam_info = info['cams'][cam_type]
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                    'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt.copy())
                intrinsics.append(intrinsic.copy())
                lidar2cam_rs.append(lidar2cam_r.copy())
                lidar2cam_ts.append(-lidar2cam_t.copy())

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=intrinsics,
                    lidar2cam_r=lidar2cam_rs,
                    lidar2cam_t=lidar2cam_ts,
                ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict
