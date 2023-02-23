import copy
import inspect
import math
import warnings

import cv2
import mmcv
import numpy as np
from numpy import random

from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class OurRandomAffine:
    """Random affine transform data augmentation.
    This operation randomly generates affine transform matrix which including
    rotation, translation, shear and scaling transforms.
    Args:
        max_rotate_degree (float): Maximum degrees of rotation transform.
            Default: 10.
        max_translate_ratio (float): Maximum ratio of translation.
            Default: 0.1.
        scaling_ratio_range (tuple[float]): Min and max ratio of
            scaling transform. Default: (0.5, 1.5).
        max_shear_degree (float): Maximum degrees of shear
            transform. Default: 2.
        border (tuple[int]): Distance from height and width sides of input
            image to adjust output shape. Only used in mosaic dataset.
            Default: (0, 0).
        border_val (tuple[int]): Border padding values of 3 channels.
            Default: (114, 114, 114).
        min_bbox_size (float): Width and height threshold to filter bboxes.
            If the height or width of a box is smaller than this value, it
            will be removed. Default: 2.
        min_area_ratio (float): Threshold of area ratio between
            original bboxes and wrapped bboxes. If smaller than this value,
            the box will be removed. Default: 0.2.
        max_aspect_ratio (float): Aspect ratio of width and height
            threshold to filter bboxes. If max(h/w, w/h) larger than this
            value, the box will be removed.
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        skip_filter (bool): Whether to skip filtering rules. If it
            is True, the filter rule will not be applied, and the
            `min_bbox_size` and `min_area_ratio` and `max_aspect_ratio`
            is invalid. Default to True.
    """

    def __init__(self,
                 # max_translate_ratio=0.1,
                 scaling_ratio_range=(0.5, 1.5),
                 flip_ratio=0.5,
                 border=(0, 0),
                 border_val=(103.53, 116.28, 123.675),
                 bbox_clip_border=True,
                 flip_sync_3d=False,
                 scaling_sync_view=False,
                 trans_when_scaling=True,
    ):
        # assert 0 <= max_translate_ratio <= 1
        assert scaling_ratio_range[0] <= scaling_ratio_range[1]
        assert scaling_ratio_range[0] > 0
        # self.max_translate_ratio = max_translate_ratio
        self.scaling_ratio_range = scaling_ratio_range
        self.flip_ratio = flip_ratio
        self.border = border
        self.border_val = border_val
        self.bbox_clip_border = bbox_clip_border
        self.flip_sync = flip_sync_3d
        self.scaling_sync_view = scaling_sync_view
        self.trans_when_scaling = trans_when_scaling

    def _transform_bbox(self, results, warp_mats, flips, width, height):
        valid_mask = np.ones(results['gt_labels'].shape[0]) > 0

        if 'gt_bboxes_cam_3d' in results:
            bboxes_cam = results['gt_bboxes_cam_3d']
        else:
            bboxes_cam = None

        for view_id in range(len(warp_mats)):
            warp_matrix = warp_mats[view_id]
            bbox_mask = results['gt_labels'][:, 1] == view_id

            if np.sum(bbox_mask) == 0:
                continue

            flip = flips[view_id]
            flip_matrix = self._get_flip_matrix(flip, width)

            if bboxes_cam is not None:
                if flip:
                    bboxes_cam.tensor[bbox_mask, 0::7] = -bboxes_cam.tensor[bbox_mask, 0::7]
                    bboxes_cam.tensor[bbox_mask, 6] = -bboxes_cam.tensor[bbox_mask, 6] + np.pi

            bbox_view = results['gt_bboxes'][bbox_mask]
            centers_view = results['gt_img_centers_2d'][bbox_mask, :2]
            num_bboxes = bbox_view.shape[0]

            xtl = bbox_view[:, 0] - bbox_view[:, 2] / 2
            ytl = bbox_view[:, 1] - bbox_view[:, 3] / 2
            xtr = bbox_view[:, 0] + bbox_view[:, 2] / 2
            ytr = bbox_view[:, 1] - bbox_view[:, 3] / 2

            xbl = bbox_view[:, 0] - bbox_view[:, 2] / 2
            ybl = bbox_view[:, 1] + bbox_view[:, 3] / 2
            xbr = bbox_view[:, 0] + bbox_view[:, 2] / 2
            ybr = bbox_view[:, 1] + bbox_view[:, 3] / 2

            xs = np.vstack([xtl, xtr, xbl, xbr]).T  # [N, 4]
            ys = np.vstack([ytl, ytr, ybl, ybr]).T  # [N, 4]

            xs = xs.reshape(-1)  # [N*4,]
            ys = ys.reshape(-1)  # [N*4,]
            ones = np.ones_like(ys)

            points = np.vstack([xs, ys, ones])  # [3, N*4]

            warp_points = warp_matrix @ flip_matrix @ points  # [3, N*4]
            warp_points = warp_points[:2] / warp_points[2]
            xs = warp_points[0].reshape(num_bboxes, 4)  # [N, 4]
            ys = warp_points[1].reshape(num_bboxes, 4)  # [N, 4]

            xs_min = xs.min(1)  # [N, ]
            ys_min = ys.min(1)  # [N, ]
            xs_max = xs.max(1)  # [N, ]
            ys_max = ys.max(1)  # [N, ]

            if self.bbox_clip_border:
                xs_min = xs_min.clip(0, width)
                xs_max = xs_max.clip(0, width)
                ys_min = ys_min.clip(0, height)
                ys_max = ys_max.clip(0, height)

            cxs = (xs_min + xs_max) / 2
            cys = (ys_min + ys_max) / 2
            ws = xs_max - xs_min
            hs = ys_max - ys_min

            warp_bboxes = np.vstack((cxs, cys, ws, hs)).T  # [N, 4]

            ones = np.ones_like(centers_view[:, :1])  # [N, 1]
            center_points = np.concatenate([centers_view, ones], axis=1).T  # [3, N]

            warp_points = warp_matrix @ flip_matrix @ center_points  # [3, N]
            warp_points = warp_points[:2] / warp_points[2]
            new_center_points = warp_points.T  # [N, 2]

            valid_mask_view = (new_center_points[:, 0] > 0) & (new_center_points[:, 0] < width-1) & (new_center_points[:, 1] > 0) & (new_center_points[:, 1] < height-1)

            valid_mask[bbox_mask] = valid_mask_view

            results['gt_bboxes'][bbox_mask] = warp_bboxes
            results['gt_img_centers_2d'][bbox_mask, :2] = new_center_points

        if 'gt_bboxes_cam_3d' in results:
            results['gt_bboxes_cam_3d'] = bboxes_cam[valid_mask]

        results['gt_bboxes_lidar'] = results['gt_bboxes_lidar'][valid_mask]

        results['gt_bboxes'] = results['gt_bboxes'][valid_mask]
        results['gt_img_centers_2d'] = results['gt_img_centers_2d'][valid_mask]
        results['gt_pts_centers_2d'] = results['gt_pts_centers_2d'][valid_mask]
        results['gt_labels'] = results['gt_labels'][valid_mask]
        return results

    def _transform_camera(self, results, warp_mats, flips, width):
        for id in range(len(warp_mats)):
            flip = flips[id]
            flip_matrix = self._get_flip_matrix(flip, width)

            intrinsic = results['cam_intrinsic'][id]
            warp_matrix = warp_mats[id] @ flip_matrix

            # intrinsic = warp_matrix @ intrinsic
            # results['cam_intrinsic'][id] = intrinsic

            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = warp_matrix
            results['lidar2img'][id] = viewpad @ results['lidar2img'][id]

            if flip:
                flip_matrix = np.eye(3)
                flip_matrix[0, 0] = -1
                results['lidar2cam_r'][id] = flip_matrix @ results['lidar2cam_r'][id]
                results['lidar2cam_t'][id] = flip_matrix @ results['lidar2cam_t'][id]
                results['cam_intrinsic'][id][0, 2] = width - results['cam_intrinsic'][id][0, 2]

            intrinsic = warp_mats[id] @ intrinsic
            results['cam_intrinsic'][id] = intrinsic
        return results

    def __call__(self, results):
        translate_mats = []
        scale_mats = []
        warp_mats = []
        flips = []
        scaling_ratios = []
        valid_shapes = []
        results['image_flip'] = []

        flip_3d = False
        if 'pcd_horizontal_flip' in results and results['pcd_horizontal_flip'] == True:
            flip_3d = not flip_3d
        if 'pcd_vertical_flip' in results and results['pcd_vertical_flip'] == True:
            flip_3d = not flip_3d

        if self.scaling_sync_view:
            scaling_ratio = random.uniform(self.scaling_ratio_range[0], self.scaling_ratio_range[1])

        for view_id in range(len(results['img'])):
            img = results['img'][view_id]
            height = img.shape[0] + self.border[0] * 2
            width = img.shape[1] + self.border[1] * 2

            if self.flip_sync:
                flip = flip_3d
            else:
                flip = True if np.random.random() < self.flip_ratio else False

            flips.append(flip)
            if flip:
                results['image_flip'].append(True)
                img = cv2.flip(img, 1)
            else:
                results['image_flip'].append(False)

            # Scaling
            if not self.scaling_sync_view:
                scaling_ratio = random.uniform(self.scaling_ratio_range[0], self.scaling_ratio_range[1])
            scaling_matrix = self._get_scaling_matrix(scaling_ratio)
            scaling_ratios.append(scaling_ratio)
            reduction_ratio = min(1.0, scaling_ratio)
            valid_shapes.append([reduction_ratio*width, reduction_ratio*height])

            # Translation
            if self.trans_when_scaling:
                if scaling_ratio <= 1:
                    trans_x = 0
                    trans_y = 0
                else:
                    trans_x = random.uniform((1 - scaling_ratio) * width, 0)
                    trans_y = random.uniform((1 - scaling_ratio) * height, 0)
            else:
                trans_x = 0
                trans_y = 0

            # trans_x = random.uniform(-self.max_translate_ratio, self.max_translate_ratio) * width
            # trans_y = random.uniform(-self.max_translate_ratio, self.max_translate_ratio) * height
            translate_matrix = self._get_translation_matrix(trans_x, trans_y)

            warp_matrix = translate_matrix  @ scaling_matrix

            img = cv2.warpPerspective(
                img,
                warp_matrix,
                dsize=(width, height),
                borderValue=self.border_val
            )

            results['img'][view_id] = img
            translate_mats.append(translate_matrix)
            scale_mats.append(scaling_matrix)
            warp_mats.append(warp_matrix)
            # results['img_shape'] = img.shape

        results['valid_shape'] = np.array(valid_shapes)
        results['img_scale_ratios'] = np.array(scaling_ratios)
        results = self._transform_bbox(results, warp_mats, flips, width, height)
        results = self._transform_camera(results, warp_mats, flips, width)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        # repr_str += f'max_translate_ratio={self.max_translate_ratio}, '
        repr_str += f'scaling_ratio={self.scaling_ratio_range}, '
        repr_str += f'flip_ratio={self.flip_ratio}, '
        repr_str += f'border={self.border}, '
        repr_str += f'border_val={self.border_val}, '
        return repr_str

    @staticmethod
    def _get_scaling_matrix(scale_ratio):
        scaling_matrix = np.array(
            [[scale_ratio, 0., 0.], [0., scale_ratio, 0.], [0., 0., 1.]],
            dtype=np.float32)
        return scaling_matrix

    @staticmethod
    def _get_translation_matrix(x, y):
        translation_matrix = np.array([[1, 0., x], [0., 1, y], [0., 0., 1.]],
                                      dtype=np.float32)
        return translation_matrix

    @staticmethod
    def _get_flip_matrix(flip, width):
        if flip:
            flip_matrix = np.array([
                [-1, 0, width],
                [0, 1, 0],
                [0, 0, 1]
            ])
        else:
            flip_matrix = np.eye(3)
        return flip_matrix


@PIPELINES.register_module()
class PhotoMetricDistortionMultiViewImage:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18,
                 swap_channel=True):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta
        self.swap_channel = swap_channel

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = results['img']
        new_imgs = []
        for img_ in imgs:
            img = img_.astype(np.float32)
            assert img.dtype == np.float32, \
                'PhotoMetricDistortion needs the input image of dtype np.float32,'\
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            # random brightness
            if random.randint(2):
                delta = random.uniform(-self.brightness_delta,
                                    self.brightness_delta)
                img += delta
                img = np.clip(img, a_max=255, a_min=0)

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha
                    img = np.clip(img, a_max=255, a_min=0)

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(self.saturation_lower,
                                            self.saturation_upper)
                img[..., 1] = np.clip(img[..., 1], a_max=1, a_min=0)

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    # import pdb
                    # pdb.set_trace()
                    alpha = random.uniform(self.contrast_lower, self.contrast_upper)
                    img *= alpha
                    # import pdb
                    # pdb.set_trace()
                    img = np.clip(img, a_max=255, a_min=0)

            # randomly swap channels
            if self.swap_channel:
                if random.randint(2):
                    img = img[..., random.permutation(3)]
            new_imgs.append(img.astype(np.uint8))
        results['img'] = new_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(\nbrightness_delta={self.brightness_delta},\n'
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)},\n'
        repr_str += 'saturation_range='
        repr_str += f'{(self.saturation_lower, self.saturation_upper)},\n'
        repr_str += f'hue_delta={self.hue_delta})'
        return repr_str
