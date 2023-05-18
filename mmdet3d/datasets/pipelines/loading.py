import mmcv
import numpy as np
import torch
import cv2
import copy

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations


@PIPELINES.register_module()
class MyResize(object):
    """Resize images & bbox & mask.

    This transform resizes the input image to some scale. Bboxes and masks are
    then resized with the same scale factor. If the input dict contains the key
    "scale", then the scale in the input dict is used, otherwise the specified
    scale in the init method is used. If the input dict contains the key
    "scale_factor" (if MultiScaleFlipAug does not give img_scale but
    scale_factor), the actual scale will be computed by image shape and
    scale_factor.

    `img_scale` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:

    - ``ratio_range is not None``: randomly sample a ratio from the ratio \
      range and multiply it with the image scale.
    - ``ratio_range is None`` and ``multiscale_mode == "range"``: randomly \
      sample a scale from the multiscale range.
    - ``ratio_range is None`` and ``multiscale_mode == "value"``: randomly \
      sample a scale from multiple scales.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
        bbox_clip_border (bool, optional): Whether clip the objects outside
            the border of the image. Defaults to True.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
        override (bool, optional): Whether to override `scale` and
            `scale_factor` so as to call resize twice. Default False. If True,
            after the first resizing, the existed `scale` and `scale_factor`
            will be ignored so the second resizing can be allowed.
            This option is a work-around for multiple times of resize in DETR.
            Defaults to False.
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 bbox_clip_border=True,
                 backend='cv2',
                 override=False):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.backend = backend
        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        # TODO: refactor the override option in Resize
        self.override = override
        self.bbox_clip_border = bbox_clip_border

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.

        Args:
            img_scales (list[tuple]): Images scales for selection.

        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``, \
                where ``img_scale`` is the selected image scale and \
                ``scale_idx`` is the selected index in the given candidates.
        """

        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``.

        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower
                and uper bound of image scales.

        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where \
                ``img_scale`` is sampled scale and None is just a placeholder \
                to be consistent with :func:`random_select`.
        """

        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.

        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.

        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where \
                ``scale`` is sampled ratio multiplied with ``img_scale`` and \
                None is just a placeholder to be consistent with \
                :func:`random_select`.
        """

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into \
                ``results``, which would be used by subsequent pipelines.
        """

        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        imgs = results['img']
        results['img'] = [imgs[i] for i in range(len(imgs))]
        for key in results.get('img_fields', ['img']):
            for idx in range(len(results['img'])):
                if self.keep_ratio:
                    img, scale_factor = mmcv.imrescale(
                        results[key][idx],
                        results['scale'],
                        return_scale=True,
                        backend=self.backend)
                    # the w_scale and h_scale has minor difference
                    # a real fix should be done in the mmcv.imrescale in the future
                    new_h, new_w = img.shape[:2]
                    h, w = results[key][idx].shape[:2]
                    w_scale = new_w / w
                    h_scale = new_h / h
                else:
                    img, w_scale, h_scale = mmcv.imresize(
                        results[key][idx],
                        results['scale'],
                        return_scale=True,
                        backend=self.backend)
                results[key][idx] = img

            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
            results['img_shape'] = img.shape
            # in case that there is no padding
            results['pad_shape'] = img.shape
            results['scale_factor'] = scale_factor
            results['keep_ratio'] = self.keep_ratio
            if 'valid_shape' in results:
                scaling = np.array([[w_scale, h_scale]])
                results['valid_shape'] = results['valid_shape'] * scaling

    def _resize_bboxes(self, results):
        """Resize bounding boxes with ``results['scale_factor']``."""
        for key in results.get('bbox_fields', []):
            bboxes = results[key] * results['scale_factor']
            if self.bbox_clip_border:
                img_shape = results['img_shape']
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            results[key] = bboxes

    def _resize_centers(self, results):
        centers = results['gt_img_centers_2d']
        centers[:, :2] = centers[:, :2] * results['scale_factor'][:2]
        img_shape = results['img_shape']
        centers[:, 0] = np.clip(centers[:, 0], 0, img_shape[1])
        centers[:, 1] = np.clip(centers[:, 1], 0, img_shape[0])
        results['gt_img_centers_2d'] = centers

    def _resize_masks(self, results):
        """Resize masks with ``results['scale']``"""
        for key in results.get('mask_fields', []):
            if results[key] is None:
                continue
            if self.keep_ratio:
                results[key] = results[key].rescale(results['scale'])
            else:
                results[key] = results[key].resize(results['img_shape'][:2])

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        for key in results.get('seg_fields', []):
            if self.keep_ratio:
                gt_seg = mmcv.imrescale(
                    results[key],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            else:
                gt_seg = mmcv.imresize(
                    results[key],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            results['gt_semantic_seg'] = gt_seg

    def _resize_camera(self, results):
        scale_factor = results['scale_factor']
        w_scale = scale_factor[0]
        h_scale = scale_factor[1]
        scaling_matrix = np.array([
            [w_scale, 0, 0],
            [0, h_scale, 0],
            [0, 0, 1]
        ])
        for i in range(len(results['cam_intrinsic'])):
            results['cam_intrinsic'][i] = scaling_matrix @ results['cam_intrinsic'][i]

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor', \
                'keep_ratio' keys are added into result dict.
        """

        if 'scale' not in results:
            if 'scale_factor' in results:
                img_shape = results['img'][0].shape[:2]
                scale_factor = results['scale_factor']
                assert isinstance(scale_factor, float)
                results['scale'] = tuple(
                    [int(x * scale_factor) for x in img_shape][::-1])
            else:
                self._random_scale(results)
        else:
            if not self.override:
                assert 'scale_factor' not in results, (
                    'scale and scale_factor cannot be both set.')
            else:
                results.pop('scale')
                if 'scale_factor' in results:
                    results.pop('scale_factor')
                self._random_scale(results)

        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        self._resize_seg(results)
        if 'gt_img_centers_2d' in results:
            self._resize_centers(results)

        if 'cam_intrinsic' in results:
            self._resize_camera(results)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'multiscale_mode={self.multiscale_mode}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'keep_ratio={self.keep_ratio}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str


@PIPELINES.register_module()
class MyNormalize(object):
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        for key in results.get('img_fields', ['img']):
            for idx in range(len(results['img'])):
                results[key][idx] = mmcv.imnormalize(results[key][idx], self.mean, self.std,
                                                     self.to_rgb)
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class MyPad(object):
    """Pad the image & mask.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        for key in results.get('img_fields', ['img']):
            if self.size is not None:
                padded_img = mmcv.impad(
                    results[key], shape=self.size, pad_val=self.pad_val)
            elif self.size_divisor is not None:
                for idx in range(len(results[key])):
                    padded_img = mmcv.impad_to_multiple(
                        results[key][idx], self.size_divisor, pad_val=self.pad_val)
                    results[key][idx] = padded_img
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_masks(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        pad_shape = results['pad_shape'][:2]
        for key in results.get('mask_fields', []):
            results[key] = results[key].pad(pad_shape, pad_val=self.pad_val)

    def _pad_seg(self, results):
        """Pad semantic segmentation map according to
        ``results['pad_shape']``."""
        for key in results.get('seg_fields', []):
            results[key] = mmcv.impad(
                results[key], shape=results['pad_shape'][:2])

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        self._pad_masks(results)
        self._pad_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class LoadMultiViewImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, img_scale=None, color_type='unchanged'):
        self.to_float32 = to_float32
        self.img_scale = img_scale
        self.color_type = color_type

    def pad(self, img):
        # to pad the 5 input images into a same size (for Waymo)
        if img.shape[0] != self.img_scale[0]:
            img = np.concatenate([img, np.zeros_like(img[0:1280-886,:])], axis=0)
        return img

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_filename']
        if self.img_scale is None:
            img = np.stack(
                [mmcv.imread(name, self.color_type) for name in filename], axis=-1)
        else:
            img = np.stack(
                [self.pad(mmcv.imread(name, self.color_type)) for name in filename], axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        # results['scale_factor'] = [1.0, 1.0]
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        results['img_fields'] = ['img']
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return "{} (to_float32={}, color_type='{}')".format(
            self.__class__.__name__, self.to_float32, self.color_type)


@PIPELINES.register_module()
class LoadPointsFromMultiSweeps(object):
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(self,
                 sweeps_num=10,
                 load_dim=5,
                 use_dim=[0, 1, 2, 4],
                 file_client_args=dict(backend='disk'),
                 pad_empty_sweeps=False,
                 remove_close=False,
                 test_mode=False):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray): Sweep points.
            radius (float): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.

                - points (np.ndarray): Multi-sweep point cloud arrays.
        """
        points = results['points']
        points.tensor[:, 4] = 0
        sweep_points_list = [points]
        ts = results['timestamp']
        if self.pad_empty_sweeps and len(results['sweeps']) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            if len(results['sweeps']) <= self.sweeps_num:
                choices = np.arange(len(results['sweeps']))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                choices = np.random.choice(
                    len(results['sweeps']), self.sweeps_num, replace=False)
            for idx in choices:
                sweep = results['sweeps'][idx]
                points_sweep = self._load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)
                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                sweep_ts = sweep['timestamp'] / 1e6
                points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                    'sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']
                points_sweep[:, 4] = ts - sweep_ts
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)
        points = points[:, self.use_dim]
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'


@PIPELINES.register_module()
class PointSegClassMapping(object):
    """Map original semantic class to valid category ids.

    Map valid classes as 0~len(valid_cat_ids)-1 and
    others as len(valid_cat_ids).

    Args:
        valid_cat_ids (tuple[int]): A tuple of valid category.
    """

    def __init__(self, valid_cat_ids):
        self.valid_cat_ids = valid_cat_ids

    def __call__(self, results):
        """Call function to map original semantic class to valid category ids.

        Args:
            results (dict): Result dict containing point semantic masks.

        Returns:
            dict: The result dict containing the mapped category ids. \
                Updated key and value are described below.

                - pts_semantic_mask (np.ndarray): Mapped semantic masks.
        """
        assert 'pts_semantic_mask' in results
        pts_semantic_mask = results['pts_semantic_mask']
        neg_cls = len(self.valid_cat_ids)

        for i in range(pts_semantic_mask.shape[0]):
            if pts_semantic_mask[i] in self.valid_cat_ids:
                converted_id = self.valid_cat_ids.index(pts_semantic_mask[i])
                pts_semantic_mask[i] = converted_id
            else:
                pts_semantic_mask[i] = neg_cls

        results['pts_semantic_mask'] = pts_semantic_mask
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += '(valid_cat_ids={})'.format(self.valid_cat_ids)
        return repr_str


@PIPELINES.register_module()
class NormalizePointsColor(object):
    """Normalize color of points.

    Args:
        color_mean (list[float]): Mean color of the point cloud.
    """

    def __init__(self, color_mean):
        self.color_mean = color_mean

    def __call__(self, results):
        """Call function to normalize color of points.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the normalized points. \
                Updated key and value are described below.

                - points (np.ndarray): Points after color normalization.
        """
        points = results['points']
        assert points.shape[1] >= 6, \
            f'Expect points have channel >=6, got {points.shape[1]}'
        points[:, 3:6] = points[:, 3:6] - np.array(self.color_mean) / 256.0
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += '(color_mean={})'.format(self.color_mean)
        return repr_str


@PIPELINES.register_module()
class LoadPointsFromFile(object):
    """Load Points From File.

    Load sunrgbd and scannet points from file.

    Args:
        load_dim (int): The dimension of the loaded points.
            Defaults to 6.
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        use_dim (list[int]): Which dimensions of the points to be used.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): Whether to use shifted height. Defaults to False.
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(self,
                 coord_type,
                 load_dim=6,
                 use_dim=[0, 1, 2],
                 shift_height=False,
                 file_client_args=dict(backend='disk')):
        self.shift_height = shift_height
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (np.ndarray): Point clouds data.
        """
        pts_filename = results['pts_filename']
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate([points, np.expand_dims(height, 1)], 1)
            attribute_dims = dict(height=3)

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += 'shift_height={}, '.format(self.shift_height)
        repr_str += 'file_client_args={}), '.format(self.file_client_args)
        repr_str += 'load_dim={}, '.format(self.load_dim)
        repr_str += 'use_dim={})'.format(self.use_dim)
        return repr_str


@PIPELINES.register_module()
class LoadAnnotations3D(LoadAnnotations):
    """Load Annotations3D.

    Load instance mask and semantic mask of points and
    encapsulate the items into related fields.

    Args:
        with_bbox_3d (bool, optional): Whether to load 3D boxes.
            Defaults to True.
        with_label_3d (bool, optional): Whether to load 3D labels.
            Defaults to True.
        with_mask_3d (bool, optional): Whether to load 3D instance masks.
            for points. Defaults to False.
        with_seg_3d (bool, optional): Whether to load 3D semantic masks.
            for points. Defaults to False.
        with_bbox (bool, optional): Whether to load 2D boxes.
            Defaults to False.
        with_label (bool, optional): Whether to load 2D labels.
            Defaults to False.
        with_mask (bool, optional): Whether to load 2D instance masks.
            Defaults to False.
        with_seg (bool, optional): Whether to load 2D semantic masks.
            Defaults to False.
        poly2mask (bool, optional): Whether to convert polygon annotations
            to bitmasks. Defaults to True.
        seg_3d_dtype (dtype, optional): Dtype of 3D semantic masks.
            Defaults to int64
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details.
    """

    def __init__(self,
                 with_bbox_3d=True,
                 with_label_3d=True,
                 with_mask_3d=False,
                 with_seg_3d=False,
                 with_bbox=False,
                 with_label=False,
                 with_mask=False,
                 with_seg=False,
                 poly2mask=True,
                 seg_3d_dtype='int',
                 file_client_args=dict(backend='disk')):
        super().__init__(
            with_bbox,
            with_label,
            with_mask,
            with_seg,
            poly2mask,
            file_client_args=file_client_args)
        self.with_bbox_3d = with_bbox_3d
        self.with_label_3d = with_label_3d
        self.with_mask_3d = with_mask_3d
        self.with_seg_3d = with_seg_3d
        self.seg_3d_dtype = seg_3d_dtype

    def _load_bboxes_3d(self, results):
        """Private function to load 3D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box annotations.
        """
        results['gt_bboxes_3d'] = results['ann_info']['gt_bboxes_3d']
        results['bbox3d_fields'].append('gt_bboxes_3d')
        return results

    def _load_labels_3d(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['gt_labels_3d'] = results['ann_info']['gt_labels_3d']
        return results

    def _load_masks_3d(self, results):
        """Private function to load 3D mask annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D mask annotations.
        """
        pts_instance_mask_path = results['ann_info']['pts_instance_mask_path']

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            mask_bytes = self.file_client.get(pts_instance_mask_path)
            pts_instance_mask = np.frombuffer(mask_bytes, dtype=np.int)
        except ConnectionError:
            mmcv.check_file_exist(pts_instance_mask_path)
            pts_instance_mask = np.fromfile(
                pts_instance_mask_path, dtype=np.long)

        results['pts_instance_mask'] = pts_instance_mask
        results['pts_mask_fields'].append('pts_instance_mask')
        return results

    def _load_semantic_seg_3d(self, results):
        """Private function to load 3D semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing the semantic segmentation annotations.
        """
        pts_semantic_mask_path = results['ann_info']['pts_semantic_mask_path']

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            mask_bytes = self.file_client.get(pts_semantic_mask_path)
            # add .copy() to fix read-only bug
            pts_semantic_mask = np.frombuffer(
                mask_bytes, dtype=self.seg_3d_dtype).copy()
        except ConnectionError:
            mmcv.check_file_exist(pts_semantic_mask_path)
            pts_semantic_mask = np.fromfile(
                pts_semantic_mask_path, dtype=np.long)

        results['pts_semantic_mask'] = pts_semantic_mask
        results['pts_seg_fields'].append('pts_semantic_mask')
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box, label, mask and
                semantic segmentation annotations.
        """
        results = super().__call__(results)
        if self.with_bbox_3d:
            results = self._load_bboxes_3d(results)
            if results is None:
                return None
        if self.with_label_3d:
            results = self._load_labels_3d(results)
        if self.with_mask_3d:
            results = self._load_masks_3d(results)
        if self.with_seg_3d:
            results = self._load_semantic_seg_3d(results)

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}with_bbox_3d={self.with_bbox_3d}, '
        repr_str += f'{indent_str}with_label_3d={self.with_label_3d}, '
        repr_str += f'{indent_str}with_mask_3d={self.with_mask_3d}, '
        repr_str += f'{indent_str}with_seg_3d={self.with_seg_3d}, '
        repr_str += f'{indent_str}with_bbox={self.with_bbox}, '
        repr_str += f'{indent_str}with_label={self.with_label}, '
        repr_str += f'{indent_str}with_mask={self.with_mask}, '
        repr_str += f'{indent_str}with_seg={self.with_seg}, '
        repr_str += f'{indent_str}poly2mask={self.poly2mask})'
        return repr_str


@PIPELINES.register_module()
class MyLoadAnnotations3D(LoadAnnotations3D):
    def __init__(self, with_bbox_3d=True, with_label_3d=True, with_mask_3d=False, with_seg_3d=False, with_bbox=False,
            with_label=False, with_mask=False, with_seg=False, poly2mask=True, with_centers=False, with_cam_bbox=False,
            with_visible=False, seg_3d_dtype='int', file_client_args=dict(backend='disk')):
        super().__init__(
            with_bbox_3d=with_bbox_3d,
            with_label_3d=with_label_3d,
            with_mask_3d=with_mask_3d,
            with_seg_3d=with_seg_3d,
            with_bbox=with_bbox,
            with_label=with_label,
            with_mask=with_mask,
            with_seg=with_seg,
            poly2mask=poly2mask,
            seg_3d_dtype=seg_3d_dtype,
            file_client_args=file_client_args)
        self.with_centers = with_centers
        self.with_cam_bbox = with_cam_bbox
        self.with_visible = with_visible

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box, label, mask and
                semantic segmentation annotations.
        """
        results = super().__call__(results)
        if self.with_centers:
            results = self._load_centers_2d(results)

        if self.with_cam_bbox:
            results = self._load_cam_box(results)

        if self.with_visible:
            results = self._load_visible(results)

        return results

    def _load_centers_2d(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['gt_pts_centers_2d'] = results['ann_info']['pts_centers_2d']
        results['gt_img_centers_2d'] = results['ann_info']['img_centers_2d']

        return results

    def _load_cam_box(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['gt_bboxes_cam_3d'] = results['ann_info']['bboxes_cam_3d']
        results['gt_bboxes_lidar'] = results['ann_info']['bboxes_lidar']

        return results

    def _load_visible(self, results):
        results['gt_visible_3d'] = results['ann_info']['gt_visible_3d']

        return results

@PIPELINES.register_module()
class SparseDepth(object):
    """
    Generate a sparse depth map from the point clouds, the depth map should have the same size with image features
    """
    def __init__(self, scale_factors, depth_mean=14.41, depth_var=156.89, exp_time=0, depth_sine_L=1, virtual_depth=False):
        self.scale_factors = scale_factors
        self.depth_mean = depth_mean
        self.depth_var = depth_var
        self.exp_time = exp_time
        self.depth_sine_L = depth_sine_L
        self.virtual_depth = virtual_depth

    def __call__(self, results):
        all_points = results['points'].tensor
        curr_mask = all_points[:, 4] == 0
        points = all_points[curr_mask]
        points = points[:, :3]
        points_4d = torch.cat([points, torch.ones_like(points[:, :1])], dim=1)
        lidar2cam_rs = results['lidar2cam_r']
        lidar2cam_ts = results['lidar2cam_t']
        cam_intrinsic = results['cam_intrinsic']

        # w_scale = results['scale_factor'][0]
        # h_scale = results['scale_factor'][1]

        # h_shape = int(results['ori_shape'][0] * h_scale)
        # w_shape = int(results['ori_shape'][1] * w_scale)

        depth_features = []
        for view_id in range(len(lidar2cam_rs)):
            if 'valid_shape' in results:
                h_shape = int(results['valid_shape'][view_id, 1])
                w_shape = int(results['valid_shape'][view_id, 0])
            else:
                h_shape = results['pad_shape'][0]
                w_shape = results['pad_shape'][1]

            cam_ext = np.eye(4)
            cam_int = np.eye(4)

            cam_ext[:3, :3] = lidar2cam_rs[view_id]
            cam_ext[:3, 3] = lidar2cam_ts[view_id]
            cam_int[:3, :3] = cam_intrinsic[view_id]

            cam_ext = torch.from_numpy(cam_ext).type_as(points_4d)
            cam_int = torch.from_numpy(cam_int).type_as(points_4d)

            points_4d_view = points_4d @ cam_ext.T
            points_4d_view = points_4d_view @ cam_int.T

            points_2d_view = points_4d_view[:, :2]
            depth = points_4d_view[:, 2]
            depth = torch.clamp(depth, min=1e-4)

            points_2d_view[:, 0] = points_2d_view[:, 0] / depth
            points_2d_view[:, 1] = points_2d_view[:, 1] / depth

            valid_mask = (points_2d_view[:, 0] > 0) & (points_2d_view[:, 0] < w_shape-1) & \
                         (points_2d_view[:, 1] > 0) & (points_2d_view[:, 1] < h_shape-1)

            points_2d_view = points_2d_view[valid_mask]
            depth = depth[valid_mask]

            sort_id = np.argsort(-depth)
            points_2d_view = points_2d_view[sort_id]
            depth = depth[sort_id]

            depth_features_view = []
            w_scale_shape = results['pad_shape'][1] // self.scale_factors[0]
            h_scale_shape = results['pad_shape'][0] // self.scale_factors[0]

            if self.virtual_depth:
                # if 'pcd_scale_factor' in results:
                #     depth = depth / results['pcd_scale_factor']
                if 'img_scale_ratios' in results:
                    depth = depth / results['img_scale_ratios'][view_id]

            for scale in self.scale_factors:
                w_scale_factor = 1.0 / scale
                h_scale_factor = 1.0 / scale

                scale_factor = torch.Tensor([[w_scale_factor, h_scale_factor]])

                depth_feature = torch.zeros((2, h_scale_shape, w_scale_shape))

                points_2d_view_scale = points_2d_view * scale_factor

                cx = points_2d_view_scale[:, 0].long()
                cy = points_2d_view_scale[:, 1].long()

                depth_feature[0, cy, cx] = depth
                depth_feature[1, cy, cx] = 1

                if self.exp_time > 0:
                    zero_inds = depth_feature[1] == 0
                    depth_map = depth_feature[0]
                    depth_map[zero_inds] = 9999
                    for i in range(self.exp_time):
                        depth_feature_new = torch.zeros_like(depth_map) + 9999
                        depth_feature_new[1:] = torch.minimum(depth_feature_new[1:], depth_map[:-1])
                        depth_feature_new[:-1] = torch.minimum(depth_feature_new[:-1], depth_map[1:])
                        depth_feature_new[:, 1:] = torch.minimum(depth_feature_new[:, 1:], depth_map[:, :-1])
                        depth_feature_new[:, :-1] = torch.minimum(depth_feature_new[:, :-1], depth_map[:, 1:])

                        depth_map = torch.where(zero_inds, depth_feature_new, depth_map)
                        zero_inds = depth_map == 9999
                    depth_map[zero_inds] = 0
                    depth_feature[0] = depth_map
                    depth_feature[1, torch.logical_not(zero_inds)] = 1

                depth_features_view.append(depth_feature)
            depth_features_view = torch.stack(depth_features_view, dim=0)  # [num_scale, 2, h_scale_shape, w_scale_shape)
            depth_features.append(depth_features_view)
        depth_features = torch.stack(depth_features, dim=0)  # [num_view, num_scale, 2, h_scale_shape, w_scale_shape)

        if self.depth_sine_L > 1:
            depth_value = depth_features[:, :, 0]  # [num_view, num_scale, h_scale_shape, w_scale_shape)
            depth_value = depth_value / 40 - 1
            depth_sine = []
            for i in range(self.depth_sine_L):
                sin = torch.sin((2**i) * np.pi * depth_value)
                cos = torch.cos((2**i) * np.pi * depth_value)
                sin = sin * depth_features[:, :, 1]
                cos = cos * depth_features[:, :, 1]
                depth_sine.append(sin)
                depth_sine.append(cos)
            depth_features = torch.stack(depth_sine, dim=2)  # [num_view, num_scale, L*2, h_scale_shape, w_scale_shape)
        else:
            depth_features[:, :, 0] = (depth_features[:, :, 0] - self.depth_mean) / np.sqrt(self.depth_var)
            depth_features[:, :, 0] = depth_features[:, :, 0] * depth_features[:, :, 1]
        results['sparse_depth'] = depth_features

        return results

# Full kernels
FULL_KERNEL_3 = np.ones((3, 3), np.uint8)
FULL_KERNEL_5 = np.ones((5, 5), np.uint8)
FULL_KERNEL_7 = np.ones((7, 7), np.uint8)
FULL_KERNEL_9 = np.ones((9, 9), np.uint8)
FULL_KERNEL_31 = np.ones((31, 31), np.uint8)

# 3x3 cross kernel
CROSS_KERNEL_3 = np.asarray(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], dtype=np.uint8)

# 5x5 cross kernel
CROSS_KERNEL_5 = np.asarray(
    [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

# 5x5 diamond kernel
DIAMOND_KERNEL_5 = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

# 7x7 cross kernel
CROSS_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)

# 7x7 diamond kernel
DIAMOND_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)


def fill_in_fast(depth_map, max_depth=100.0, custom_kernel=DIAMOND_KERNEL_5,
                 extrapolate=False, blur_type='bilateral'):
    """Fast, in-place depth completion.
    Args:
        depth_map: projected depths
        max_depth: max depth value for inversion
        custom_kernel: kernel to apply initial dilation
        extrapolate: whether to extrapolate by extending depths to top of
            the frame, and applying a 31x31 full kernel dilation
        blur_type:
            'bilateral' - preserves local structure (recommended)
            'gaussian' - provides lower RMSE
    Returns:
        depth_map: dense depth map
    """

    # Invert
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    # Dilate
    depth_map = cv2.dilate(depth_map, custom_kernel)

    # Hole closing
    # depth_map = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, FULL_KERNEL_3)
    depth_map = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, FULL_KERNEL_5)

    # Fill empty spaces with dilated values
    empty_pixels = (depth_map < 0.1)
    # dilated = cv2.dilate(depth_map, FULL_KERNEL_7)
    dilated = cv2.dilate(depth_map, FULL_KERNEL_5)
    depth_map[empty_pixels] = dilated[empty_pixels]

    # Extend highest pixel to top of image
    if extrapolate:
        top_row_pixels = np.argmax(depth_map > 0.1, axis=0)
        top_pixel_values = depth_map[top_row_pixels, range(depth_map.shape[1])]

        for pixel_col_idx in range(depth_map.shape[1]):
            depth_map[0:top_row_pixels[pixel_col_idx], pixel_col_idx] = \
                top_pixel_values[pixel_col_idx]

        # Large Fill
        empty_pixels = depth_map < 0.1
        dilated = cv2.dilate(depth_map, FULL_KERNEL_31)
        depth_map[empty_pixels] = dilated[empty_pixels]

    # # Median blur
    depth_map = cv2.medianBlur(depth_map, 3)
    # depth_map = cv2.medianBlur(depth_map, 5)

    # Bilateral or Gaussian blur
    if blur_type == 'bilateral':
        # Bilateral blur
        depth_map = cv2.bilateralFilter(depth_map, 5, 1.5, 2.0)
        # depth_map = cv2.bilateralFilter(depth_map, 3, 1.5, 2.0)

    elif blur_type == 'gaussian':
        # Gaussian blur
        valid_pixels = (depth_map > 0.1)
        blurred = cv2.GaussianBlur(depth_map, (5, 5), 0)
        depth_map[valid_pixels] = blurred[valid_pixels]

    # Invert
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    return depth_map


@PIPELINES.register_module()
class SparseDepthCompletion(object):
    """
    Generate a sparse depth map from the point clouds, the depth map should have the same size with image features
    """
    def __init__(self, scale_factors, depth_mean=14.41, depth_var=156.89, virtual_depth=False):
        self.scale_factors = scale_factors
        self.depth_mean = depth_mean
        self.depth_var = depth_var
        self.virtual_depth = virtual_depth

    def __call__(self, results):
        all_points = results['points'].tensor
        curr_mask = all_points[:, 4] == 0
        points = all_points[curr_mask]
        points = points[:, :3]
        points_4d = torch.cat([points, torch.ones_like(points[:, :1])], dim=1)
        lidar2cam_rs = results['lidar2cam_r']
        lidar2cam_ts = results['lidar2cam_t']
        cam_intrinsic = results['cam_intrinsic']

        # w_scale = results['scale_factor'][0]
        # h_scale = results['scale_factor'][1]

        # h_shape = int(results['ori_shape'][0] * h_scale)
        # w_shape = int(results['ori_shape'][1] * w_scale)

        depth_features = []
        sparse_depth_features = []
        for view_id in range(len(lidar2cam_rs)):
            if 'valid_shape' in results:
                h_shape = int(results['valid_shape'][view_id, 1])
                w_shape = int(results['valid_shape'][view_id, 0])
            else:
                h_shape = results['pad_shape'][0]
                w_shape = results['pad_shape'][1]

            cam_ext = np.eye(4)
            cam_int = np.eye(4)

            cam_ext[:3, :3] = lidar2cam_rs[view_id]
            cam_ext[:3, 3] = lidar2cam_ts[view_id]
            cam_int[:3, :3] = cam_intrinsic[view_id]

            cam_ext = torch.from_numpy(cam_ext).type_as(points_4d)
            cam_int = torch.from_numpy(cam_int).type_as(points_4d)

            points_4d_view = points_4d @ cam_ext.T
            points_4d_view = points_4d_view @ cam_int.T

            points_2d_view = points_4d_view[:, :2]
            depth = points_4d_view[:, 2]
            depth = torch.clamp(depth, min=1e-4)

            points_2d_view[:, 0] = points_2d_view[:, 0] / depth
            points_2d_view[:, 1] = points_2d_view[:, 1] / depth

            valid_mask = (points_2d_view[:, 0] > 0) & (points_2d_view[:, 0] < w_shape-1) & \
                         (points_2d_view[:, 1] > 0) & (points_2d_view[:, 1] < h_shape-1)

            points_2d_view = points_2d_view[valid_mask]
            depth = depth[valid_mask]

            sort_id = np.argsort(-depth)
            points_2d_view = points_2d_view[sort_id]
            depth = depth[sort_id]

            depth_features_view = []
            sparse_depth_features_view = []
            w_scale_shape = results['pad_shape'][1] // self.scale_factors[0]
            h_scale_shape = results['pad_shape'][0] // self.scale_factors[0]

            if self.virtual_depth:
                if 'pcd_scale_factor' in results:
                    depth = depth / results['pcd_scale_factor']
                if 'img_scale_ratios' in results:
                    depth = depth / results['img_scale_ratios'][view_id]

            for scale in self.scale_factors:
                w_scale_factor = 1.0 / scale
                h_scale_factor = 1.0 / scale

                scale_factor = torch.Tensor([[w_scale_factor, h_scale_factor]])
                depth_feature = torch.zeros((2, h_scale_shape, w_scale_shape))

                points_2d_view_scale = points_2d_view * scale_factor

                cx = points_2d_view_scale[:, 0].long()
                cy = points_2d_view_scale[:, 1].long()

                depth_feature[0, cy, cx] = depth
                depth_feature[1, cy, cx] = 1

                depth_map = copy.deepcopy(depth_feature[0]).numpy()
                depth_max = max(np.max(depth_map) + 20, 100)
                dense_depth_map = fill_in_fast(depth_map, depth_max, extrapolate=True)

                # scale_depth_map = cv2.resize(depth_map, (0, 0), fx=w_scale_factor, fy=h_scale_factor, interpolation=cv2.INTER_NEAREST)
                dense_depth_feature = torch.from_numpy(dense_depth_map)

                depth_features_view.append(dense_depth_feature[None])
                sparse_depth_features_view.append(depth_feature)

            depth_features_view = torch.stack(depth_features_view, dim=0)  # [num_scale, 1, h_scale_shape, w_scale_shape)
            depth_features.append(depth_features_view)

            sparse_depth_features_view = torch.stack(sparse_depth_features_view, dim=0)  # [num_scale, 2, h_scale_shape, w_scale_shape)
            sparse_depth_features.append(sparse_depth_features_view)
        depth_features = torch.stack(depth_features, dim=0)  # [num_view, num_scale, 1, h_scale_shape, w_scale_shape)
        sparse_depth_features = torch.stack(sparse_depth_features, dim=0)  # [num_view, num_scale, 2, h_scale_shape, w_scale_shape)

        # depth_mask = depth_features[:, :, 0] > 0

        sparse_depth_features[:, :, 0] = (sparse_depth_features[:, :, 0] - self.depth_mean) / np.sqrt(self.depth_var)
        sparse_depth_features[:, :, 0] = sparse_depth_features[:, :, 0] * sparse_depth_features[:, :, 1]

        results['sparse_depth'] = torch.cat([sparse_depth_features, depth_features], dim=2)

        return results