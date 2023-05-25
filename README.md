# SparseFusion: Fusing Multi-Modal Sparse Representations for Multi-Sensor 3D Object Detection
![video](video.gif)

## Abstract
We propose SparseFusion, a novel multi-sensor 3D detection method that exclusively uses sparse candidates and sparse representations. Specifically, SparseFusion utilizes the outputs of parallel detectors in the LiDAR and camera modalities as sparse candidates for fusion. We transform the camera candidates into the LiDAR coordinate space by disentangling the object representations. Then, we can fuse the multi-modality candidates in a unified 3D space by a lightweight self-attention module. To mitigate negative transfer between modalities, we propose novel semantic and geometric cross-modality transfer modules that are applied prior to the modality-specific detectors. SparseFusion achieves state-of-the-art performance on the nuScenes benchmark while also running at the fastest speed.

[[paper link]](https://arxiv.org/abs/2304.14340)


## Overview
![teaser](teaser.png)
Compared to existing fusion algorithms, SparseFusion achieves state-of-the-art performance as well as the fastest inference speed on nuScenes test set. †: Official [repository](https://github.com/zehuichen123/AutoAlignV2) of AutoAlignV2 uses flip as test-time augmentation. ‡: We use BEVFusion-base results in the official [repository](https://github.com/mit-han-lab/bevfusion) of BEVFusion to match the input resolutions of other methods. $\S:$ Swin-T is adopted as image backbone.
## NuScene Performance
We do not use any test-time augmentations or model ensembles to get these results. We have released the configure files and pretrained checkpoints to reproduce our results.
#### Validation Set

| Image Backbone | Point Cloud Backbone | mAP | NDS | Link |
| --------- | ------ | ------ | --------- | --------- |
| ResNet50  | VoxelNet             | 70.5 | 72.8 | [config](configs/sparsefusion_nusc_voxel_LC_r50.py)/[ckpt](https://drive.google.com/file/d/1NZIrg7s-VwxkwuPHTTWSQQO7T7IILBGC/view?usp=share_link) |
| Swin-T  | VoxelNet             | 71.0 | 73.1 | [config](configs/sparsefusion_nusc_voxel_LC_SwinT.py)/[ckpt](https://drive.google.com/file/d/1dAhOKtbLd1e3I5jwk_3E1gzbl61P24qy/view?usp=share_link) |

#### Test Set

| Image Backbone | Point Cloud Backbone | mAP  | NDS |
| --------- | ------ | ------ | --------- |
| ResNet50  | VoxelNet             | 72.0 | 73.8 |

## Usage 

#### Installation
+ We test our code on an environment with CUDA 11.5, python 3.7, PyTorch 1.7.1, TorchVision 0.8.2, NumPy 1.20.0, and numba 0.48.0.

+ We use `mmdet3d==0.11.0, mmdet==2.10.0, mmcv==1.2.7 ` for our code. Please refer to the [official instruction](https://mmdetection3d.readthedocs.io/en/v0.11.0/getting_started.html#install-mmdetection3d) of mmdet3d for installation. 

+ We use `spconv==2.3.3`. Please follow the [official instruction](https://github.com/traveller59/spconv) to install it based on your CUDA version.

  ```
  pip install spconv-cuxxx 
  # e.g. pip install spconv-cu114	
  ```

+ You also need to install the deformable attention module with the following command.

  ```
  pip install ./mmdet3d/models/utils/ops
  ```

#### Data Preparation

Download nuScenes full dataset from the [official website](https://www.nuscenes.org/download). You should have a folder structure like this:

```
SparseFusion
├── mmdet3d
├── tools
├── configs
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
```

Then, you can select  **either** of the two ways to preprocess the data.

1. Run the following two commands sequentially. 

   ```
   python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
   python tools/combine_view_info.py
   ```

2. Alternatively, you may directly download our preprocessed data from [Google Drive](https://drive.google.com/drive/folders/1L5lvLsNWBA0vfTlNSMa4OXXBLoZgJbg4?usp=share_link), and put these files in `data/nuscenes`.

#### Initial Weights

Please download the [initial weights](https://drive.google.com/drive/folders/1wmYBi3PBprdcegF843AU-22q2OwDgoZk?usp=share_link) for model training, and put them in `checkpoints/`.

#### Train & Test

In our default setting, we train the model with 4 GPUs.

```
# training
bash tools/dist_train.sh configs/sparsefusion_nusc_voxel_LC_r50.py 4 --work-dir work_dirs/sparsefusion_nusc_voxel_LC_r50

# test
bash tools/dist_test.sh configs/sparsefusion_nusc_voxel_LC_r50.py ${CHECKPOINT_FILE} 4 --eval=bbox
```

## Acknowledgments

We sincerely thank the authors of [mmdetetcion3d](https://github.com/open-mmlab/mmdetection3d), [TransFusion](https://github.com/XuyangBai/TransFusion), [BEVFusion](https://github.com/mit-han-lab/bevfusion), [MSMDFusion](https://github.com/SxJyJay/MSMDFusion), and [DeepInteraction](https://github.com/fudan-zvg/DeepInteraction) for providing their codes or pretrained weights.

## Reference

If you find our work useful, please consider citing the following paper:

```
@article{xie2023sparsefusion,
  title={SparseFusion: Fusing Multi-Modal Sparse Representations for Multi-Sensor 3D Object Detection},
  author={Xie, Yichen and Xu, Chenfeng and Rakotosaona, Marie-Julie and Rim, Patrick and Tombari, Federico and Keutzer, Kurt and Tomizuka, Masayoshi and Zhan, Wei},
  journal={arXiv preprint arXiv:2304.14340},
  year={2023}
}
```

