# MODEL ZOO

## Common settings and notes

- The experiments are run with PyTorch 1.7.0, CUDA 10.1 and CUDNN 7.6
- The training is conducted on 8 Telsa V100 GPUs
- For the *fade strategy* proposed by PointAugmenting(disenable the copy-and-paste augmentation for the last 5 epochs), we currently implement this strategy by manually stop training at 15 epoch and resume the training without copy-and-paste augmentation. If you find more elegant ways to implement such strategy, please let we know and we really appreciate it. The fade strategy reduces lots of false positive, improving the mAP remarkably especially for TransFusion-L while having less influence on TransFusion.  

## Pretrained 2D Backbones
- DLA34: Following PointAugmenting, we directly reuse the checkpoints pretrained on monocular 3D detection task provided by [CenterNet]((https://github.com/xingyizhou/CenterTrack/blob/master/readme/MODEL_ZOO.md#monocular-3d-detection-tracking)).
- ResNet50 on instance segmentation:  We acquire the model pretrained on nuImages from [MMDetection3D](https://github.com/open-mmlab/mmdetection3d/blob/v0.12.0/configs/nuimages/README.md).
- ResNet50 on 2D detection: We train a model using the [config](https://github.com/open-mmlab/mmdetection3d/blob/v0.12.0/configs/nuimages/mask_rcnn_r50_fpn_1x_nuim.py) of instance segmentation but remove the mask head.


## nuScenes 3D Detection

All the LiDAR-only models are trained in 20 epochs, the fusion-based models are further trained for 6 epochs from the pretrained LiDAR backbone. We freeze the weight of LiDAR backbone to save GPU memory.

| Model   | Backbone | mAP | NDS  |
|---------|--------|--------|---------|
| [TransFusion-L](configs/transfusion_nusc_pillar_L.py) | PointPillars | 54.51 | 62.66 |
| [TransFusion](configs/transfusion_nusc_pillar_LC.py) | PointPillars | 60.21 | 65.50 |
| [TransFusion-L](configs/transfusion_nusc_voxel_L.py) | VoxelNet | 65.06 | 70.10 |
| [TransFusion](configs/transfusion_nusc_voxel_LC.py) | VoxelNet | 67.49 | 71.28 |

## nuScenes 3D Tracking

We perform tracking-by-detection with the same tracking algorithms proposed by CenterPoint. 

| Model   | Backbone | AMOTA | AMOTP  |
|---------|--------|--------|---------|
| [TransFusion-L](configs/transfusion_nusc_voxel_L.py) | VoxelNet | 0.703 | 0.553 |
| [TransFusion](configs/transfusion_nusc_voxel_LC.py) | VoxelNet | 0.725 | 0.561 |


## nuScenes Leaderboard


### Detection

We use 300 object queries during inference for online submission for a slightly better performance. We do not use any test-time-augmentation and model ensemble.

| Model   | Backbone | Test mAP | Test NDS  | Link  |
|---------|--------|--------|---------|---------|
| TransFusion-L | VoxelNet | 65.52 | 70.23 | [Detection](https://drive.google.com/file/d/1Wk8p2LJEhwfKfhsKzlU9vDBOd0zn38dN/view?usp=sharing)
| TransFusion | VoxelNet | 68.90 | 71.68 | [Detection](https://drive.google.com/file/d/1X7_ig4v5A2vKsiHtUGtgeMN-0RJKsM6W/view?usp=sharing)

### Tracking

| Model | Backbone | Test AMOTA |  Test AMOTP   | Link  |
|---------|--------|--------|---------|---------|
| TranFusion-L| VoxelNet | 0.686 | 0.529 | [Detection](https://drive.google.com/file/d/1Wk8p2LJEhwfKfhsKzlU9vDBOd0zn38dN/view?usp=sharing) / [Tracking](https://drive.google.com/file/d/1pKvRBUsM9h1Xgturd0Ae_bnGt0m_j3hk/view?usp=sharing)| 
| TranFusion| VoxelNet | 0.718 | 0.551 | [Detection](https://drive.google.com/file/d/1X7_ig4v5A2vKsiHtUGtgeMN-0RJKsM6W/view?usp=sharing) / [Tracking](https://drive.google.com/file/d/1EVuS-MAg_HSXUVqMrXEs4-RpZp0p5cfv/view?usp=sharing)| 


 
