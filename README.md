# SparseFusion: Fusing Multi-Modal Sparse Representations for Multi-Sensor 3D Object Detection
![video](video.gif)

## Abstract
We propose SparseFusion, a novel multi-sensor 3D detection method that exclusively uses sparse candidates and sparse representations. Specifically, SparseFusion utilizes the outputs of parallel detectors in the LiDAR and camera modalities as sparse candidates for fusion. We transform the camera candidates into the LiDAR coordinate space by disentangling the object representations. Then, we can fuse the multi-modality candidates in a unified 3D space by a lightweight self-attention module. To mitigate negative transfer between modalities, we propose novel semantic and geometric cross-modality transfer modules that are applied prior to the modality-specific detectors. SparseFusion achieves state-of-the-art performance on the nuScenes benchmark while also running at the fastest speed.

[[paper link]](https://arxiv.org/abs/2304.14340)


## Overview
![teaser](teaser.png)
Compared to existing fusion algorithms, SparseFusion achieves state-of-the-art performance as well as the fastest inference speed on nuScenes test set. †: Official [repository](https://github.com/zehuichen123/AutoAlignV2) of AutoAlignV2 uses flip as test-time augmentation. ‡: We use BEVFusion-base results in the official [repository](https://github.com/mit-han-lab/bevfusion) of BEVFusion to match the input resolutions of other methods. $\S:$ Swin-T is adopted as image backbone.
## nuScene Performance
We do not use any test-time augmentations or model ensembles.
### Validataion Set

| Image Backbone | mAP    | NDS    |
| --------- | ------ | ------ |
| ResNet50  | 70.4 | 72.8 |
| Swin-T  | 71.0 | 73.1 |

### Test Set

| Image Backbone | mAP    | NDS    |
| --------- | ------ | ------ |
| ResNet50  | 72.0 | 73.8 |


