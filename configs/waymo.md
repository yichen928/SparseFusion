# MODEL ZOO

## Common settings and notes

- The experiments are run with PyTorch 1.7.0, CUDA 10.1 and CUDNN 7.6
- The training is conducted on 8 Telsa V100 GPUs

## Waymo 3D Detection

We try a few training schedules for TransFusion-L and list the performance below. The fusion-based models are further trained for 6 epochs from the pretrained LiDAR backbone. We freeze the weight of LiDAR backbone to save GPU memory.

| Model   | Backbone | epoch | Veh_L2 | Ped_L2 | Cyc_L2  | MAPH   |
|---------|--------|--------|---------|---------|---------|---------|
| [TransFusion-L](configs/transfusion_waymo_voxel_L.py) | VoxelNet | 12 | 63.86 | 62.84 | 67.17 | 64.63
| [TransFusion-L](configs/transfusion_waymo_voxel_L.py) | VoxelNet | 24 | 64.54 | 63.39 | 66.43 | 64.78
| [TransFusion-L](configs/transfusion_waymo_voxel_L.py) | VoxelNet | 36 | 65.07 | 63.70 | 65.97 | 64.91
| [TransFusion](configs/transfusion_waymo_voxel_LC.py) | VoxelNet | 36 + 6| 65.11 | 64.02 | 67.40 | 65.51

