import numpy as np
import pickle

scale_factor = 0.47
base = 36

with open("data/nuscenes/nuscenes_infos_train_with_proj2d_wDepth_vis_woRange.pkl", "rb") as file:
    data = pickle.load(file, encoding="bytes")

infos = data['infos']

all_sizes = []
for info in infos:
    box_2d = info['gt_bboxes_2d']
    box_2d = box_2d * scale_factor
    curr_size = np.where(box_2d[:, -1]>box_2d[:, -2], box_2d[:, -1], box_2d[:, -2])
    curr_size = curr_size.tolist()
    all_sizes.extend(curr_size)

all_sizes = np.array(all_sizes)

print("total:", len(all_sizes))
print(np.sum(all_sizes < base))
print(np.sum(np.logical_and(all_sizes > base, all_sizes < base * 2)))
print(np.sum(np.logical_and(all_sizes > base * 2, all_sizes < base * 4)))
print(np.sum(all_sizes > base * 4))
