import torch
import numpy as np

pos_inds_path = "vis/pos_inds_%d.pt"
pos_view_inds_path = "vis/pos_view_inds_%d.pt"

for i in range(4):
    pos_inds = torch.load(pos_inds_path%i, map_location='cpu').numpy().tolist()
    pos_view_inds = torch.load(pos_view_inds_path%i, map_location='cpu').numpy().tolist()

    pos_inds = set(pos_inds)
    pos_view_inds = set(pos_view_inds)

    import pdb
    pdb.set_trace()