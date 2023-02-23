import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')

heatmap_gt = torch.load('vis/heatmap_gt.pt')
heatmap_gt = torch.max(heatmap_gt, dim=0)[0].detach().cpu().numpy()

plt.imshow(heatmap_gt, cmap='plasma', interpolation='nearest')
plt.colorbar()
plt.savefig('vis/images/heatmap_gt.png')
plt.show()
plt.close()
