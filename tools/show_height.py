import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')

voxel_height = torch.load('vis/voxel_height.pt')
# voxel_height = voxel_height[:, None]
# voxel_height_pool = F.max_pool2d(voxel_height, kernel_size=3, padding=1, stride=1)
# voxel_height_pool_1 = F.max_pool2d(voxel_height, kernel_size=5, padding=2, stride=1)
#
# voxel_height_pool = voxel_height_pool.squeeze(1)
# voxel_height = voxel_height.squeeze(1)
# print(torch.sum(voxel_height!=-50), torch.sum(voxel_height_pool!=-50), torch.sum(voxel_height_pool_1!=-50))
# voxel_height = torch.where(voxel_height==-50, voxel_height_pool, voxel_height)
voxel_height = voxel_height.detach().cpu().numpy()

batch_size = voxel_height.shape[0]
for i in range(batch_size):
    height = voxel_height[i]
    min_value = np.min(height[height!=-50])
    max_value = np.max(height[height!=-50])

    height[height==-50] = min_value
    print(min_value)
    print(np.min(height))
    plt.imshow(height, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.savefig('vis/images/height_%d.png'%i)
    plt.show()
    plt.close()
