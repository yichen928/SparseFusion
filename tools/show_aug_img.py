import numpy as np
import torch
import cv2

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

img_tensor = torch.load("vis/img.pt")

bs = img_tensor.shape[0]
view_num = img_tensor.shape[1]
unnormal = UnNormalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])

for i in range(bs):
    for j in range(view_num):
        img_tensor[i, j] = unnormal(img_tensor[i,j])

imgs = img_tensor.cpu().numpy()

for bid in range(bs):
    for view_id in range(view_num):
        view_img = imgs[bid, view_id].transpose(1, 2, 0)
        view_img = cv2.cvtColor(view_img, cv2.COLOR_RGB2BGR)

        cv2.imwrite("vis/img_b%d_v%d.png"%(bid, view_id), view_img)