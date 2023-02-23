import torch
from torch import nn as nn

from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weighted_loss

@weighted_loss
def laplacian_aleatoric_uncertainty_loss(pred, target):
    '''
    References:
        MonoPair: Monocular 3D Object Detection Using Pairwise Spatial Relationships, CVPR'20
        Geometry and Uncertainty in Deep Learning for Computer Vision, University of Cambridge
    '''

    log_variance = pred[..., 1:]
    pred = pred[..., :1]
    if target.numel() == 0:
        return pred.sum() * 0
    assert pred.size() == target.size()
    assert pred.size() == log_variance.size()

    loss = 1.4142 * torch.exp(-log_variance) * torch.abs(pred - target) + log_variance
    return loss


@LOSSES.register_module()
class LaplaceL1Loss(nn.Module):
    """L1 loss.
    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(LaplaceL1Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = laplacian_aleatoric_uncertainty_loss(pred, target, weight=weight, reduction=reduction, avg_factor=avg_factor)
        loss_bbox = self.loss_weight * loss
        return loss_bbox