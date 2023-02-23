import torch

from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS


@BBOX_CODERS.register_module()
class CameraBBoxCoder(BaseBBoxCoder):
    def __init__(self, code_size=8):
        self.code_size = code_size

    def encode(self, dst_boxes):
        targets = torch.zeros([dst_boxes.shape[0], self.code_size]).to(dst_boxes.device)
        targets[:, 3] = dst_boxes[:, 3].log()
        targets[:, 4] = dst_boxes[:, 4].log()
        targets[:, 5] = dst_boxes[:, 5].log()
        targets[:, 6] = torch.sin(dst_boxes[:, 6])
        targets[:, 7] = torch.cos(dst_boxes[:, 6])

        targets[:, 0] = dst_boxes[:, 0]
        targets[:, 1] = dst_boxes[:, 1] - 0.5 * dst_boxes[:, 4]
        targets[:, 2] = dst_boxes[:, 2]

        if self.code_size == 10:
            targets[:, 8:10] = dst_boxes[:, 7:]
        return targets

    def decode(self, cls, rot, dim, center, vel):
        """Decode bboxes.

        Args:
            cls (torch.Tensor): Heatmap with the shape of [B, num_cls, num_proposals].
            rot (torch.Tensor): Rotation with the shape of
                [B, 2, num_proposals].
            dim (torch.Tensor): Dim of the boxes with the shape of
                [B, 3, num_proposals].
            center (torch.Tensor): bev center of the boxes with the shape of
                [B, 3, num_proposals]. (in feature map metric)
            vel (torch.Tensor): Velocity with the shape of [B, 2, num_proposals].

        Returns:
            list[dict]: Decoded boxes.
        """
        # class label
        final_preds = cls.max(1, keepdims=False).indices
        final_scores = cls.max(1, keepdims=False).values

        dim[:, 0, :] = dim[:, 0, :].exp()
        dim[:, 1, :] = dim[:, 1, :].exp()
        dim[:, 2, :] = dim[:, 2, :].exp()
        # dim = torch.exp(dim)
        rots, rotc = rot[:, 0:1, :], rot[:, 1:2, :]
        rot = torch.atan2(rots, rotc)

        center = center.clone()
        center[:, 1, :] = center[:, 1, :] + 0.5 * dim[:, 1, :]

        if vel is None:
            final_box_preds = torch.cat([center, dim, rot], dim=1).permute(0, 2, 1)
        else:
            final_box_preds = torch.cat([center, dim, rot, vel], dim=1).permute(0, 2, 1)

        predictions_dicts = []
        for i in range(cls.shape[0]):
            boxes3d = final_box_preds[i]
            scores = final_scores[i]
            labels = final_preds[i]
            predictions_dict = {
                'bboxes': boxes3d,
                'scores': scores,
                'labels': labels
            }
            predictions_dicts.append(predictions_dict)

        return predictions_dicts

    @staticmethod
    def decode_yaw(bbox, centers2d, cam2img):
        bbox[:, 6] = torch.atan2(centers2d[:, 0] - cam2img[0, 2], cam2img[0, 0]) + bbox[:, 6]

        return bbox
