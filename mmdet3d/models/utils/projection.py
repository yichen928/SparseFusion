import torch
import torch.nn as nn
from mmdet3d.models.utils import PositionEmbeddingLearned

class PointProjection(nn.Module):
    def __init__(self, pos_channel, hidden_channel):
        super(PointProjection, self).__init__()
        self.feat_proj = nn.Conv1d(hidden_channel, hidden_channel, kernel_size=1)
        self.pos_embed = nn.Sequential(
            nn.Conv1d(pos_channel, hidden_channel*4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channel*4, hidden_channel, kernel_size=1)
        )
        self.fuse_proj = nn.Sequential(
            nn.Conv1d(hidden_channel, hidden_channel, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channel, hidden_channel, kernel_size=1)
        )

    def forward(self, query_feat, query_pos):
        pos_embed = self.pos_embed(query_pos.permute(0, 2, 1))
        feat_embed = self.feat_proj(query_feat)
        proj_embed = self.fuse_proj(feat_embed + pos_embed)
        return proj_embed

class ImageProjection(nn.Module):
    def __init__(self, pos_channel, hidden_channel):
        super(ImageProjection, self).__init__()
        self.feat_proj = nn.Conv1d(hidden_channel, hidden_channel, kernel_size=1)
        self.pos_proj = nn.Sequential(
            nn.Conv1d(pos_channel, hidden_channel*4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channel*4, hidden_channel, kernel_size=1),
        )
        self.fuse_proj = nn.Sequential(
            nn.Conv1d(hidden_channel, hidden_channel, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channel, hidden_channel, kernel_size=1)
        )

    def forward(self, query_feat, query_pos):
        feat_embed = self.feat_proj(query_feat)
        pos_embed = self.pos_proj(query_pos.permute(0, 2, 1))
        proj_embed = self.fuse_proj(feat_embed + pos_embed)
        return proj_embed


class ProjectionL2Norm(nn.Module):
    def __init__(self, hidden_channel):
        super(ProjectionL2Norm, self).__init__()
        self.hidden_channel = hidden_channel
        self.feat_proj = nn.Conv1d(hidden_channel, hidden_channel, kernel_size=1)

    def forward(self, query_feat):
        query_feat = self.feat_proj(query_feat)
        assert query_feat.shape[1] == self.hidden_channel
        query_feat = query_feat / torch.norm(query_feat, p=2, keepdim=True, dim=1)
        return query_feat

class ProjectionLayerNorm(nn.Module):
    def __init__(self, hidden_channel, norm=True, input_channel=None):
        super(ProjectionLayerNorm, self).__init__()
        if input_channel is None:
            input_channel = hidden_channel
        self.hidden_channel = hidden_channel
        self.feat_proj = nn.Linear(input_channel, hidden_channel)
        self.norm = norm
        if norm:
            self.norm = nn.LayerNorm(hidden_channel)

    def forward(self, query_feat):
        query_feat = query_feat.transpose(2, 1)
        query_feat = self.feat_proj(query_feat)
        if self.norm:
            query_feat = self.norm(query_feat)
        query_feat = query_feat.transpose(2, 1)
        return query_feat

class Projection_wPos(nn.Module):
    def __init__(self, hidden_channel, pos_embed):
        super(Projection_wPos, self).__init__()
        self.hidden_channel = hidden_channel
        self.pos_proj = pos_embed
        self.feat_proj = ProjectionLayerNorm(hidden_channel)

    def forward(self, query_feat, query_pos):
        feat_embed = self.feat_proj(query_feat)
        pos_embed = self.pos_proj(query_pos)
        return feat_embed + pos_embed
