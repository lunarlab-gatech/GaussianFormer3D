import torch, torch.nn as nn
from mmseg.registry import MODELS
from .base_lifter import BaseLifter
from ..utils.safe_ops import safe_inverse_sigmoid, point_cloud_map


@MODELS.register_module()
class GaussianLifter(BaseLifter):
    def __init__(
        self,
        num_anchor, # number of gaussians
        embed_dims,
        anchor_grad=True,
        feat_grad=True,
        phi_activation='sigmoid',
        semantics=False,
        semantic_dim=None,
        include_opa=True,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        
        xyz = torch.rand(num_anchor, 3, dtype=torch.float) # randomly set x y z in range [0, 1]
        if phi_activation == 'sigmoid':
            xyz = safe_inverse_sigmoid(xyz) # map x y z from [0, 1] to [-inf, inf] and safely apply inverse sigmoid
        elif phi_activation == 'loop':
            xyz[:, :2] = safe_inverse_sigmoid(xyz[:, :2])
        else:
            raise NotImplementedError
            
        scale = torch.rand_like(xyz) # ranging between [0, 1)
        scale = safe_inverse_sigmoid(scale) # ranging between (-9.21, 9.21)

        rots = torch.zeros(num_anchor, 4, dtype=torch.float)
        rots[:, 0] = 1

        if include_opa:
            opacity = safe_inverse_sigmoid(0.1 * torch.ones((num_anchor, 1), dtype=torch.float))
        else:
            opacity = torch.ones((num_anchor, 0), dtype=torch.float)

        if semantics:
            assert semantic_dim is not None
        else:
            semantic_dim = 0
        semantic = torch.randn(num_anchor, semantic_dim, dtype=torch.float) # (N, 17)

        anchor = torch.cat([xyz, scale, rots, opacity, semantic], dim=-1) # gaussians initialization, (N, 28)

        self.num_anchor = num_anchor
        self.anchor = nn.Parameter(
            torch.tensor(anchor, dtype=torch.float32),
            requires_grad=anchor_grad,
        )
        self.anchor_init = anchor
        self.instance_feature = nn.Parameter(
            torch.zeros([self.anchor.shape[0], self.embed_dims]), # (N, 128)
            requires_grad=feat_grad,
        )

    def init_weight(self):
        self.anchor.data = self.anchor.data.new_tensor(self.anchor_init)
        if self.instance_feature.requires_grad:
            torch.nn.init.xavier_uniform_(self.instance_feature.data, gain=1)

    def forward(self, ms_img_feats, **kwargs):
        # ms_img_feats: list of multi-scale image features, each element is a tensor of shape (B, BN/B, C, H, W)
        batch_size = ms_img_feats[0].shape[0]
        instance_feature = torch.tile(
            self.instance_feature[None], (batch_size, 1, 1) # (B, N, 128)
        )
        anchor = torch.tile(self.anchor[None], (batch_size, 1, 1)) # (B, N, 28)

        return {
            'rep_features': instance_feature,
            'representation': anchor,
        }


@MODELS.register_module()
class GaussianLifterLiDARPoint(BaseLifter):
    def __init__(
        self,
        num_anchor, # number of gaussians
        embed_dims,
        anchor_grad=True,
        feat_grad=True,
        phi_activation='sigmoid',
        semantics=False,
        semantic_dim=None,
        include_opa=True,
        use_intensity=False,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        
        xyz = torch.rand(num_anchor, 3, dtype=torch.float) # randomly set x y z in range [0, 1]
        if phi_activation == 'sigmoid':
            xyz = safe_inverse_sigmoid(xyz) # map x y z from [0, 1] to [-inf, inf] and safely apply inverse sigmoid
        elif phi_activation == 'loop':
            xyz[:, :2] = safe_inverse_sigmoid(xyz[:, :2])
        else:
            raise NotImplementedError
            
        scale = torch.rand_like(xyz) # ranging between [0, 1)
        scale = safe_inverse_sigmoid(scale) # ranging between (-9.21, 9.21)

        rots = torch.zeros(num_anchor, 4, dtype=torch.float)
        rots[:, 0] = 1

        if include_opa:
            opacity = safe_inverse_sigmoid(0.1 * torch.ones((num_anchor, 1), dtype=torch.float))
        else:
            opacity = torch.ones((num_anchor, 0), dtype=torch.float)

        if semantics:
            assert semantic_dim is not None
        else:
            semantic_dim = 0
        semantic = torch.randn(num_anchor, semantic_dim, dtype=torch.float) # (N, 17)

        anchor = torch.cat([xyz, scale, rots, opacity, semantic], dim=-1) # gaussians initialization, (N, 28)

        self.num_anchor = num_anchor
        self.anchor = nn.Parameter(
            torch.tensor(anchor, dtype=torch.float32),
            requires_grad=anchor_grad,
        )
        self.anchor_init = anchor
        self.instance_feature = nn.Parameter(
            torch.zeros([self.anchor.shape[0], self.embed_dims]), # (N, 128)
            requires_grad=feat_grad,
        )
        self.use_intensity = use_intensity

    def init_weight(self):
        self.anchor.data = self.anchor.data.new_tensor(self.anchor_init)
        if self.instance_feature.requires_grad:
            torch.nn.init.xavier_uniform_(self.instance_feature.data, gain=1)

    def forward(self, ms_img_feats, anchor_points, **kwargs):
        # ms_img_feats: list of multi-scale image features, each element is a tensor of shape (B, BN/B, C, H, W)
        batch_size = ms_img_feats[0].shape[0]
        instance_feature = torch.tile(
            self.instance_feature[None], (batch_size, 1, 1) # (B, N, 128)
        )
        anchor = torch.tile(self.anchor[None], (batch_size, 1, 1)) # (B, N, 28)

        for batch_idx in range(batch_size):
            anchor_points_single = anchor_points[batch_idx] # (N, 4) or (N, 3)
            assert anchor_points_single.shape[0] == self.num_anchor
            anchor_points_coords = anchor_points_single[:, :3] # (N, 3)
            anchor_points_coords_logits = safe_inverse_sigmoid(anchor_points_coords) # (N, 3) and each element is in range [-9.21024, 9.21024]
            anchor[batch_idx][:, :3] = anchor_points_coords_logits
            if self.use_intensity:
                anchor_points_intensity = anchor_points_single[:, 3]
                anchor_points_intensity_logits = safe_inverse_sigmoid(anchor_points_intensity)
                anchor[batch_idx][:, 10] = anchor_points_intensity_logits

        return {
            'rep_features': instance_feature,
            'representation': anchor,
        }


@MODELS.register_module()
class GaussianLifterLiDAR(BaseLifter):
    def __init__(
        self,
        num_anchor, # number of gaussians
        embed_dims,
        anchor_grad=True,
        feat_grad=True,
        phi_activation='sigmoid',
        semantics=False,
        semantic_dim=None,
        include_opa=True,
        use_intensity=True,
        occ_annotation="surroundocc",
    ):
        super().__init__()
        self.embed_dims = embed_dims
        
        xyz = torch.rand(num_anchor, 3, dtype=torch.float) # randomly set x y z in range [0, 1]
        if phi_activation == 'sigmoid':
            xyz = safe_inverse_sigmoid(xyz) # map x y z from [0, 1] to [-9.21024, 9.21024]
        elif phi_activation == 'loop':
            xyz[:, :2] = safe_inverse_sigmoid(xyz[:, :2])
        else:
            raise NotImplementedError
            
        scale = torch.rand_like(xyz)
        scale = safe_inverse_sigmoid(scale)

        rots = torch.zeros(num_anchor, 4, dtype=torch.float)
        rots[:, 0] = 1

        if include_opa:
            opacity = safe_inverse_sigmoid(0.1 * torch.ones((num_anchor, 1), dtype=torch.float))
        else:
            opacity = torch.ones((num_anchor, 0), dtype=torch.float)

        if semantics:
            assert semantic_dim is not None
        else:
            semantic_dim = 0
        semantic = torch.randn(num_anchor, semantic_dim, dtype=torch.float) # (N, 17)

        anchor = torch.cat([xyz, scale, rots, opacity, semantic], dim=-1) # gaussians initialization, (N, 28)

        self.num_anchor = num_anchor
        self.anchor = nn.Parameter(
            torch.tensor(anchor, dtype=torch.float32),
            requires_grad=anchor_grad,
        )
        self.anchor_init = anchor
        self.instance_feature = nn.Parameter(
            torch.zeros([self.anchor.shape[0], self.embed_dims]), # (N, 128)
            requires_grad=feat_grad,
        )
        self.use_intensity = use_intensity
        self.occ_annotation = occ_annotation

    def init_weight(self):
        self.anchor.data = self.anchor.data.new_tensor(self.anchor_init)
        if self.instance_feature.requires_grad:
            torch.nn.init.xavier_uniform_(self.instance_feature.data, gain=1)

    def forward(self, ms_img_feats, voxel_lidar_feats, coors_batch, **kwargs):
        batch_size = ms_img_feats[0].shape[0]
        instance_feature = torch.tile(
            self.instance_feature[None], (batch_size, 1, 1) # (B, N, 128)
        )
        anchor = torch.tile(self.anchor[None], (batch_size, 1, 1)) # (B, N, 28)

        for batch_idx in range(batch_size):
            batch_mask = coors_batch[:, 0] == batch_idx
            voxel_lidar_feats_single = voxel_lidar_feats[batch_mask] # (Ni, 5)
            voxel_map_coords = point_cloud_map(voxel_lidar_feats_single, self.occ_annotation)
            voxel_map_coords_logits = safe_inverse_sigmoid(voxel_map_coords) # (Ni, 3)
            if self.use_intensity:
                voxel_map_intensity = voxel_lidar_feats_single[:, 3] / 255.0 # [0, 1]
                voxel_map_intensity_logits = safe_inverse_sigmoid(voxel_map_intensity) # (Ni, )

            N = anchor.shape[1]
            Ni = voxel_map_coords_logits.shape[0]
            if N > Ni:
                # randomly sample Ni points from N points
                idx = torch.randperm(N)[:Ni]
                anchor[batch_idx][idx][:, :3] = voxel_map_coords_logits
                if self.use_intensity:
                    anchor[batch_idx][idx][:, 10] = voxel_map_intensity_logits
            else:
                # randomly sample N points from Ni points
                idx = torch.randperm(Ni)[:N]
                anchor[batch_idx][:, :3] = voxel_map_coords_logits[idx]
                if self.use_intensity:
                    anchor[batch_idx][:, 10] = voxel_map_intensity_logits[idx]

        return {
            'rep_features': instance_feature,
            'representation': anchor,
        }



