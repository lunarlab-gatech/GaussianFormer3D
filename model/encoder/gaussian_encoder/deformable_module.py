from mmengine.registry import MODELS
from mmengine.model import BaseModule
from mmengine import build_from_cfg
from mmengine.model import xavier_init, constant_init
import torch, torch.nn as nn
import numpy as np
from typing import List, Optional
from ...utils.safe_ops import safe_sigmoid
from ...utils.utils import get_rotation_matrix
from .utils import linear_relu_ln
try:
    from .ops import DeformableAggregationFunction as DAF
except:
    DAF = None


@MODELS.register_module()
class SparseGaussian3DKeyPointsGenerator(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        num_learnable_pts=0,
        fix_scale=None,
        pc_range=None,
        scale_range=None,
        phi_activation='sigmoid',
        xyz_coordinate='polar'
    ):
        super(SparseGaussian3DKeyPointsGenerator, self).__init__()
        self.embed_dims = embed_dims
        self.num_learnable_pts = num_learnable_pts
        if fix_scale is None:
            fix_scale = ((0.0, 0.0, 0.0),)
        self.fix_scale = np.array(fix_scale)
        self.num_pts = len(self.fix_scale) + num_learnable_pts
        if num_learnable_pts > 0:
            self.learnable_fc = nn.Linear(self.embed_dims, num_learnable_pts * 3)

        self.pc_range = pc_range
        self.scale_range = scale_range
        self.phi_activation = phi_activation
        self.xyz_coordinate = xyz_coordinate

    def init_weight(self):
        if self.num_learnable_pts > 0:
            xavier_init(self.learnable_fc, distribution="uniform", bias=0.0)

    def forward(
        self,
        anchor,
        instance_feature=None,
    ):
        bs, num_anchor = anchor.shape[:2]
        fix_scale = anchor.new_tensor(self.fix_scale)
        scale = fix_scale[None, None].tile([bs, num_anchor, 1, 1])
        if self.num_learnable_pts > 0 and instance_feature is not None:
            learnable_scale = (
                safe_sigmoid(self.learnable_fc(instance_feature)
                .reshape(bs, num_anchor, self.num_learnable_pts, 3))
                - 0.5 # since safe_sigmoid is in [0, 1], we have to account for those points located on the negative side
            )
            scale = torch.cat([scale, learnable_scale], dim=-2) # [bs, num_anchor, num_pts, 3] where num_pts = len(fix_scale) + num_learnable_pts
        
        gs_scales = safe_sigmoid(anchor[..., None, 3:6]) # [bs, num_anchor, 1, 3]
        gs_scales = self.scale_range[0] + (self.scale_range[1] - self.scale_range[0]) * gs_scales

        key_points = scale * gs_scales # [bs, num_anchor, num_pts, 3]
        rots = anchor[..., 6:10] # [bs, num_anchor, 4]
        rotation_mat = get_rotation_matrix(rots).transpose(-1, -2) # [bs, num_anchor, 3, 3]
        
        key_points = torch.matmul(
            rotation_mat[:, :, None], key_points[..., None]
        ).squeeze(-1) # [bs, num_anchor, num_pts, 3, 1] -> [bs, num_anchor, num_pts, 3]

        if self.phi_activation == 'sigmoid':
            xyz = safe_sigmoid(anchor[..., :3]) # xyz is from [-9.21, 9.21] to [0, 1]
        elif self.phi_activation == 'loop':
            xy = safe_sigmoid(anchor[..., :2])
            z = torch.remainder(anchor[..., 2:3], 1.0)
            xyz = torch.cat([xy, z], dim=-1)
        else:
            raise NotImplementedError
        
        if self.xyz_coordinate == 'polar': # convert to polar coordinate
            rrr = xyz[..., 0] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            theta = xyz[..., 1] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            phi = xyz[..., 2] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            xxx = rrr * torch.sin(theta) * torch.cos(phi)
            yyy = rrr * torch.sin(theta) * torch.sin(phi)
            zzz = rrr * torch.cos(theta)
        else: # convert to cartesian coordinate (default)
            xxx = xyz[..., 0] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            yyy = xyz[..., 1] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            zzz = xyz[..., 2] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
        xyz = torch.stack([xxx, yyy, zzz], dim=-1) # [bs, num_anchor, 3] in LiDAR coordinate system
        
        # xyz is the original anchor point in LiDAR coordinate system
        # key_points are the offset from the anchor point in LiDAR coordinate system
        key_points = key_points + xyz.unsqueeze(2) # [bs, num_anchor, num_pts, 3] + [bs, num_anchor, 1, 3] = [bs, num_anchor, num_pts, 3]
        return key_points # in the LiDAR coordinate system, containing z values


@MODELS.register_module()
class DeformableFeatureAggregation(BaseModule):
    def __init__(
        self,
        embed_dims: int = 256,
        num_groups: int = 8,
        num_levels: int = 4,
        num_cams: int = 6,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        kps_generator: dict = None,
        use_deformable_func=False,
        use_camera_embed=False,
        residual_mode="add",
    ):
        super(DeformableFeatureAggregation, self).__init__()
        if embed_dims % num_groups != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_groups, "
                f"but got {embed_dims} and {num_groups}"
            )
        self.group_dims = int(embed_dims / num_groups)
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_groups = num_groups
        self.num_cams = num_cams
        self.use_deformable_func = use_deformable_func and DAF is not None
        self.attn_drop = attn_drop
        self.residual_mode = residual_mode
        self.proj_drop = nn.Dropout(proj_drop)
        kps_generator["embed_dims"] = embed_dims
        self.kps_generator = build_from_cfg(kps_generator, MODELS)
        self.num_pts = self.kps_generator.num_pts # containing the number of fixed points and learnable points, 7 + 2 = 9
        self.output_proj = nn.Linear(embed_dims, embed_dims)

        if use_camera_embed:
            self.camera_encoder = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2, 12) # input dim = 12
            )
            self.weights_fc = nn.Linear(
                embed_dims, num_groups * num_levels * self.num_pts
            )
        else:
            self.camera_encoder = None
            self.weights_fc = nn.Linear(
                embed_dims, num_groups * num_cams * num_levels * self.num_pts
            )

    def init_weight(self):
        constant_init(self.weights_fc, val=0.0, bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        feature_maps: List[torch.Tensor],
        metas: dict,
        anchor_encoder=None,
        **kwargs: dict,
    ):
        bs, num_anchor = instance_feature.shape[:2]
        key_points = self.kps_generator(anchor, instance_feature) # key_points in LiDAR coordinate system, containing z values, shape of [bs, num_anchor, num_pts, 3]
        temp_key_points_list = (
            feature_queue
        ) = meta_queue = temp_anchor_embeds = []
        if self.use_deformable_func:
            feature_maps = DAF.feature_maps_format(feature_maps) # list of three elements

        for (
            temp_feature_maps,
            temp_metas,
            temp_key_points,
            temp_anchor_embed,
        ) in zip(
            feature_queue[::-1] + [feature_maps],
            meta_queue[::-1] + [metas],
            temp_key_points_list[::-1] + [key_points],
            temp_anchor_embeds[::-1] + [anchor_embed],
        ):
            weights = self._get_weights(
                instance_feature, temp_anchor_embed, metas
            ) # [bs, num_anchor, num_cams, num_levels, num_pts, num_groups]
            if self.use_deformable_func:
                weights = (
                    weights.permute(0, 1, 4, 2, 3, 5)
                    .contiguous()
                    .reshape(
                        bs,
                        num_anchor * self.num_pts,
                        self.num_cams,
                        self.num_levels,
                        self.num_groups,
                    )
                ) # [bs, num_anchor * num_pts, num_cams, num_levels, num_groups]
                points_2d = (
                    self.project_points(
                        temp_key_points,
                        temp_metas["projection_mat"],
                        temp_metas.get("image_wh"),
                    )
                    .permute(0, 2, 3, 1, 4)
                    .reshape(bs, num_anchor * self.num_pts, self.num_cams, 2)
                ) # [bs, num_anchor * num_pts, num_cams, 2]
                # temp_feature_maps: list of 3 elements.
                # mc_ms_feat: (bs, num_cams, H0*W0+H1*W1+H2*W2+H3*W3, channel)
                # spatial_shape: list of element with shape (Hi, Wi), after loop, we have 4 scales [216, 400], [108, 200], [54, 100], [27, 50]
                # scale_start_index: list of element with shape [0, H0*W0, H0*W0+H1*W1, H0*W0+H1*W1+H2*W2, H0*W0+H1*W1+H2*W2+H3*W3]
                # points_2d: (bs, num_anchor * num_pts, num_cams, 2)
                # weights: (bs, num_anchor * num_pts, num_cams, num_levels, num_groups)
                temp_features_next = DAF.apply(
                    *temp_feature_maps, points_2d, weights
                ).reshape(bs, num_anchor, self.num_pts, self.embed_dims) # [bs, num_anchor, num_pts, embed_dims], already aggregate information from different views and different levels
            else:
                temp_features_next = self.feature_sampling(
                    temp_feature_maps,
                    temp_key_points,
                    temp_metas["projection_mat"],
                    temp_metas.get("image_wh"),
                )
                temp_features_next = self.multi_view_level_fusion(
                    temp_features_next, weights
                )

            features = temp_features_next

        features = features.sum(dim=2)  # fuse multi-point features, [bs, num_anchor, embed_dims], sum of all sampled points, now we already aggregate information from different views and different points
        output = self.proj_drop(self.output_proj(features)) # [bs, num_anchor, embed_dims]
        if self.residual_mode == "add":
            output = output + instance_feature
        elif self.residual_mode == "cat":
            output = torch.cat([output, instance_feature], dim=-1) # [bs, num_anchor, 2 * embed_dims]
        return output

    def _get_weights(self, instance_feature, anchor_embed, metas=None):
        bs, num_anchor = instance_feature.shape[:2]
        feature = instance_feature + anchor_embed # shape of [bs, num_anchor, embed_dims]
        if self.camera_encoder is not None:
            # [bs, num_cams, 3, 4] -> [bs, num_cams, 3 * 4] -> [bs, num_cams, embed_dims]
            camera_embed = self.camera_encoder(
                metas["projection_mat"][:, :, :3].reshape(
                    bs, self.num_cams, -1
                )
            ) # shape of [bs, num_cams, embed_dims]
            feature = feature[:, :, None] + camera_embed[:, None] # [bs, num_anchor, 1, embed_dims] + [bs, 1, num_cams, embed_dims] = [bs, num_anchor, num_cams, embed_dims]
        weights = (
            self.weights_fc(feature) # [bs, num_anchor, num_cams, embed_dims] -> [bs, num_anchor, num_cams, num_groups * num_levels * num_pts]
            .reshape(bs, num_anchor, -1, self.num_groups)# [bs, num_anchor, num_cams, num_groups * num_levels * num_pts] -> [bs, num_anchor, num_cams * num_levels * num_pts, num_groups]
            .softmax(dim=-2)
            .reshape(
                bs,
                num_anchor,
                self.num_cams,
                self.num_levels,
                self.num_pts,
                self.num_groups,
            )
        ) # [bs, num_anchor, num_cams * num_levels * num_pts, num_groups] -> [bs, num_anchor, num_cams, num_levels, num_pts, num_groups]
        # weights here are softmax over all cameras, levels and points.
        if self.training and self.attn_drop > 0:
            mask = torch.rand(
                bs, num_anchor, self.num_cams, 1, self.num_pts, 1
            )
            mask = mask.to(device=weights.device, dtype=weights.dtype)
            weights = ((mask > self.attn_drop) * weights) / (
                1 - self.attn_drop
            )
        return weights

    @staticmethod
    def project_points(key_points, projection_mat, image_wh=None):
        bs, num_anchor, num_pts = key_points.shape[:3] # [bs, num_anchor, num_pts, 3] in lidar coordinate

        pts_extend = torch.cat(
            [key_points, torch.ones_like(key_points[..., :1])], dim=-1
        ) # [bs, num_anchor, num_pts, 4], xyz1
        # projection_mat[:, :, None, None]: [bs, num_cams, 4, 4] -> [bs, num_cams, 1, 1, 4, 4], lidar to image projection matrix
        # pts_extend[:, None, ..., None]: [bs, num_anchor, num_pts, 4] -> [bs, 1, num_anchor, num_pts, 4, 1]
        # torch.matmul(...).squeeze(-1): [bs, num_cams, 1, 1, 4, 4] * [bs, 1, num_anchor, num_pts, 4, 1] -> [bs, num_cams, num_anchor, num_pts, 4, 1] -> [bs, num_cams, num_anchor, num_pts, 4]
        points_2d = torch.matmul(
            projection_mat[:, :, None, None], pts_extend[:, None, ..., None]
        ).squeeze(-1) # for dfa3d, we probably don't need to divide by the last dimension of points_2d, keep the original dimension of (u, v, d)
        # but have to convert the third dimension to depth bin like encoder.py in dfa3d
        points_2d = points_2d[..., :2] / torch.clamp(
            points_2d[..., 2:3], min=1e-5
        ) # [bs, num_cams, num_anchor, num_pts, 2]
        if image_wh is not None:
            points_2d = points_2d / image_wh[:, :, None, None] # [bs, num_cams, num_anchor, num_pts, 2]
        return points_2d

    @staticmethod
    def feature_sampling(
        feature_maps: List[torch.Tensor],
        key_points: torch.Tensor,
        projection_mat: torch.Tensor,
        image_wh: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_levels = len(feature_maps)
        num_cams = feature_maps[0].shape[1]
        bs, num_anchor, num_pts = key_points.shape[:3]

        points_2d = DeformableFeatureAggregation.project_points(
            key_points, projection_mat, image_wh
        )
        points_2d = points_2d * 2 - 1
        points_2d = points_2d.flatten(end_dim=1)

        features = []
        for fm in feature_maps:
            features.append(
                torch.nn.functional.grid_sample(
                    fm.flatten(end_dim=1), points_2d
                )
            )
        features = torch.stack(features, dim=1)
        features = features.reshape(
            bs, num_cams, num_levels, -1, num_anchor, num_pts
        ).permute(
            0, 4, 1, 2, 5, 3
        )  # bs, num_anchor, num_cams, num_levels, num_pts, embed_dims

        return features

    def multi_view_level_fusion(
        self,
        features: torch.Tensor,
        weights: torch.Tensor,
    ):
        # features: [bs, num_anchor, num_cams, num_levels, num_pts, embed_dims]
        # weights: [bs, num_anchor, num_cams, num_levels, num_pts, num_groups]
        # weights[..., None]: # [bs, num_anchor, num_cams, num_levels, num_pts, num_groups, 1]
        # after reshape: [bs, num_anchor, num_cams, num_levels, num_pts, num_groups, group_dims]
        bs, num_anchor = weights.shape[:2]
        features = weights[..., None] * features.reshape(
            features.shape[:-1] + (self.num_groups, self.group_dims)
        ) # [bs, num_anchor, num_cams, num_levels, num_pts, num_groups, group_dims]
        features = features.sum(dim=2).sum(dim=2) # first sum over num_cams, then sum over num_levels
        features = features.reshape(
            bs, num_anchor, self.num_pts, self.embed_dims
        )
        return features
