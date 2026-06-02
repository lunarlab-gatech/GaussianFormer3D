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
import math
# For 2D scenario
# try:
#     from .ops import DeformableAggregationFunction as DAF
# except:
#     DAF = None
# For 3D scenario
from .ops import WeightedMultiScaleDeformableAttnFunction_fp32, \
WeightedMultiScaleDeformableAttnFunction_fp16, MultiScaleDepthScoreSampleFunction_fp32, MultiScaleDepthScoreSampleFunction_fp16, \
MultiScale3DDeformableAttnFunction_fp16, MultiScale3DDeformableAttnFunction_fp32 


@MODELS.register_module()
class SparseGaussian3DKeyPointsGenerator3D(BaseModule):
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
        super(SparseGaussian3DKeyPointsGenerator3D, self).__init__()
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
                - 0.5
            )
            scale = torch.cat([scale, learnable_scale], dim=-2) # [bs, num_anchor, num_pts, 3]
        
        gs_scales = safe_sigmoid(anchor[..., None, 3:6]) # [bs, num_anchor, 1, 3]
        gs_scales = self.scale_range[0] + (self.scale_range[1] - self.scale_range[0]) * gs_scales

        key_points = scale * gs_scales # [bs, num_anchor, num_pts, 3]
        rots = anchor[..., 6:10] # [bs, num_anchor, 4]
        rotation_mat = get_rotation_matrix(rots).transpose(-1, -2) # [bs, num_anchor, 3, 3]
        
        key_points = torch.matmul(
            rotation_mat[:, :, None], key_points[..., None]
        ).squeeze(-1) # [bs, num_anchor, num_pts, 3, 1] -> [bs, num_anchor, num_pts, 3]

        if self.phi_activation == 'sigmoid':
            xyz = safe_sigmoid(anchor[..., :3])
        elif self.phi_activation == 'loop':
            xy = safe_sigmoid(anchor[..., :2])
            z = torch.remainder(anchor[..., 2:3], 1.0)
            xyz = torch.cat([xy, z], dim=-1)
        else:
            raise NotImplementedError
        
        if self.xyz_coordinate == 'polar':
            rrr = xyz[..., 0] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            theta = xyz[..., 1] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            phi = xyz[..., 2] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            xxx = rrr * torch.sin(theta) * torch.cos(phi)
            yyy = rrr * torch.sin(theta) * torch.sin(phi)
            zzz = rrr * torch.cos(theta)
        else:
            xxx = xyz[..., 0] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            yyy = xyz[..., 1] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            zzz = xyz[..., 2] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
        xyz = torch.stack([xxx, yyy, zzz], dim=-1) # [bs, num_anchor, 3]

        key_points = key_points + xyz.unsqueeze(2) # [bs, num_anchor, num_pts, 3]
        return key_points


@MODELS.register_module()
class DeformableFeatureAggregation3D(BaseModule):
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
        d_bound=[2.0, 58, 0.5],
        im2col_step: int = 32,
        use_visibility=False,
        use_sampling_offsets=False,
        num_pts_per_keypoint=2,
        value_projection=False,
    ):
        super(DeformableFeatureAggregation3D, self).__init__()
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
        self.use_deformable_func = use_deformable_func
        self.attn_drop = attn_drop
        self.residual_mode = residual_mode
        self.proj_drop = nn.Dropout(proj_drop)
        kps_generator["embed_dims"] = embed_dims
        self.kps_generator = build_from_cfg(kps_generator, MODELS)
        self.use_sampling_offsets = use_sampling_offsets
        self.num_pts_per_keypoint = num_pts_per_keypoint
        if use_sampling_offsets:
            self.num_pts = self.kps_generator.num_pts * num_pts_per_keypoint
            self.sampling_offsets = nn.Linear(embed_dims, num_groups * num_levels * self.num_pts * 2)
            self.sampling_offsets_depth = nn.Linear(embed_dims, num_groups * num_levels * self.num_pts * 1)
        else:
            self.num_pts = self.kps_generator.num_pts
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.d_bound = d_bound
        self.im2col_step = im2col_step
        self.use_visibility = use_visibility
        self.value_projection = value_projection
        if value_projection:
            self.value_proj = nn.Linear(embed_dims, embed_dims)

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

        # For initialization of sampling offsets and depth
        if self.use_sampling_offsets:
            constant_init(self.sampling_offsets, 0.)
            constant_init(self.sampling_offsets_depth, 0.)
            thetas = torch.arange(
                self.num_groups,
                dtype=torch.float32) * (2.0 * math.pi / self.num_groups)
            # for uv sampling offsets
            grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
            grid_init = (grid_init /
                        grid_init.abs().max(-1, keepdim=True)[0]).view(
                self.num_groups, 1, 1,
                2).repeat(1, self.num_levels, self.num_pts, 1)
            for i in range(self.num_pts):
                grid_init[:, :, i, :] *= i + 1
            self.sampling_offsets.bias.data = grid_init.view(-1)
            # for depth sampling offsets
            grid_init_depth = torch.stack([(thetas.cos() + thetas.sin()) / 2], -1)
            grid_init_depth = grid_init_depth.view(self.num_groups, 1, 1, 1).repeat(1, self.num_levels, self.num_pts, 1)
            for i in range(self.num_pts):
                grid_init_depth[:, :, i, :] *= i + 1
            self.sampling_offsets_depth.bias.data = grid_init_depth.view(-1)
        
        if self.value_projection:
            xavier_init(self.value_proj, distribution='uniform', bias=0.)

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        feature_maps: List[torch.Tensor],
        dpt_feature_maps: List[torch.Tensor], # depth feature maps
        metas: dict,
        anchor_encoder=None,
        **kwargs: dict,
    ):
        # Choose the float type based on the feature maps dtype
        if feature_maps[0].dtype == torch.float16:
            MultiScaleDeformableAttnFunction = MultiScale3DDeformableAttnFunction_fp16
        else:
            MultiScaleDeformableAttnFunction = MultiScale3DDeformableAttnFunction_fp32
        
        bs, num_anchor = instance_feature.shape[:2]
        key_points = self.kps_generator(anchor, instance_feature) # [bs, num_anchor, num_pts, 3]
        # print("key_points shape: ", key_points.shape) # torch.Size([1, 25600, 9, 3])

        temp_key_points_list = (
            feature_queue
        ) = meta_queue = temp_anchor_embeds = []
        # Format the feature maps to the one used in DAT
        if self.use_deformable_func:
            feature_maps_lst = MultiScaleDeformableAttnFunction.feature_maps_format(feature_maps, dpt_feature_maps, self.num_groups)
            feat_flatten, dpt_dist_flatten, spatial_shape_3D, level_start_index = feature_maps_lst

        if self.value_projection:
            # print("before feat_flatten shape: ", feat_flatten.shape)
            _, l, num_groups, group_dims = feat_flatten.shape
            assert num_groups == self.num_groups and group_dims == self.group_dims
            feat_flatten = feat_flatten.view(bs*self.num_cams, l, -1)
            feat_flatten = self.value_proj(feat_flatten)
            feat_flatten = feat_flatten.view(bs*self.num_cams, l, self.num_groups, self.group_dims)
            # print("after feat_flatten shape: ", feat_flatten.shape)

        for (
            temp_feature_maps,
            temp_metas,
            temp_key_points,
            temp_anchor_embed,
        ) in zip(
            feature_queue[::-1] + [feature_maps_lst],
            meta_queue[::-1] + [metas],
            temp_key_points_list[::-1] + [key_points],
            temp_anchor_embeds[::-1] + [anchor_embed],
        ):
            weights = self._get_weights_3d(
                instance_feature, temp_anchor_embed, metas
            )# (bs*num_cam, num_anchor, num_groups, num_levels, num_pts)
            # print("temp_key_points shape: ", temp_key_points.shape) # torch.Size([1, 25600, 9, 3])
            if self.use_deformable_func:
                points_3d, bev_mask = self.project_points_3d(
                    temp_key_points, 
                    temp_metas["projection_mat"],
                    self.d_bound,
                    temp_metas.get("image_wh")
                ) # (num_cam, bs, num_anchor, num_pts, 3), # (num_cam, bs, num_anchor, num_pts)

                # For visibility mask
                indexes = []
                for i, mask_per_img in enumerate(bev_mask):
                    index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
                    indexes.append(index_query_per_img) # (N,)
                # max_len = max([len(each) for each in indexes]) # max number of valid anchors across all cameras

                if not self.use_sampling_offsets:
                    points_3d = points_3d.view(self.num_cams*bs, num_anchor, self.num_pts, 3)
                    points_3d = points_3d[:, :, None, None, :, :].repeat(1, 1, self.num_groups, self.num_levels, 1, 1)
                    # print(f"points_3d: {points_3d.shape}") # torch.Size([6, 25600, 4, 4, 9, 3])
                    # print(f"weights: {weights.shape}") # torch.Size([6, 25600, 4, 4, 9])
                else:
                    feature = instance_feature + anchor_embed
                    if self.camera_encoder is not None:
                        # [bs, num_cams, 3, 4] -> [bs, num_cams, 3 * 4] -> [bs, num_cams, embed_dims]
                        camera_embed = self.camera_encoder(
                            temp_metas["projection_mat"][:, :, :3].reshape(
                                bs, self.num_cams, -1
                            )
                        ) # shape of [bs, num_cams, embed_dims]
                        feature = feature[:, None] + camera_embed[:, :, None] # [bs, 1, num_anchor, embed_dims] + [bs, num_cams, 1, embed_dims] = [bs, num_cams, num_anchor, embed_dims]
                    else:
                        feature = feature[:, None].repeat(1, self.num_cams, 1, 1)
                    feature = feature.view(bs*self.num_cams, num_anchor, -1)
                    bs_num_cams, num_anchor, embed_dims = feature.shape
                    assert embed_dims == self.embed_dims and bs_num_cams == bs*self.num_cams

                    points_3d = points_3d.view(self.num_cams*bs, num_anchor, -1, 3)
                    _, _, num_keypoints, _ = points_3d.shape

                    # Learn the sampling offsets
                    sampling_offsets_uv = self.sampling_offsets(feature).view(
                        bs_num_cams, num_anchor, self.num_groups, self.num_levels, self.num_pts, 2
                    )
                    sampling_offsets_depth = self.sampling_offsets_depth(feature).view(
                        bs_num_cams, num_anchor, self.num_groups, self.num_levels, self.num_pts, 1
                    )
                    sampling_offsets = torch.cat([sampling_offsets_uv, sampling_offsets_depth], -1)
                    # (4, 3), (w, h, d)
                    offset_normalizer = torch.stack([spatial_shape_3D[..., 1], spatial_shape_3D[..., 0], spatial_shape_3D[..., 2]], -1)
                    sampling_offsets = sampling_offsets / \
                        offset_normalizer[None, None, None, :, None, :]
                    bs_num_cams, num_anchor, num_groups, num_levels, num_all_points, coord_xyz = sampling_offsets.shape
                    assert num_all_points == self.num_pts
                    sampling_offsets = sampling_offsets.view(
                        bs_num_cams, num_anchor, num_groups, num_levels, num_all_points // num_keypoints, num_keypoints, coord_xyz)

                    # Update the points_3d with sampling offsets
                    points_3d = points_3d[:, :, None, None, None, :, :]
                    points_3d = points_3d + sampling_offsets
                    points_3d = points_3d.view(self.num_cams*bs, num_anchor, self.num_groups, self.num_levels, self.num_pts, coord_xyz)

                temp_features_next, depth_score = MultiScaleDeformableAttnFunction.apply(
                feat_flatten, dpt_dist_flatten, spatial_shape_3D, level_start_index, points_3d,
                weights, self.im2col_step)

                temp_features_next = temp_features_next.view(bs, self.num_cams, num_anchor, self.embed_dims)

                # TODO: (1) Use visibility to get camera aggregated features; (2) Average the camera dimension
                slots = torch.zeros_like(instance_feature) # [bs, num_anchor, embed_dims]
                if self.use_visibility:
                    for j in range(bs):
                        for i, index_query_per_img in enumerate(indexes):
                            slots[j, index_query_per_img] += temp_features_next[j, i, :len(index_query_per_img)]

                    count = bev_mask.sum(-1) > 0
                    count = count.permute(1, 2, 0).sum(-1) # (bs, num_anchor)
                    count = torch.clamp(count, min=1.0)
                    slots = slots / count[..., None] # (bs, num_anchor, embed_dims) / (bs, num_anchor, 1)
                else:
                    slots = temp_features_next.mean(1) # [bs, num_anchor, embed_dims]

        output = self.proj_drop(self.output_proj(slots)) # [bs, num_anchor, embed_dims]
        if self.residual_mode == "add":
            output = output + instance_feature
        elif self.residual_mode == "cat":
            output = torch.cat([output, instance_feature], dim=-1)
        return output

    def _get_weights(self, instance_feature, anchor_embed, metas=None):
        bs, num_anchor = instance_feature.shape[:2]
        feature = instance_feature + anchor_embed # [bs, num_anchor, embed_dims]
        if self.camera_encoder is not None:
            # [bs, num_cams, 3, 4] -> [bs, num_cams, 3 * 4] -> [bs, num_cams, embed_dims]
            camera_embed = self.camera_encoder(
                metas["projection_mat"][:, :, :3].reshape(
                    bs, self.num_cams, -1
                )
            ) # [bs, num_cams, embed_dims]
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

    def _get_weights_3d(self, instance_feature, anchor_embed, metas=None):
        bs, num_anchor = instance_feature.shape[:2]
        feature = instance_feature + anchor_embed # shape of [bs, num_anchor, embed_dims]
        if self.camera_encoder is not None:
            # [bs, num_cams, 3, 4] -> [bs, num_cams, 3 * 4] -> [bs, num_cams, embed_dims]
            camera_embed = self.camera_encoder(
                metas["projection_mat"][:, :, :3].reshape(
                    bs, self.num_cams, -1
                )
            ) # shape of [bs, num_cams, embed_dims]
            feature = feature[:, None] + camera_embed[:, :, None] # [bs, 1, num_anchor, embed_dims] + [bs, num_cams, 1, embed_dims] = [bs, num_cams, num_anchor, embed_dims]
        weights = (
            self.weights_fc(feature) # [bs, num_cams, num_anchor, embed_dims] -> [bs, num_cams, num_anchor, num_groups * num_levels * num_pts]
            .view(bs*self.num_cams, num_anchor, self.num_groups, self.num_levels*self.num_pts)# [bs * num_cams, num_anchor, num_groups, num_levels * num_pts]
            .softmax(dim=-1)
            .view(
                bs*self.num_cams,
                num_anchor,
                self.num_groups,
                self.num_levels,
                self.num_pts,
            )
        )
        if self.training and self.attn_drop > 0:
            mask = torch.rand(
                bs*self.num_cams, num_anchor, 1, 1, self.num_pts
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
        points_2d = torch.matmul(
            projection_mat[:, :, None, None], pts_extend[:, None, ..., None]
        ).squeeze(-1) 
        points_2d = points_2d[..., :2] / torch.clamp(
            points_2d[..., 2:3], min=1e-5
        ) # [bs, num_cams, num_anchor, num_pts, 2]
        if image_wh is not None:
            points_2d = points_2d / image_wh[:, :, None, None] # [bs, num_cams, num_anchor, num_pts, 2]
        return points_2d
    
    @staticmethod
    def project_points_3d(key_points, projection_mat, d_bound, image_wh=None):
        # print('key_points', key_points.shape) # torch.Size([1, 25600, 9, 3])
        # print('projection_mat', projection_mat.shape) # torch.Size([1, 6, 4, 4])
        bs, num_anchor, num_pts = key_points.shape[:3] # [bs, num_anchor, num_pts, 3] in lidar coordinate
        num_cam = projection_mat.size(1)

        pts_extend = torch.cat(
            [key_points, torch.ones_like(key_points[..., :1])], dim=-1
        ) # [bs, num_anchor, num_pts, 4], xyz1
        pts_extend = pts_extend.permute(2, 0, 1, 3) # (num_pts, bs, num_anchor, 4)
        # print('pts_extend', pts_extend.shape) # torch.Size([9, 1, 25600, 4])
        pts_extend = pts_extend.view(
            num_pts, bs, 1, num_anchor, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1) # (num_pts, bs, num_cam, num_anchor, 4, 1)
        projection_mat = projection_mat.view(
            1, bs, num_cam, 1, 4, 4).repeat(num_pts, 1, 1, num_anchor, 1, 1) # (num_pts, bs, num_cam, num_anchor, 4, 4)

        points_3d = torch.matmul(projection_mat.to(torch.float32),
                                pts_extend.to(torch.float32)).squeeze(-1) # (num_pts, bs, num_cam, num_anchor, 4)
                                    
        eps = 1e-5
        bev_mask = (points_3d[..., 2:3] > eps) # (num_pts, bs, num_cam, num_anchor, 1), in front of camera
        points_3d[..., 0:2] = points_3d[..., 0:2].clone() / torch.maximum(
            points_3d[..., 2:3].clone(), torch.ones_like(points_3d[..., 2:3].clone()) * eps)

        points_3d[..., 0] = points_3d[..., 0].clone() / image_wh[None, :, :, None, :][..., 0]
        points_3d[..., 1] = points_3d[..., 1].clone() / image_wh[None, :, :, None, :][..., 1]
        points_3d[..., 2]  = (points_3d[..., 2].clone() - d_bound[0]) / (d_bound[1] - d_bound[0])
        points_3d = points_3d[..., :3].clone() # (num_pts, bs, num_cam, num_anchor, 3)
        bev_mask = (bev_mask & (points_3d[..., 1:2] > 0.0)
                    & (points_3d[..., 1:2] < 1.0)
                    & (points_3d[..., 0:1] < 1.0)
                    & (points_3d[..., 0:1] > 0.0))
        # Option 1
        bev_mask = torch.nan_to_num(bev_mask)

        # Option 2
        # bev_mask = bev_mask.new_tensor(
        #         np.nan_to_num(bev_mask.cpu().numpy()))
        
        points_3d = points_3d.permute(2, 1, 3, 0, 4) # (num_cam, bs, num_anchor, num_pts, 3)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1) # (num_cam, bs, num_anchor, num_pts)

        return points_3d, bev_mask

