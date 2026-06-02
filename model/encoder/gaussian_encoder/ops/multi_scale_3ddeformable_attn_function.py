# ------------------------------------------------------------------------
# DFA3D
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the IDEA License, Version 1.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from BEVFormer (https://github.com/fundamentalvision/BEVFormer)
# Copyright (c) fundamentalvision. All rights reserved
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
#  Modified by Hongyang Li
# ---------------------------------------------

import torch
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.autograd.function import Function, once_differentiable
from dfa3D import ext_loader
ext_module = ext_loader.load_ext(
    '_ext', ['wms_deform_attn_backward', 'wms_deform_attn_forward', 'ms_depth_score_sample_forward', 'ms_depth_score_sample_backward'])

class WeightedMultiScaleDeformableAttnFunction_fp16(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, depth_score, im2col_step):
        """GPU version of multi-scale deformable attention.

        Args:
            value (Tensor): The value has shape
                (bs, num_keys, mum_heads, embed_dims//num_heads)
            value_spatial_shapes (Tensor): Spatial shape of
                each feature map, has shape (num_levels, 2),
                last dimension 2 represent (h, w)
            sampling_locations (Tensor): The location of sampling points,
                has shape
                (bs ,num_queries, num_heads, num_levels, num_points, 2),
                the last dimension 2 represent (x, y).
            attention_weights (Tensor): The weight of sampling points used
                when calculate the attention, has shape
                (bs ,num_queries, num_heads, num_levels, num_points),
            im2col_step (Tensor): The step used in image to column.

        Returns:
            Tensor: has shape (bs, num_queries, embed_dims)
        """
        ctx.im2col_step = im2col_step
        output = ext_module.wms_deform_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            depth_score,
            im2col_step=ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes,
                              value_level_start_index, sampling_locations,
                              attention_weights, depth_score)
        return output

    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx, grad_output):
        """GPU version of backward function.

        Args:
            grad_output (Tensor): Gradient
                of output tensor of forward.

        Returns:
             Tuple[Tensor]: Gradient
                of input tensors in forward.
        """
        value, value_spatial_shapes, value_level_start_index, \
            sampling_locations, attention_weights, depth_score = ctx.saved_tensors
        grad_value = torch.zeros_like(value)
        grad_sampling_loc = torch.zeros_like(sampling_locations)
        grad_attn_weight = torch.zeros_like(attention_weights)
        grad_depth_score = torch.zeros_like(depth_score)

        ext_module.wms_deform_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            depth_score,
            grad_output.contiguous(),
            grad_value,
            grad_sampling_loc,
            grad_attn_weight,
            grad_depth_score,
            im2col_step=ctx.im2col_step)

        return grad_value, None, None, \
            grad_sampling_loc, grad_attn_weight, grad_depth_score, None
    
    @staticmethod
    def feature_maps_format(feature_maps, dpt_feature_maps, num_groups):
        bs, num_cams, channel_num = feature_maps[0].shape[:3] # feature_maps: list of element with shape (bs, num_cams, channel, Hi, Wi), each element is a feature map from a scale, we have 4 scales
        feature_maps_lst = [] # should include feature_maps, dpt_feature_maps, spatial_shapes, level_start_index

        feat_flatten = [] # keep track of the flattened feature maps
        dpt_dist_flatten = [] # keep track of the flattened depth maps
        spatial_shapes = [] # keep track of the spatial shapes of the feature maps
        for lvl, (feat, dpt_dist) in enumerate(zip(feature_maps, dpt_feature_maps)):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            # feat: bs, num_cam, c, h, w -> bs, num_cam, c, h*w -> num_cam, bs, h*w, c
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            # dpt_dist: bs, num_cam, 112, h, w -> bs, num_cam, 112, h*w -> num_cam, bs, h*w, 112
            dpt_dist = dpt_dist.flatten(3).permute(1, 0, 3, 2)

            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)
            dpt_dist_flatten.append(dpt_dist)

        # feat_flatten: num_cam, bs, h*w, c -> num_cam, bs, sigma(h*w), c
        # dpt_dist_flatten: num_cam, bs, h*w, 112 -> num_cam, bs, sigma(h*w), 112
        feat_flatten = torch.cat(feat_flatten, 2)
        dpt_dist_flatten = torch.cat(dpt_dist_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feature_maps[0].device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, sigma:Hi*Wi, bs, c)
        dpt_dist_flatten = dpt_dist_flatten.permute(
            0, 2, 1, 3)  # (num_cam, sigma:Hi*Wi, bs, 112)
        num_cam, l, bs, channel_num = feat_flatten.shape
        _, _, _, depth_num = dpt_dist_flatten.shape

        # feat_flatten: (N, sigma, B, c) -> (B, N, sigma, c) -> (B*N, sigma, c) -> (B*N, sigma, num_group, group_dim)
        feat_flatten = feat_flatten.permute(2, 0, 1, 3).reshape(
            bs * num_cams, l, channel_num)
        feat_flatten = feat_flatten.view(bs * num_cams, l, num_groups, -1)
        # dpt_dist_flatten: (N, sigma, B, 112) -> (B, N, sigma, 112) -> (B*N, sigma, 112) -> (B*N, sigma, num_group, 112)
        dpt_dist_flatten = dpt_dist_flatten.permute(2, 0, 1, 3).reshape(
            bs * num_cams, l, depth_num)
        dpt_dist_flatten = dpt_dist_flatten.view(bs * num_cams, l, 1, depth_num).repeat(1, 1, num_groups, 1)

        spatial_shape_depth = spatial_shapes.new_ones(*spatial_shapes.shape[:-1], 1) * depth_num
        spatial_shape_3D = torch.cat([spatial_shapes, spatial_shape_depth], dim=-1)
        spatial_shape_3D = spatial_shape_3D.contiguous()

        # feat_flatten: (B*N, sigma, num_group, group_dim)
        # dpt_dist_flatten: (B*N, sigma, num_group, 112)
        # spatial_shape_3D: (num_cam, 3)
        # level_start_index: (num_cam)
        feature_maps_lst = [feat_flatten, dpt_dist_flatten, spatial_shape_3D, level_start_index]

        return feature_maps_lst


class WeightedMultiScaleDeformableAttnFunction_fp32(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, depth_score, im2col_step):
        """GPU version of multi-scale deformable attention.

        Args:
            value (Tensor): The value has shape
                (bs, num_keys, mum_heads, embed_dims//num_heads)
            value_spatial_shapes (Tensor): Spatial shape of
                each feature map, has shape (num_levels, 2),
                last dimension 2 represent (h, w)
            sampling_locations (Tensor): The location of sampling points,
                has shape
                (bs ,num_queries, num_heads, num_levels, num_points, 2),
                the last dimension 2 represent (x, y).
            attention_weights (Tensor): The weight of sampling points used
                when calculate the attention, has shape
                (bs ,num_queries, num_heads, num_levels, num_points),
            im2col_step (Tensor): The step used in image to column.

        Returns:
            Tensor: has shape (bs, num_queries, embed_dims)
        """

        ctx.im2col_step = im2col_step
        output = ext_module.wms_deform_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            depth_score,
            im2col_step=ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes,
                              value_level_start_index, sampling_locations,
                              attention_weights, depth_score)
        return output

    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx, grad_output):
        """GPU version of backward function.

        Args:
            grad_output (Tensor): Gradient
                of output tensor of forward.

        Returns:
             Tuple[Tensor]: Gradient
                of input tensors in forward.
        """
        value, value_spatial_shapes, value_level_start_index, \
            sampling_locations, attention_weights, depth_score = ctx.saved_tensors
        grad_value = torch.zeros_like(value)
        grad_sampling_loc = torch.zeros_like(sampling_locations)
        grad_attn_weight = torch.zeros_like(attention_weights)
        grad_depth_score = torch.zeros_like(depth_score)

        ext_module.wms_deform_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            depth_score,
            grad_output.contiguous(),
            grad_value,
            grad_sampling_loc,
            grad_attn_weight,
            grad_depth_score,
            im2col_step=ctx.im2col_step)

        return grad_value, None, None, \
            grad_sampling_loc, grad_attn_weight, grad_depth_score, None
    
    @staticmethod
    def feature_maps_format(feature_maps, dpt_feature_maps, num_groups):
        bs, num_cams, channel_num = feature_maps[0].shape[:3] # feature_maps: list of element with shape (bs, num_cams, channel, Hi, Wi), each element is a feature map from a scale, we have 4 scales
        feature_maps_lst = [] # should include feature_maps, dpt_feature_maps, spatial_shapes, level_start_index

        feat_flatten = [] # keep track of the flattened feature maps
        dpt_dist_flatten = [] # keep track of the flattened depth maps
        spatial_shapes = [] # keep track of the spatial shapes of the feature maps
        for lvl, (feat, dpt_dist) in enumerate(zip(feature_maps, dpt_feature_maps)):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            # feat: bs, num_cam, c, h, w -> bs, num_cam, c, h*w -> num_cam, bs, h*w, c
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            # dpt_dist: bs, num_cam, 112, h, w -> bs, num_cam, 112, h*w -> num_cam, bs, h*w, 112
            dpt_dist = dpt_dist.flatten(3).permute(1, 0, 3, 2)

            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)
            dpt_dist_flatten.append(dpt_dist)

        # feat_flatten: num_cam, bs, h*w, c -> num_cam, bs, sigma(h*w), c
        # dpt_dist_flatten: num_cam, bs, h*w, 112 -> num_cam, bs, sigma(h*w), 112
        feat_flatten = torch.cat(feat_flatten, 2)
        dpt_dist_flatten = torch.cat(dpt_dist_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feature_maps[0].device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, sigma:Hi*Wi, bs, c)
        dpt_dist_flatten = dpt_dist_flatten.permute(
            0, 2, 1, 3)  # (num_cam, sigma:Hi*Wi, bs, 112)
        num_cam, l, bs, channel_num = feat_flatten.shape
        _, _, _, depth_num = dpt_dist_flatten.shape

        # feat_flatten: (N, sigma, B, c) -> (B, N, sigma, c) -> (B*N, sigma, c) -> (B*N, sigma, num_group, group_dim)
        feat_flatten = feat_flatten.permute(2, 0, 1, 3).reshape(
            bs * num_cams, l, channel_num)
        feat_flatten = feat_flatten.view(bs * num_cams, l, num_groups, -1)
        # dpt_dist_flatten: (N, sigma, B, 112) -> (B, N, sigma, 112) -> (B*N, sigma, 112) -> (B*N, sigma, num_group, 112)
        dpt_dist_flatten = dpt_dist_flatten.permute(2, 0, 1, 3).reshape(
            bs * num_cams, l, depth_num)
        dpt_dist_flatten = dpt_dist_flatten.view(bs * num_cams, l, 1, depth_num).repeat(1, 1, num_groups, 1)

        spatial_shape_depth = spatial_shapes.new_ones(*spatial_shapes.shape[:-1], 1) * depth_num
        spatial_shape_3D = torch.cat([spatial_shapes, spatial_shape_depth], dim=-1)
        spatial_shape_3D = spatial_shape_3D.contiguous()

        # feat_flatten: (B*N, sigma, num_group, group_dim)
        # dpt_dist_flatten: (B*N, sigma, num_group, 112)
        # spatial_shape_3D: (num_cam, 3)
        # level_start_index: (num_cam)
        feature_maps_lst = [feat_flatten, dpt_dist_flatten, spatial_shape_3D, level_start_index]

        return feature_maps_lst


class MultiScaleDepthScoreSampleFunction_fp16(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, value: torch.Tensor, value_spatial_shapes: torch.Tensor,
                value_level_start_index: torch.Tensor,
                sampling_locations: torch.Tensor,
                im2col_step: torch.Tensor) -> torch.Tensor:
        ctx.im2col_step = im2col_step
        output = ext_module.ms_depth_score_sample_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            im2col_step=ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes,
                              value_level_start_index, sampling_locations)
        return output
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        """GPU version of backward function.

        Args:
            grad_output (torch.Tensor): Gradient of output tensor of forward.

        Returns:
            tuple[Tensor]: Gradient of input tensors in forward.
        """
        value, value_spatial_shapes, value_level_start_index,\
            sampling_locations = ctx.saved_tensors
        # ToDo: backward do not consider the bilinear weights currently.
        grad_value = torch.zeros_like(value)
        grad_sampling_loc = torch.zeros_like(sampling_locations)

        ext_module.ms_depth_score_sample_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            grad_output.contiguous(),
            grad_value,
            grad_sampling_loc,
            im2col_step=ctx.im2col_step)

        return grad_value, None, None, \
            grad_sampling_loc, None
        
    @staticmethod
    def feature_maps_format(feature_maps, dpt_feature_maps, num_groups):
        bs, num_cams, channel_num = feature_maps[0].shape[:3] # feature_maps: list of element with shape (bs, num_cams, channel, Hi, Wi), each element is a feature map from a scale, we have 4 scales
        feature_maps_lst = [] # should include feature_maps, dpt_feature_maps, spatial_shapes, level_start_index

        feat_flatten = [] # keep track of the flattened feature maps
        dpt_dist_flatten = [] # keep track of the flattened depth maps
        spatial_shapes = [] # keep track of the spatial shapes of the feature maps
        for lvl, (feat, dpt_dist) in enumerate(zip(feature_maps, dpt_feature_maps)):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            # feat: bs, num_cam, c, h, w -> bs, num_cam, c, h*w -> num_cam, bs, h*w, c
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            # dpt_dist: bs, num_cam, 112, h, w -> bs, num_cam, 112, h*w -> num_cam, bs, h*w, 112
            dpt_dist = dpt_dist.flatten(3).permute(1, 0, 3, 2)

            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)
            dpt_dist_flatten.append(dpt_dist)

        # feat_flatten: num_cam, bs, h*w, c -> num_cam, bs, sigma(h*w), c
        # dpt_dist_flatten: num_cam, bs, h*w, 112 -> num_cam, bs, sigma(h*w), 112
        feat_flatten = torch.cat(feat_flatten, 2)
        dpt_dist_flatten = torch.cat(dpt_dist_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feature_maps[0].device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, sigma:Hi*Wi, bs, c)
        dpt_dist_flatten = dpt_dist_flatten.permute(
            0, 2, 1, 3)  # (num_cam, sigma:Hi*Wi, bs, 112)
        num_cam, l, bs, channel_num = feat_flatten.shape
        _, _, _, depth_num = dpt_dist_flatten.shape

        # feat_flatten: (N, sigma, B, c) -> (B, N, sigma, c) -> (B*N, sigma, c) -> (B*N, sigma, num_group, group_dim)
        feat_flatten = feat_flatten.permute(2, 0, 1, 3).reshape(
            bs * num_cams, l, channel_num)
        feat_flatten = feat_flatten.view(bs * num_cams, l, num_groups, -1)
        # dpt_dist_flatten: (N, sigma, B, 112) -> (B, N, sigma, 112) -> (B*N, sigma, 112) -> (B*N, sigma, num_group, 112)
        dpt_dist_flatten = dpt_dist_flatten.permute(2, 0, 1, 3).reshape(
            bs * num_cams, l, depth_num)
        dpt_dist_flatten = dpt_dist_flatten.view(bs * num_cams, l, 1, depth_num).repeat(1, 1, num_groups, 1)

        spatial_shape_depth = spatial_shapes.new_ones(*spatial_shapes.shape[:-1], 1) * depth_num
        spatial_shape_3D = torch.cat([spatial_shapes, spatial_shape_depth], dim=-1)
        spatial_shape_3D = spatial_shape_3D.contiguous()

        # feat_flatten: (B*N, sigma, num_group, group_dim)
        # dpt_dist_flatten: (B*N, sigma, num_group, 112)
        # spatial_shape_3D: (num_cam, 3)
        # level_start_index: (num_cam)
        feature_maps_lst = [feat_flatten, dpt_dist_flatten, spatial_shape_3D, level_start_index]

        return feature_maps_lst


class MultiScaleDepthScoreSampleFunction_fp32(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, value: torch.Tensor, value_spatial_shapes: torch.Tensor,
                value_level_start_index: torch.Tensor,
                sampling_locations: torch.Tensor,
                im2col_step: torch.Tensor) -> torch.Tensor:
        ctx.im2col_step = im2col_step
        output = ext_module.ms_depth_score_sample_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            im2col_step=ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes,
                              value_level_start_index, sampling_locations)
        return output
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        """GPU version of backward function.

        Args:
            grad_output (torch.Tensor): Gradient of output tensor of forward.

        Returns:
            tuple[Tensor]: Gradient of input tensors in forward.
        """
        value, value_spatial_shapes, value_level_start_index,\
            sampling_locations = ctx.saved_tensors
        # ToDo: backward do not consider the bilinear weights currently.
        grad_value = torch.zeros_like(value)
        grad_sampling_loc = torch.zeros_like(sampling_locations)

        ext_module.ms_depth_score_sample_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            grad_output.contiguous(),
            grad_value,
            grad_sampling_loc,
            im2col_step=ctx.im2col_step)

        return grad_value, None, None, \
            grad_sampling_loc, None
    
    @staticmethod
    def feature_maps_format(feature_maps, dpt_feature_maps, num_groups):
        bs, num_cams, channel_num = feature_maps[0].shape[:3] # feature_maps: list of element with shape (bs, num_cams, channel, Hi, Wi), each element is a feature map from a scale, we have 4 scales
        feature_maps_lst = [] # should include feature_maps, dpt_feature_maps, spatial_shapes, level_start_index

        feat_flatten = [] # keep track of the flattened feature maps
        dpt_dist_flatten = [] # keep track of the flattened depth maps
        spatial_shapes = [] # keep track of the spatial shapes of the feature maps
        for lvl, (feat, dpt_dist) in enumerate(zip(feature_maps, dpt_feature_maps)):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            # feat: bs, num_cam, c, h, w -> bs, num_cam, c, h*w -> num_cam, bs, h*w, c
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            # dpt_dist: bs, num_cam, 112, h, w -> bs, num_cam, 112, h*w -> num_cam, bs, h*w, 112
            dpt_dist = dpt_dist.flatten(3).permute(1, 0, 3, 2)

            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)
            dpt_dist_flatten.append(dpt_dist)

        # feat_flatten: num_cam, bs, h*w, c -> num_cam, bs, sigma(h*w), c
        # dpt_dist_flatten: num_cam, bs, h*w, 112 -> num_cam, bs, sigma(h*w), 112
        feat_flatten = torch.cat(feat_flatten, 2)
        dpt_dist_flatten = torch.cat(dpt_dist_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feature_maps[0].device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, sigma:Hi*Wi, bs, c)
        dpt_dist_flatten = dpt_dist_flatten.permute(
            0, 2, 1, 3)  # (num_cam, sigma:Hi*Wi, bs, 112)
        num_cam, l, bs, channel_num = feat_flatten.shape
        _, _, _, depth_num = dpt_dist_flatten.shape

        # feat_flatten: (N, sigma, B, c) -> (B, N, sigma, c) -> (B*N, sigma, c) -> (B*N, sigma, num_group, group_dim)
        feat_flatten = feat_flatten.permute(2, 0, 1, 3).reshape(
            bs * num_cams, l, channel_num)
        feat_flatten = feat_flatten.view(bs * num_cams, l, num_groups, -1)
        # dpt_dist_flatten: (N, sigma, B, 112) -> (B, N, sigma, 112) -> (B*N, sigma, 112) -> (B*N, sigma, num_group, 112)
        dpt_dist_flatten = dpt_dist_flatten.permute(2, 0, 1, 3).reshape(
            bs * num_cams, l, depth_num)
        dpt_dist_flatten = dpt_dist_flatten.view(bs * num_cams, l, 1, depth_num).repeat(1, 1, num_groups, 1)

        spatial_shape_depth = spatial_shapes.new_ones(*spatial_shapes.shape[:-1], 1) * depth_num
        spatial_shape_3D = torch.cat([spatial_shapes, spatial_shape_depth], dim=-1)
        spatial_shape_3D = spatial_shape_3D.contiguous()

        # feat_flatten: (B*N, sigma, num_group, group_dim)
        # dpt_dist_flatten: (B*N, sigma, num_group, 112)
        # spatial_shape_3D: (num_cam, 3)
        # level_start_index: (num_cam)
        feature_maps_lst = [feat_flatten, dpt_dist_flatten, spatial_shape_3D, level_start_index]

        return feature_maps_lst


class MultiScale3DDeformableAttnFunction_fp32(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, value: torch.Tensor, value_dpt_dist: torch.Tensor, value_spatial_shapes: torch.Tensor,
                value_level_start_index: torch.Tensor,
                sampling_locations: torch.Tensor,
                attention_weights: torch.Tensor,
                im2col_step: torch.Tensor) -> torch.Tensor:
        ctx.im2col_step = im2col_step

        depth_score = ext_module.ms_depth_score_sample_forward(
            value_dpt_dist,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            im2col_step=ctx.im2col_step)
        
        output = ext_module.wms_deform_attn_forward(
            value,
            value_spatial_shapes[..., :2].contiguous(),
            value_level_start_index,
            sampling_locations[..., :2].contiguous(),
            attention_weights,
            depth_score,
            im2col_step=ctx.im2col_step)
        ctx.save_for_backward(value, value_dpt_dist, value_spatial_shapes,
                              value_level_start_index, sampling_locations, attention_weights, depth_score)
        return output, depth_score
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor, grad_depth_score_: torch.Tensor) -> tuple:
        """GPU version of backward function.

        Args:
            grad_output (torch.Tensor): Gradient of output tensor of forward.

        Returns:
            tuple[Tensor]: Gradient of input tensors in forward.
        """
        if grad_depth_score_.sum() != 0.0:
            raise NotImplementedError
        value, value_dpt_dist, value_spatial_shapes, value_level_start_index,\
            sampling_locations, attention_weights, depth_score = ctx.saved_tensors
        # ToDo: backward do not consider the bilinear weights currently.
        grad_value = torch.zeros_like(value)
        grad_sampling_loc_ = torch.zeros([*sampling_locations.shape[:-1]] + [2, ], dtype = sampling_locations.dtype, device = sampling_locations.device)
        grad_attn_weight = torch.zeros_like(attention_weights)
        grad_depth_score = torch.zeros_like(depth_score)

        ext_module.wms_deform_attn_backward(
            value,
            value_spatial_shapes[..., :2].contiguous(),
            value_level_start_index,
            sampling_locations[..., :2].contiguous(),
            attention_weights,
            depth_score,
            grad_output.contiguous(),
            grad_value,
            grad_sampling_loc_.contiguous(),
            grad_attn_weight,
            grad_depth_score,
            im2col_step=ctx.im2col_step)
        
        grad_value_dpt_dist = torch.zeros_like(value_dpt_dist)
        grad_sampling_loc = torch.zeros_like(sampling_locations)
        ext_module.ms_depth_score_sample_backward(
            value_dpt_dist,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            grad_depth_score.contiguous(),
            grad_value_dpt_dist,
            grad_sampling_loc,
            im2col_step=ctx.im2col_step)
        grad_sampling_loc[..., :2] = grad_sampling_loc[..., :2] + grad_sampling_loc_
        return grad_value, grad_value_dpt_dist, None, None, \
            grad_sampling_loc, grad_attn_weight, None
    
    @staticmethod
    def feature_maps_format(feature_maps, dpt_feature_maps, num_groups):
        bs, num_cams, channel_num = feature_maps[0].shape[:3] # feature_maps: list of element with shape (bs, num_cams, channel, Hi, Wi), each element is a feature map from a scale, we have 4 scales
        feature_maps_lst = [] # should include feature_maps, dpt_feature_maps, spatial_shapes, level_start_index

        feat_flatten = [] # keep track of the flattened feature maps
        dpt_dist_flatten = [] # keep track of the flattened depth maps
        spatial_shapes = [] # keep track of the spatial shapes of the feature maps
        for lvl, (feat, dpt_dist) in enumerate(zip(feature_maps, dpt_feature_maps)):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            # feat: bs, num_cam, c, h, w -> bs, num_cam, c, h*w -> num_cam, bs, h*w, c
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            # dpt_dist: bs, num_cam, 112, h, w -> bs, num_cam, 112, h*w -> num_cam, bs, h*w, 112
            dpt_dist = dpt_dist.flatten(3).permute(1, 0, 3, 2)

            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)
            dpt_dist_flatten.append(dpt_dist)

        # feat_flatten: num_cam, bs, h*w, c -> num_cam, bs, sigma(h*w), c
        # dpt_dist_flatten: num_cam, bs, h*w, 112 -> num_cam, bs, sigma(h*w), 112
        feat_flatten = torch.cat(feat_flatten, 2)
        dpt_dist_flatten = torch.cat(dpt_dist_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feature_maps[0].device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, sigma:Hi*Wi, bs, c)
        dpt_dist_flatten = dpt_dist_flatten.permute(
            0, 2, 1, 3)  # (num_cam, sigma:Hi*Wi, bs, 112)
        num_cam, l, bs, channel_num = feat_flatten.shape
        _, _, _, depth_num = dpt_dist_flatten.shape

        # feat_flatten: (N, sigma, B, c) -> (B, N, sigma, c) -> (B*N, sigma, c) -> (B*N, sigma, num_group, group_dim)
        feat_flatten = feat_flatten.permute(2, 0, 1, 3).reshape(
            bs * num_cams, l, channel_num)
        feat_flatten = feat_flatten.view(bs * num_cams, l, num_groups, -1)
        # dpt_dist_flatten: (N, sigma, B, 112) -> (B, N, sigma, 112) -> (B*N, sigma, 112) -> (B*N, sigma, num_group, 112)
        dpt_dist_flatten = dpt_dist_flatten.permute(2, 0, 1, 3).reshape(
            bs * num_cams, l, depth_num)
        dpt_dist_flatten = dpt_dist_flatten.view(bs * num_cams, l, 1, depth_num).repeat(1, 1, num_groups, 1)

        spatial_shape_depth = spatial_shapes.new_ones(*spatial_shapes.shape[:-1], 1) * depth_num
        spatial_shape_3D = torch.cat([spatial_shapes, spatial_shape_depth], dim=-1)
        spatial_shape_3D = spatial_shape_3D.contiguous()

        # feat_flatten: (B*N, sigma, num_group, group_dim)
        # dpt_dist_flatten: (B*N, sigma, num_group, 112)
        # spatial_shape_3D: (num_levels, 3)
        # level_start_index: (num_levels)

        # print("feat_flatten.shape: ", feat_flatten.shape)
        # print("dpt_dist_flatten.shape: ", dpt_dist_flatten.shape)
        # print("spatial_shape_3D.shape: ", spatial_shape_3D.shape)
        # print("level_start_index.shape: ", level_start_index.shape)
        # print(level_start_index.shape[1])

        # feat_flatten.shape:  torch.Size([6, 30825, 4, 32])
        # dpt_dist_flatten.shape:  torch.Size([6, 30825, 4, 112])
        # spatial_shape_3D.shape:  torch.Size([4, 3])
        # level_start_index.shape:  torch.Size([4])
        feature_maps_lst = [feat_flatten, dpt_dist_flatten, spatial_shape_3D, level_start_index]

        return feature_maps_lst


class MultiScale3DDeformableAttnFunction_fp16(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, value: torch.Tensor, value_dpt_dist: torch.Tensor, value_spatial_shapes: torch.Tensor,
                value_level_start_index: torch.Tensor,
                sampling_locations: torch.Tensor,
                attention_weights: torch.Tensor,
                im2col_step: torch.Tensor) -> torch.Tensor:
        ctx.im2col_step = im2col_step

        depth_score = ext_module.ms_depth_score_sample_forward(
            value_dpt_dist,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            im2col_step=ctx.im2col_step)
        
        output = ext_module.wms_deform_attn_forward(
            value,
            value_spatial_shapes[..., :2].contiguous(),
            value_level_start_index,
            sampling_locations[..., :2].contiguous(),
            attention_weights,
            depth_score,
            im2col_step=ctx.im2col_step)
        ctx.save_for_backward(value, value_dpt_dist, value_spatial_shapes,
                              value_level_start_index, sampling_locations, attention_weights, depth_score)
        return output, depth_score
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor, grad_depth_score_: torch.Tensor) -> tuple:
        """GPU version of backward function.

        Args:
            grad_output (torch.Tensor): Gradient of output tensor of forward.

        Returns:
            tuple[Tensor]: Gradient of input tensors in forward.
        """
        if grad_depth_score_.sum() != 0.0:
            raise NotImplementedError
        value, value_dpt_dist, value_spatial_shapes, value_level_start_index,\
            sampling_locations, attention_weights, depth_score = ctx.saved_tensors
        # ToDo: backward do not consider the bilinear weights currently.
        grad_value = torch.zeros_like(value)
        grad_sampling_loc_ = torch.zeros([*sampling_locations.shape[:-1]] + [2, ], dtype = sampling_locations.dtype, device = sampling_locations.device)
        grad_attn_weight = torch.zeros_like(attention_weights)
        grad_depth_score = torch.zeros_like(depth_score)

        ext_module.wms_deform_attn_backward(
            value,
            value_spatial_shapes[..., :2].contiguous(),
            value_level_start_index,
            sampling_locations[..., :2].contiguous(),
            attention_weights,
            depth_score,
            grad_output.contiguous(),
            grad_value,
            grad_sampling_loc_.contiguous(),
            grad_attn_weight,
            grad_depth_score,
            im2col_step=ctx.im2col_step)
        
        grad_value_dpt_dist = torch.zeros_like(value_dpt_dist)
        grad_sampling_loc = torch.zeros_like(sampling_locations)
        ext_module.ms_depth_score_sample_backward(
            value_dpt_dist,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            grad_depth_score.contiguous(),
            grad_value_dpt_dist,
            grad_sampling_loc,
            im2col_step=ctx.im2col_step)
        grad_sampling_loc[..., :2] = grad_sampling_loc[..., :2] + grad_sampling_loc_
        return grad_value, grad_value_dpt_dist, None, None, \
            grad_sampling_loc, grad_attn_weight, None
    
    @staticmethod
    def feature_maps_format(feature_maps, dpt_feature_maps, num_groups):
        bs, num_cams, channel_num = feature_maps[0].shape[:3] # feature_maps: list of element with shape (bs, num_cams, channel, Hi, Wi), each element is a feature map from a scale, we have 4 scales
        feature_maps_lst = [] # should include feature_maps, dpt_feature_maps, spatial_shapes, level_start_index

        feat_flatten = [] # keep track of the flattened feature maps
        dpt_dist_flatten = [] # keep track of the flattened depth maps
        spatial_shapes = [] # keep track of the spatial shapes of the feature maps
        for lvl, (feat, dpt_dist) in enumerate(zip(feature_maps, dpt_feature_maps)):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            # feat: bs, num_cam, c, h, w -> bs, num_cam, c, h*w -> num_cam, bs, h*w, c
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            # dpt_dist: bs, num_cam, 112, h, w -> bs, num_cam, 112, h*w -> num_cam, bs, h*w, 112
            dpt_dist = dpt_dist.flatten(3).permute(1, 0, 3, 2)

            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)
            dpt_dist_flatten.append(dpt_dist)

        # feat_flatten: num_cam, bs, h*w, c -> num_cam, bs, sigma(h*w), c
        # dpt_dist_flatten: num_cam, bs, h*w, 112 -> num_cam, bs, sigma(h*w), 112
        feat_flatten = torch.cat(feat_flatten, 2)
        dpt_dist_flatten = torch.cat(dpt_dist_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feature_maps[0].device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, sigma:Hi*Wi, bs, c)
        dpt_dist_flatten = dpt_dist_flatten.permute(
            0, 2, 1, 3)  # (num_cam, sigma:Hi*Wi, bs, 112)
        num_cam, l, bs, channel_num = feat_flatten.shape
        _, _, _, depth_num = dpt_dist_flatten.shape

        # feat_flatten: (N, sigma, B, c) -> (B, N, sigma, c) -> (B*N, sigma, c) -> (B*N, sigma, num_group, group_dim)
        feat_flatten = feat_flatten.permute(2, 0, 1, 3).reshape(
            bs * num_cams, l, channel_num)
        feat_flatten = feat_flatten.view(bs * num_cams, l, num_groups, -1)
        # dpt_dist_flatten: (N, sigma, B, 112) -> (B, N, sigma, 112) -> (B*N, sigma, 112) -> (B*N, sigma, num_group, 112)
        dpt_dist_flatten = dpt_dist_flatten.permute(2, 0, 1, 3).reshape(
            bs * num_cams, l, depth_num)
        dpt_dist_flatten = dpt_dist_flatten.view(bs * num_cams, l, 1, depth_num).repeat(1, 1, num_groups, 1)

        spatial_shape_depth = spatial_shapes.new_ones(*spatial_shapes.shape[:-1], 1) * depth_num
        spatial_shape_3D = torch.cat([spatial_shapes, spatial_shape_depth], dim=-1)
        spatial_shape_3D = spatial_shape_3D.contiguous()

        # feat_flatten: (B*N, sigma, num_group, group_dim)
        # dpt_dist_flatten: (B*N, sigma, num_group, 112)
        # spatial_shape_3D: (num_cam, 3)
        # level_start_index: (num_cam)
        feature_maps_lst = [feat_flatten, dpt_dist_flatten, spatial_shape_3D, level_start_index]

        return feature_maps_lst
