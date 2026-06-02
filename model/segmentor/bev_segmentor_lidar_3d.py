
from mmseg.models import SEGMENTORS
from mmseg.models import build_backbone
from mmseg.models import build_head

from model.utils.grid_mask import GridMask, GridMaskHybrid

from .base_segmentor import CustomBaseSegmentor
from mmcv.ops.voxelize import Voxelization
from mmdet3d.registry import MODELS
import torch
from torch.nn import functional as F

@SEGMENTORS.register_module()
class BEVSegmentorLiDAR3D(CustomBaseSegmentor):

    def __init__(
        self,
        freeze_img_backbone=False,
        freeze_img_neck=False,
        img_backbone_out_indices=[1, 2, 3],
        extra_img_backbone=None,
        voxelize_lidar=None,
        lidar_voxel_encoder=None,
        use_grid_mask=False,
        d_bound=[2.0, 58, 0.5],
        pts_dpt_head=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # self.fp16_enabled = False
        self.freeze_img_backbone = freeze_img_backbone
        self.freeze_img_neck = freeze_img_neck
        self.img_backbone_out_indices = img_backbone_out_indices
        self.use_grid_mask = use_grid_mask
        self.d_bound = d_bound
        self.grid_mask = GridMaskHybrid(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)

        if freeze_img_backbone:
            self.img_backbone.requires_grad_(False)
        if freeze_img_neck:
            self.img_neck.requires_grad_(False)
        if extra_img_backbone is not None:
            self.extra_img_backbone = build_backbone(extra_img_backbone)
        if pts_dpt_head is not None:
            self.pts_dpt_head = build_head(pts_dpt_head)
        if voxelize_lidar is not None:
            self.voxelize_lidar = Voxelization(**voxelize_lidar)
        if lidar_voxel_encoder is not None:
            self.lidar_voxel_encoder = MODELS.build(lidar_voxel_encoder)

    def extract_img_feat(self, imgs, **kwargs):
        """Extract features of images."""
        B = imgs.size(0)

        B, N, C, H, W = imgs.size()
        imgs = imgs.reshape(B * N, C, H, W)
        img_feats_backbone = self.img_backbone(imgs)
        if isinstance(img_feats_backbone, dict):
            img_feats_backbone = list(img_feats_backbone.values())
        img_feats = []
        for idx in self.img_backbone_out_indices:
            img_feats.append(img_feats_backbone[idx])
        img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return {'ms_img_feats': img_feats_reshaped}
    
    @torch.no_grad()
    def voxelize(self, points, **kwargs):
        """Apply dynamic voxelization to points.
        Args:
            points (list[torch.Tensor]): Points of each sample.
        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.voxelize_lidar(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        
        return voxels, num_points, coors_batch
    
    def extract_lidar_feat(self, points, **kwargs):
        """Extract features of lidar."""
        voxels, num_points, coors_batch = self.voxelize(points)
        voxel_lidar_feats = self.lidar_voxel_encoder(voxels, num_points, coors_batch)
        
        return {'voxel_lidar_feats': voxel_lidar_feats, 'coors_batch': coors_batch}
    
    def extract_img_dpt_feat(self, imgs, dpt, **kwargs):
        """Extract features of images."""
        B = imgs.size(0)
        if imgs is not None:
            if imgs.dim() == 5 and imgs.size(0) == 1:
                imgs.squeeze_(0)
                if not (dpt is None):
                    dpt.squeeze_(0)
            elif imgs.dim() == 5 and imgs.size(0) > 1:
                B, N, C, H, W = imgs.size()
                imgs = imgs.reshape(B * N, C, H, W)
                if not (dpt is None):
                    _, _, C_dpt, _, _ = dpt.size()
                    dpt = dpt.reshape(B * N, C_dpt, H, W)
            
            # data augmentation
            if self.use_grid_mask:
                if not (dpt is None):
                    imgs, dpt = self.grid_mask(imgs, dpt)
                else:
                    imgs = self.grid_mask(imgs)

            img_feats = self.img_backbone(imgs)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
            
        return {'ms_img_feats': img_feats_reshaped, 'dpt_masked': dpt}
    
    def extract_multiscale_dpt(self, ms_img_feats, dpt_masked, metas, **kwargs):
        """Extract multi-scale depth features."""
        lidar2img = metas["projection_mat"].to(ms_img_feats[0].device)
        dpt_dist, out_dpt_multiscale = self.pts_dpt_head(ms_img_feats, lidar2img.flatten(2), dpt_masked, return_dpt=True)    
        out_dpt_multiscale = [outdpt.view(*lidar2img.shape[:2], *outdpt.shape[1:]) for outdpt in out_dpt_multiscale]

        return {'dpt_dist': dpt_dist, 'out_dpt_multiscale': out_dpt_multiscale}    
    
    def forward_extra_img_backbone(self, imgs, **kwargs):
        """Extract features of images."""
        B, N, C, H, W = imgs.size()
        imgs = imgs.reshape(B * N, C, H, W)
        img_feats_backbone = self.extra_img_backbone(imgs)

        if isinstance(img_feats_backbone, dict):
            img_feats_backbone = list(img_feats_backbone.values())

        img_feats_backbone_reshaped = []
        for img_feat_backbone in img_feats_backbone:
            BN, C, H, W = img_feat_backbone.size()
            img_feats_backbone_reshaped.append(
                img_feat_backbone.view(B, int(BN / B), C, H, W))
        return img_feats_backbone_reshaped

    def forward(self,
                imgs=None,
                metas=None,
                points=None,
                dpt=None,
                extra_backbone=False,
                occ_only=False,
                rep_only=False,
                **kwargs,
        ):
        """Forward training function.
        """
        if extra_backbone:
            return self.forward_extra_img_backbone(imgs=imgs)
        
        results = {
            'imgs': imgs,
            'metas': metas,
            'points': points,
            'dpt': dpt,
        }
        results.update(kwargs)
        outs = self.extract_img_dpt_feat(**results)
        results.update(outs)
        outs = self.extract_multiscale_dpt(**results)
        results.update(outs)
        outs = self.extract_lidar_feat(**results)
        results.update(outs)
        outs = self.lifter(**results)
        results.update(outs)
        outs = self.encoder(**results)
        if rep_only:
            return outs['representation']
        results.update(outs)
        if occ_only and hasattr(self.head, "forward_occ"):
            outs = self.head.forward_occ(**results)
        else:
            outs = self.head(**results)
        results.update(outs)
        return results