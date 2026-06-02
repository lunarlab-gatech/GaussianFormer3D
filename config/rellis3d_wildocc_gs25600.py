_base_ = [
    './_base_/misc.py',
    './_base_/model.py',
    './_base_/wildocc_pcd_dfa3d.py'
]

# =========== data config ==============
input_shape = (1920, 1200)
# data_aug_conf = {
#     "resize_lim": (1.0, 1.0),
#     "final_dim": input_shape[::-1],
#     "bot_pct_lim": (0.0, 0.0),
#     "rot_lim": (0.0, 0.0),
#     "H": 900,
#     "W": 1600,
#     "rand_flip": True,
# }

# val_dataset_config = dict(
#     data_aug_conf=data_aug_conf
# )
# train_dataset_config = dict(
#     data_aug_conf=data_aug_conf
# )
# =========== misc config ==============
optimizer = dict(
    optimizer = dict(
        type="AdamW", lr=3e-4, weight_decay=0.01,
    ),
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1)}
    )
)
grad_max_norm = 35
max_epochs = 20
# ========= model config ===============
loss = dict(
    type='MultiLoss',
    loss_cfgs=[
        dict(
            type='OccupancyLoss',
            weight=1.0,
            empty_label=8,
            num_classes=9,
            use_focal_loss=False,
            use_dice_loss=False,
            balance_cls_weight=True,
            multi_loss_weights=dict(
                loss_voxel_ce_weight=1.0,
                loss_voxel_lovasz_weight=1.0),
            use_sem_geo_scal_loss=False,
            use_lovasz_loss=True,
            lovasz_ignore=8,
            manual_class_weight=[0.7701, 0.6606, 0.6658, 0.6964, 0.9002, 0.8705, 1.9400, 1.9963, 0.5000]
            )
        ])

loss_input_convertion = dict(
    pred_occ="pred_occ",
    sampled_xyz="sampled_xyz",
    sampled_label="sampled_label",
    occ_mask="occ_mask"
)
# ========= model config ===============
embed_dims = 128
num_decoder = 4
num_single_frame_decoder = 1
pc_range = [-20.0, -10.0, -2.0, 0.0, 10.0, 6.0]
scale_range = [0.01, 0.72]
xyz_coordinate = 'cartesian'
phi_activation = 'sigmoid'
include_opa = True
load_from = 'ckpts/r101_dcn_fcos3d_pretrain.pth'
semantics = True
semantic_dim = 8
d_bound=[1.0, 24, 0.2]
downsample_factors = [8, 16, 32, 64]
indice_layer_depthnet = 2
voxel_size = [0.075, 0.075, 0.2]

model = dict(
    type="BEVSegmentorLiDAR3D",
    use_grid_mask=True,
    d_bound=d_bound,
    pts_dpt_head = dict(
        type='DepthHead_GTDpt',
        max_tol=2, 
        in_channels=embed_dims, 
        cam_channel=0,  
        mid_channels=embed_dims,
        out_channels=embed_dims, 
        downsample_factor=downsample_factors[indice_layer_depthnet],
        dbound=d_bound,
        loss_weight=0.5,
        indice_layer=indice_layer_depthnet,
        sfm_or_sig=True),
    voxelize_lidar=dict(
        max_num_points=10,
        point_cloud_range=pc_range,
        voxel_size=voxel_size,
        max_voxels=[120000, 160000]),
    lidar_voxel_encoder=dict(
        type='HardSimpleVFE',
        num_features=5),
    img_backbone_out_indices=[0, 1, 2, 3],
    img_backbone=dict(
        _delete_=True,
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        with_cp = True,
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        start_level=1),
    lifter=dict( #TODO: 
        type='GaussianLifterLiDAR',
        num_anchor=25600,
        embed_dims=embed_dims,
        anchor_grad=True,
        feat_grad=False,
        phi_activation=phi_activation,
        semantics=semantics,
        semantic_dim=semantic_dim,
        include_opa=include_opa,
        use_intensity=True,
        occ_annotation="wildocc",
    ),
    encoder=dict(
        type='GaussianOccEncoder3D',
        anchor_encoder=dict(
            type='SparseGaussian3DEncoder',
            embed_dims=embed_dims, 
            include_opa=include_opa,
            semantics=semantics,
            semantic_dim=semantic_dim
        ),
        norm_layer=dict(type="LN", normalized_shape=embed_dims),
        ffn=dict(
            type="AsymmetricFFN",
            in_channels=embed_dims * 2,
            embed_dims=embed_dims,
            feedforward_channels=embed_dims * 4,
        ),
        deformable_model=dict(
            type='DeformableFeatureAggregation3D',
            embed_dims=embed_dims,
            kps_generator=dict(
                type="SparseGaussian3DKeyPointsGenerator3D",
                embed_dims=embed_dims,
                phi_activation=phi_activation,
                xyz_coordinate=xyz_coordinate,
                num_learnable_pts=2,
                fix_scale=[
                    [0, 0, 0],
                    [0.15, 0, 0],
                    [-0.15, 0, 0],
                    [0, 0.15, 0],
                    [0, -0.15, 0],
                    [0, 0, 0.15],
                    [0, 0, -0.15],
                ],
                pc_range=pc_range,
                scale_range=scale_range
            ),
            d_bound=d_bound,
            im2col_step=32,
            use_visibility=False,
            use_sampling_offsets=True,
            num_pts_per_keypoint=2,
            value_projection=False,
            num_cams=1
        ),
        refine_layer=dict(
            type='SparseGaussian3DRefinementModule',
            embed_dims=embed_dims,
            pc_range=pc_range,
            scale_range=scale_range,
            restrict_xyz=True,
            unit_xyz=[4.0, 4.0, 1.0],
            refine_manual=[0, 1, 2],
            phi_activation=phi_activation,
            semantics=semantics,
            semantic_dim=semantic_dim,
            include_opa=include_opa,
            xyz_coordinate=xyz_coordinate,
            semantics_activation='softplus',
        ),
        spconv_layer=dict(
            _delete_=True,
            type="SparseConv3D",
            in_channels=embed_dims,
            embed_channels=embed_dims,
            pc_range=pc_range,
            grid_size=[0.2, 0.2, 0.2],
            phi_activation=phi_activation,
            xyz_coordinate=xyz_coordinate,
            use_out_proj=True,
        ),
        num_decoder=num_decoder,
        num_single_frame_decoder=num_single_frame_decoder,
        operation_order=[
            "deformable",
            "ffn",
            "norm",
            "refine",
        ] * num_single_frame_decoder + [
            "spconv",
            "norm",
            "deformable",
            "ffn",
            "norm",
            "refine",
        ] * (num_decoder - num_single_frame_decoder),
    ),
    head=dict(
        type='GaussianHead',
        apply_loss_type='random_1',
        num_classes=semantic_dim + 1,
        empty_args=dict(
            _delete_=True,
            mean=[-10, 0, 2],
            scale=[20, 20, 8],
        ),
        with_empty=True,
        empty_label=8,
        cuda_kwargs=dict(
            _delete_=True,
            scale_multiplier=3,
            H=100, W=100, D=40,
            pc_min=[-20, -10, -2],
            grid_size=0.2),
    )
)
