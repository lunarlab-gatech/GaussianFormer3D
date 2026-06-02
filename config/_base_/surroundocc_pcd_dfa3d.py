# ================== data ========================
data_root = "data/nuscenes/"
anno_root = "data/nuscenes_cam/"
occ_path = "data/surroundocc/samples"
input_shape = (704, 256)
batch_size = 1

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

train_pipeline = [
    dict(type="LoadPointFromFileLiDAR", coord_type="LIDAR", load_dim=5, use_dim=5),
    dict(type="LoadPointsFromMultiSweepsLiDAR", sweeps_num=10, load_dim=5, use_dim=5, pad_empty_sweeps=True, remove_close=True),
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="LoadOccupancySurroundOcc", occ_path=occ_path, semantic=True, use_ego=False),
    # dict(type="ResizeCropFlipImage"),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type='LoadMultiViewDepthFromFiles', is_to_depth_map=True, map_size=None),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="NuScenesAdaptor", use_ego=False, num_cams=6),
]

test_pipeline = [
    dict(type="LoadPointFromFileLiDAR", coord_type="LIDAR", load_dim=5, use_dim=5),
    dict(type="LoadPointsFromMultiSweepsLiDAR", sweeps_num=10, load_dim=5, use_dim=5, pad_empty_sweeps=True, remove_close=True),
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="LoadOccupancySurroundOcc", occ_path=occ_path, semantic=True, use_ego=False),
    # dict(type="ResizeCropFlipImage"),
    dict(type='LoadMultiViewDepthFromFiles', is_to_depth_map=True, map_size=None),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="NuScenesAdaptor", use_ego=False, num_cams=6),
]

data_aug_conf = None
 
train_dataset_config = dict(
    type='NuScenesDataset',
    data_root=data_root,
    imageset=anno_root + "nuscenes_infos_gf3d_train.pkl",
    data_aug_conf=data_aug_conf,
    pipeline=train_pipeline,
    phase='train'
)

val_dataset_config = dict(
    type='NuScenesDataset',
    data_root=data_root,
    imageset=anno_root + "nuscenes_infos_gf3d_val.pkl",
    data_aug_conf=data_aug_conf,
    pipeline=test_pipeline,
    phase='val'
)

train_loader = dict(
    batch_size=batch_size,
    num_workers=2,
    shuffle=True
)

val_loader = dict(
    batch_size=batch_size,
    num_workers=2
)