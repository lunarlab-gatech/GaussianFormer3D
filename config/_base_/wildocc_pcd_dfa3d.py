# ================== data ========================
import os
data_root = "data/WildOcc"
anno_root = "Rellis-3D-split/Rellis_3D_lidar_split"

input_shape = (704, 256)
batch_size = 1

img_norm_cfg = dict(
    mean=[145.95516077, 164.89545458, 140.05822218], 
    std=[74.06626308, 54.52684476, 51.01978296], 
    to_rgb=True
)


train_pipeline = [
    dict(type="LoadPointFromFileLiDARWildOcc", coord_type="LIDAR", load_dim=5, use_dim=5),
    dict(type="LoadPointsFromMultiSweepsLiDARWildOcc", sweeps_num=10, load_dim=5, use_dim=5, pad_empty_sweeps=True, remove_close=True),
    dict(type="LoadImageFromFilesWildOcc", to_float32=True),
    dict(type="LoadOccupancyWildOcc"),
    # dict(type="ResizeCropFlipImage"),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type='LoadDepthFromFileWildOcc', is_to_depth_map=True, map_size=None),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type='PadMultiViewImage', size=(1216, 1920)),
    dict(type="DefaultFormatBundle"),
    dict(type="NuScenesAdaptor", use_ego=False, num_cams=1),
]

test_pipeline = [
    dict(type="LoadPointFromFileLiDARWildOcc", coord_type="LIDAR", load_dim=5, use_dim=5),
    dict(type="LoadPointsFromMultiSweepsLiDARWildOcc", sweeps_num=10, load_dim=5, use_dim=5, pad_empty_sweeps=True, remove_close=True),
    dict(type="LoadImageFromFilesWildOcc", to_float32=True),
    dict(type="LoadOccupancyWildOcc"),
    # dict(type="ResizeCropFlipImage"),
    dict(type='LoadDepthFromFileWildOcc', is_to_depth_map=True, map_size=None),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type='PadMultiViewImage', size=(1216, 1920)),
    dict(type="DefaultFormatBundle"),
    dict(type="NuScenesAdaptor", use_ego=False, num_cams=1),
]

data_aug_conf = None

train_dataset_config = dict(
    type='WildOccDataset',
    data_root=data_root,
    split_file=os.path.join(data_root, anno_root, "pt_train.pkl"),
    data_aug_conf=data_aug_conf,
    pipeline=train_pipeline,
    phase='train'
)

val_dataset_config = dict(
    type='WildOccDataset',
    data_root=data_root,
    split_file=os.path.join(data_root, anno_root, "pt_val.pkl"),
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