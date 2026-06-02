import os
from copy import deepcopy
import numpy as np
from pyquaternion import Quaternion
from torch.utils.data import Dataset

import mmengine
from . import OPENOCC_DATASET, OPENOCC_TRANSFORMS
from .utils import get_img2global, get_lidar2global
import pickle

@OPENOCC_DATASET.register_module() # an subclass of the Registry class: OPENOCC_DATASET
class WildOccDataset(Dataset):

    def __init__(
        self,
        data_root=None,
        split_file=None,
        data_aug_conf=None,
        pipeline=None,
        vis_indices=None,
        num_samples=0,
        phase='train',
        num_sweeps = 4
    ):
        self.data_path = data_root
        with open(split_file, 'rb') as f:
            data = pickle.load(f)
        
        self.scene_infos = data['infos']
        self.metadata = data['metadata']
        self.num_sweeps = num_sweeps
        self.accumulate_sweeps_end_idx_list = []
        self.samples_idx_list = []
        for idx, (scene_token, scene_params) in enumerate(self.scene_infos.items()):
            num_samples = scene_params['num_samples'] - (self.num_sweeps - 1)
            if idx == 0:
                samples_idx = list(np.arange(num_samples) + self.num_sweeps - 1)
                self.accumulate_sweeps_end_idx_list.append(num_samples - 1)
            else:
                samples_idx = list(np.arange(num_samples) + self.accumulate_sweeps_end_idx_list[-1] + self.num_sweeps)
                self.accumulate_sweeps_end_idx_list.append(self.accumulate_sweeps_end_idx_list[-1] + num_samples)
            self.samples_idx_list.extend(samples_idx)

        self.data_aug_conf = data_aug_conf
        self.test_mode = (phase != 'train')
        self.pipeline = []

        for t in pipeline:
            self.pipeline.append(OPENOCC_TRANSFORMS.build(t))

    def _sample_augmentation(self):
        '''
        This function isn't used in the code.
        '''
        return None, None, None, None, None
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fH, fW = self.data_aug_conf["final_dim"]
        if not self.test_mode:
            resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                int(
                    (1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"]))
                    * newH
                )
                - fH
            )
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH)
                - fH
            )
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def __getitem__(self, index):

        input_dict = self.get_data_info(index)

        if self.data_aug_conf is not None:
            input_dict["aug_configs"] = self._sample_augmentation()
        for t in self.pipeline:
            input_dict = t(input_dict)
        
        return_dict = {
            'img': input_dict['img'],
            'projection_mat': input_dict['projection_mat'],
            'image_wh': input_dict['image_wh'],
            'occ_label': input_dict['occ_label'],
            'occ_xyz': input_dict['occ_xyz'],
            'occ_cam_mask': input_dict['occ_cam_mask']
        }
        if 'lidar_feature_maps' in input_dict:
            return_dict['lidar_feature_maps'] = input_dict['lidar_feature_maps']
        if 'points' in input_dict:
            return_dict['points'] = input_dict['points']
        if 'img_filename' in input_dict:
            return_dict['img_filename'] = input_dict['img_filename']
        if 'dpt' in input_dict:
            return_dict['dpt'] = input_dict['dpt']
        if 'anchor_points' in input_dict:
            return_dict['anchor_points'] = input_dict['anchor_points']
        if 'occ3d_mask_camera' in input_dict:
            return_dict['occ3d_mask_camera'] = input_dict['occ3d_mask_camera']
        if 'depth_path' in input_dict:
            return_dict['depth_path'] = input_dict['depth_path']
        if 'lidar_path' in input_dict:
            return_dict['lidar_path'] = input_dict['lidar_path']
        return return_dict
    
    def get_data_info(self, index):
        info = deepcopy(self.metadata[self.samples_idx_list[index]])
        fx, fy, cx, cy = self.scene_infos[info['scene_token']]['cam_intrinsic_data']
        K = np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]])

        # Construct 4x4 view transformation matrix (for homogeneous coordinates)
        viewpad = np.eye(4)
        viewpad[:3, :3] = K

        # Parse LiDAR → Camera rotation and translation
        q = self.scene_infos[info['scene_token']]['cam2lidar_q']
        t = self.scene_infos[info['scene_token']]['cam2lidar_r']

        # Use pyquaternion to parse quaternion (w, x, y, z)
        quat = Quaternion(q["w"], q["x"], q["y"], q["z"])
        R_cam2lidar = quat.rotation_matrix
        T_cam2lidar = np.array([t["x"], t["y"], t["z"]])

        # Construct 4x4 LiDAR → Camera transformation matrix
        cam2lidar = np.eye(4)
        cam2lidar[:3, :3] = R_cam2lidar
        cam2lidar[:3, 3] = T_cam2lidar

        lidar2cam = np.linalg.inv(cam2lidar)

        # Calculate img2cam transformation matrix (inverse of K)
        img2cam = np.linalg.inv(viewpad)

        # Calculate img2lidar transformation matrix
        img2lidar = cam2lidar @ img2cam

        lidar2img = np.linalg.inv(img2lidar)

        input_dict =dict(
            lidar_path=info['lidar_path'],
            occ_path=info['occ_path'],
            depth_path=info['depth_path'],
            image_path=info['image_path'],
            img2lidar=img2lidar,
            img2cam=img2cam,
            lidar2img=lidar2img,
            cam2lidar=cam2lidar,
            lidar2cam=lidar2cam,
            scene_token=info['scene_token'],
            lidar_frame_token=info['lidar_frame_token'],
            image_frame_token=info['image_frame_token'],
            timestamp=info["timestamp"],
            lidar_pose=info['lidar_pose'],
            img_filename=[info['image_path']]
        )
        
        sweeps = []
        for sweep_idx in range(1, self.num_sweeps):
            sweep_info = deepcopy(self.metadata[self.samples_idx_list[index - sweep_idx]])
            sweep_data = dict(
                lidar_path = sweep_info['lidar_path'],
                lidar_pose = sweep_info['lidar_pose'],
                timestamp = sweep_info['timestamp']
            )
            sweeps.append(sweep_data)
        input_dict['sweeps'] = sweeps
        
        return input_dict

    def __len__(self):
        return len(self.samples_idx_list)
