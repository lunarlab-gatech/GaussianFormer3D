import os
import torch
import numpy as np
from numpy import random
import mmcv
from PIL import Image
import math

from . import OPENOCC_TRANSFORMS

from .loading_utils import load_augmented_point_cloud, reduce_LiDAR_beams
# from mmdet3d.core.points import BasePoints, get_points_type
from mmdet3d.structures.points import BasePoints, get_points_type
import mmengine

"""
A series of instances from the Registry class: OPENOCC_TRANSFORMS
"""

@OPENOCC_TRANSFORMS.register_module()
class DefaultFormatBundle(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img",
    "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    - gt_masks: (1)to tensor, (2)to DataContainer (cpu_only=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor,
                       (3)to DataContainer (stack=True)
    """

    def __init__(self, ):
        return

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        # If the key is not in the results, skip it.
        if 'img' in results:
            if isinstance(results['img'], list):
                # process multiple imgs in single frame
                # original shape: [H, W, C]
                # new shape: [C, H, W]
                imgs = [img.transpose(2, 0, 1) for img in results['img']]
                # shape: [N, C, H, W]
                imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
            else:
                imgs = np.ascontiguousarray(results['img'].transpose(2, 0, 1))
            # transform to tensor and with shape of [N, C, H, W]
            results['img'] = torch.from_numpy(imgs)

        if 'points' in results:
            assert isinstance(results["points"], BasePoints)
            results["points"] = results["points"].tensor
            # results["points"] = torch.from_numpy(results["points"])

        if 'lidar_feature_maps' in results:
            assert isinstance(results["lidar_feature_maps"], dict)
            for key, value in results["lidar_feature_maps"].items():
                assert isinstance(value, np.ndarray)
                results["lidar_feature_maps"][key] = torch.from_numpy(value)

        if 'dpt' in results:
            if isinstance(results['dpt'], list):
                dpts = results['dpt'] # list of (H, W)
                # (N, 1, H, W)
                dpts = np.ascontiguousarray(np.stack(dpts, axis=0))
                if len(dpts.shape) == 2:
                    dpts = dpts[None, :, :]
                dpts = dpts[:, None, :, :]
            else:
                dpts = np.ascontiguousarray(results['dpt'][None,:,:])
            # from numpy to tensor
            results['dpt'] = torch.from_numpy(dpts) # (N, 1, H, W)
        
        if 'anchor_points' in results:
            assert isinstance(results["anchor_points"], np.ndarray)
            results["anchor_points"] = torch.from_numpy(results["anchor_points"])

        if 'occ3d_mask_camera' in results:
            assert isinstance(results["occ3d_mask_camera"], np.ndarray)
            results["occ3d_mask_camera"] = torch.from_numpy(results["occ3d_mask_camera"])
        
        return results

    def __repr__(self):
        return self.__class__.__name__


@OPENOCC_TRANSFORMS.register_module()
class NuScenesAdaptor(object):
    def __init__(self, num_cams, use_ego=False):
        self.num_cams = num_cams
        self.projection_key = 'ego2img' if use_ego else 'lidar2img'
        self.T_key = 'ego2global' if use_ego else 'lidar2global'
        pass

    def __call__(self, input_dict):
        input_dict["projection_mat"] = np.float32(
            np.stack(input_dict[self.projection_key])
        )
        if len(input_dict["projection_mat"].shape) == 2:
            input_dict["projection_mat"] = input_dict["projection_mat"][None, :, :]
        input_dict["image_wh"] = np.ascontiguousarray(
            np.array(input_dict["img_shape"], dtype=np.float32)[:, :2][:, ::-1]
        )
        # input_dict["T_global_inv"] = np.linalg.inv(input_dict[self.T_key])
        # input_dict["T_global"] = input_dict[self.T_key]
        # if "cam_intrinsic" in input_dict:
        #     input_dict["cam_intrinsic"] = np.float32(
        #         np.stack(input_dict["cam_intrinsic"]))
        #     input_dict["focal"] = input_dict["cam_intrinsic"][..., 0, 0]
        # input_dict["extrinsics"] = input_dict["lidar2temCam"]
        # input_dict["intrinsics"] = input_dict["ori_intrinsic"][..., :3, :3]
        return input_dict


@OPENOCC_TRANSFORMS.register_module()
class ResizeCropFlipImage(object):
    def __call__(self, results):
        aug_configs = results.get("aug_configs")
        if aug_configs is None:
            return results
        resize, resize_dims, crop, flip, rotate = aug_configs # aug_configs = [resize, resize_dims, crop, flip, rotate]
        imgs = results["img"]
        N = len(imgs) # N: number of views
        new_imgs = []
        if 'dpt' in results:
            dpts = results['dpt']
            new_dpts = []
        for i in range(N):
            # here imgs[i] is in shape of [H, W, C]
            # data is always in [H, W, C] format but Image processes them in [W, H, C] format!!
            img = Image.fromarray(np.uint8(imgs[i]))
            img, ida_mat = self._img_transform(
                img,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            if 'dpt' in results:
                dpt = dpts[i]
                if dpt.dtype != np.uint8:
                    dpt = Image.fromarray(dpt, mode='F')  # keeps float32 depth values
                else:
                    dpt = Image.fromarray(dpt, mode='L')  # uint8 depth
                dpt, _ = self._img_transform(
                    dpt,
                    resize=resize,
                    resize_dims=resize_dims,
                    crop=crop,
                    flip=flip,
                    rotate=rotate,
                )
                new_dpts.append(np.array(dpt).astype(np.float32))

            mat = np.eye(4)
            mat[:3, :3] = ida_mat # Rotation matrix
            # Store the transformed image to new_imgs
            new_imgs.append(np.array(img).astype(np.float32))
            results["lidar2img"][i] = mat @ results["lidar2img"][i]
            results["ego2img"][i] = mat @ results["ego2img"][i]
            if "cam_intrinsic" in results:
                results["cam_intrinsic"][i][:3, :3] *= resize

        # Update the results with the new images
        results["img"] = new_imgs
        if 'dpt' in results:
            results['dpt'] = new_dpts
        results["img_shape"] = [x.shape[:2] for x in new_imgs]
        return results

    def _get_rot(self, h):
        return torch.Tensor(
            [
                [np.cos(h), np.sin(h)],
                [-np.sin(h), np.cos(h)],
            ]
        )

    def _img_transform(self, img, resize, resize_dims, crop, flip, rotate):
        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        ida_rot *= resize
        ida_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b
        A = self._get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
        ida_mat = torch.eye(3)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran
        return img, ida_mat


@OPENOCC_TRANSFORMS.register_module()
class NormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        results["img"] = [
            mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
            for img in results["img"]
        ]
        results["img_norm_cfg"] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb
        )
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})"
        return repr_str


@OPENOCC_TRANSFORMS.register_module()
class PhotoMetricDistortionMultiViewImage:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(
        self,
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
    ):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = results["img"]
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, (
                "PhotoMetricDistortion needs the input image of dtype np.float32,"
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            )
            # random brightness
            if random.randint(2):
                delta = random.uniform(
                    -self.brightness_delta, self.brightness_delta
                )
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(
                        self.contrast_lower, self.contrast_upper
                    )
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(
                    self.saturation_lower, self.saturation_upper
                )

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(
                        self.contrast_lower, self.contrast_upper
                    )
                    img *= alpha

            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]
            new_imgs.append(img)
        # Update new image list
        results["img"] = new_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(\nbrightness_delta={self.brightness_delta},\n"
        repr_str += "contrast_range="
        repr_str += f"{(self.contrast_lower, self.contrast_upper)},\n"
        repr_str += "saturation_range="
        repr_str += f"{(self.saturation_lower, self.saturation_upper)},\n"
        repr_str += f"hue_delta={self.hue_delta})"
        return repr_str


@OPENOCC_TRANSFORMS.register_module()
class LoadMultiViewImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool, optional): Whether to convert the img to float32.
            Defaults to False.
        color_type (str, optional): Color type of the file.
            Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_filename']
        # img is of shape (h, w, c, num_views)
        img = np.stack(
            [mmcv.imread(name, self.color_type) for name in filename], axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formatting.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])] # unpack multi-view images to list
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@OPENOCC_TRANSFORMS.register_module()
class LoadPointFromFileLiDAR(object):
    """Load LiDAR Points From File.

    Load sunrgbd and scannet points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int]): Which dimensions of the points to be used.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): Whether to use shifted height. Defaults to False.
        use_color (bool): Whether to use color features. Defaults to False.
    """

    def __init__(
        self,
        coord_type,
        load_dim=6, # for our case: 5
        use_dim=[0, 1, 2], # for our case: 5
        shift_height=False,
        use_color=False,
        load_augmented=None,
        reduce_beams=None,
    ):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert (
            max(use_dim) < load_dim
        ), f"Expect all used dimensions < {load_dim}, got {use_dim}"
        assert coord_type in ["CAMERA", "LIDAR", "DEPTH"]

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.load_augmented = load_augmented
        self.reduce_beams = reduce_beams

    def _load_points(self, lidar_path):
        """Private function to load point clouds data.

        Args:
            lidar_path (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        # mmcv.check_file_exist(lidar_path)
        mmengine.check_file_exist(lidar_path)
        if self.load_augmented:
            assert self.load_augmented in ["pointpainting", "mvp"]
            virtual = self.load_augmented == "mvp"
            points = load_augmented_point_cloud(
                lidar_path, virtual=virtual, reduce_beams=self.reduce_beams
            )
        elif lidar_path.endswith(".npy"):
            points = np.load(lidar_path)
        else:
            points = np.fromfile(lidar_path, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        lidar_path = results["lidar_path"]
        # lidar_path = results["pts_filename"]
        points = self._load_points(lidar_path)
        points = points.reshape(-1, self.load_dim)
        # check reduced beams
        if self.reduce_beams and self.reduce_beams < 32:
            points = reduce_LiDAR_beams(points, self.reduce_beams)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3], np.expand_dims(height, 1), points[:, 3:]], 1
            )
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(
                    color=[
                        points.shape[1] - 3,
                        points.shape[1] - 2,
                        points.shape[1] - 1,
                    ]
                )
            )

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims
        )
        results["points"] = points

        return results
    
    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


@OPENOCC_TRANSFORMS.register_module()
class LoadPointsFromMultiSweepsLiDAR(object):
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(
        self,
        sweeps_num=10,
        load_dim=5,
        use_dim=[0, 1, 2, 4],
        pad_empty_sweeps=False,
        remove_close=False,
        test_mode=False,
        load_augmented=None,
        reduce_beams=None,
    ):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        self.use_dim = use_dim
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode
        self.load_augmented = load_augmented
        self.reduce_beams = reduce_beams

    def _load_points(self, lidar_path):
        """Private function to load point clouds data.

        Args:
            lidar_path (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        # mmcv.check_file_exist(lidar_path)
        mmengine.check_file_exist(lidar_path)
        if self.load_augmented:
            assert self.load_augmented in ["pointpainting", "mvp"]
            virtual = self.load_augmented == "mvp"
            points = load_augmented_point_cloud(
                lidar_path, virtual=virtual, reduce_beams=self.reduce_beams
            )
        elif lidar_path.endswith(".npy"):
            points = np.load(lidar_path)
        else:
            points = np.fromfile(lidar_path, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point \
                    cloud arrays.
        """
        points = results["points"]
        points = points[:, self.use_dim]
        points.tensor[:, 4] = 0
        sweep_points_list = [points]
        ts = results["timestamp"] / 1e6
        if self.pad_empty_sweeps and len(results["sweeps"]) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            if len(results["sweeps"]) <= self.sweeps_num:
                choices = np.arange(len(results["sweeps"]))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                # NOTE: seems possible to load frame -11?
                if not self.load_augmented:
                    choices = np.random.choice(
                        len(results["sweeps"]), self.sweeps_num, replace=False
                    )
                else:
                    # don't allow to sample the earliest frame, match with Tianwei's implementation.
                    choices = np.random.choice(
                        len(results["sweeps"]) - 1, self.sweeps_num, replace=False
                    )
            for idx in choices:
                sweep = results["sweeps"][idx]
                points_sweep = self._load_points(sweep["data_path"])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)

                if self.reduce_beams and self.reduce_beams < 32:
                    points_sweep = reduce_LiDAR_beams(points_sweep, self.reduce_beams)

                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                points_sweep = points_sweep[:, self.use_dim]
                sweep_ts = sweep["timestamp"] / 1e6 # TODO: check sweep["timestamp"], sweep["sensor2lidar_rotation"], sweep["sensor2lidar_translation"]
                points_sweep[:, :3] = (
                    points_sweep[:, :3] @ sweep["sensor2lidar_rotation"].T
                )
                points_sweep[:, :3] += sweep["sensor2lidar_translation"]
                points_sweep[:, 4] = ts - sweep_ts
                # TODO: check whether our points are BasePoints and can use new_point function!
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)

        results["points"] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f"{self.__class__.__name__}(sweeps_num={self.sweeps_num})"


@OPENOCC_TRANSFORMS.register_module()
class LoadMultiViewDepthFromFiles(object):
    """Load the gound truth depth map generated from BEVDepth using lidar.
    """

    def __init__(self, is_to_depth_map=True, map_size=None):
        self.is_to_depth_map = is_to_depth_map
        self.map_size = map_size

    def __call__(self, results):
        if self.map_size is None:
            self.map_size = results['img'][0].shape[:2] # List of (H, W, C) numpy array, here should be (900, 1600)
        img_paths = results['img_filename']
        dpt_paths = []
        map_depths = []
        for img_path in img_paths:
            dpt_path = os.path.join(img_path.split("/samples/")[0], "depth_gt", img_path.split("/")[-1]+".bin")
            point_depth = np.fromfile(dpt_path, dtype=np.float32, count=-1).reshape(-1, 3)
            dpt_paths.append(dpt_path)
            if self.is_to_depth_map:
                map_depth = self.to_depth_map(point_depth)
                map_depths.append(map_depth)
            
        # img is of shape (h, w, c, num_views)
        # map_depths is a List of depth maps, each of shape (H, W)
        # N * (H, W), N = num_views
        results['dpt'] = map_depths
        results['filename_dpt'] = dpt_paths
        return results
    
    def to_depth_map(self, point_depth):
        """Transform depth based on ida augmentation configuration.

        Args:
            cam_depth (np array): Nx3, 3: x,y,d.
            resize (float): Resize factor.
            resize_dims (list): Final dimension.
            crop (list): x1, y1, x2, y2
            flip (bool): Whether to flip.
            rotate (float): Rotation value.

        Returns:
            np array: [h/down_ratio, w/down_ratio, d]
        """

        # Here they assume the point depth coordinates are 900, 1600?
        # TODO: check the point depth coordinate is (H, W) or (W, H)
        depth_coords = point_depth[:, :2].astype(np.int16)

        # The loaded image shape is also 900, 1600?
        depth_map = np.zeros(self.map_size) # (H, W)
        valid_mask = ((depth_coords[:, 1] < self.map_size[0])
                    & (depth_coords[:, 0] < self.map_size[1])
                    & (depth_coords[:, 1] >= 0)
                    & (depth_coords[:, 0] >= 0))
        depth_map[depth_coords[valid_mask, 1],
                depth_coords[valid_mask, 0]] = point_depth[valid_mask, 2]

        return depth_map
    
    def __repr__(self):
        return self.__class__.__name__


@OPENOCC_TRANSFORMS.register_module()
class PadMultiViewImage(object):
    """Pad the multi-view image.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor # 32, TODO: choose (864, 1600), (896, 1600), (928, 1600)
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        if self.size is not None:
            padded_img = [mmcv.impad(
                img, shape=self.size, pad_val=self.pad_val) for img in results['img']]
            if "dpt" in results.keys():
                padded_dpt = [mmcv.impad(
                img, shape=self.size, pad_val=self.pad_val) for img in results['dpt']]
        elif self.size_divisor is not None: # here, 32
            padded_img = [mmcv.impad_to_multiple(
                img, self.size_divisor, pad_val=self.pad_val) for img in results['img']]
            if "dpt" in results.keys():
                padded_dpt = [mmcv.impad_to_multiple(
                    img, self.size_divisor, pad_val=self.pad_val) for img in results['dpt']]
        
        results['ori_shape'] = [img.shape for img in results['img']] # keep track of original shape which should be (900, 1600, 3)
        results['img'] = padded_img # now should be (928, 1600, 3), list of np.array (H, W, C)
        if "dpt" in results.keys():
            results['dpt'] = padded_dpt # now should be (928, 1600), list of np.array (H, W)
        results['img_shape'] = [img.shape for img in padded_img] # padded version will go into image_wh key
        results['pad_shape'] = [img.shape for img in padded_img]
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@OPENOCC_TRANSFORMS.register_module()
class LoadOccupancySurroundOcc(object):

    def __init__(self, occ_path, semantic=False, use_ego=False, use_sweeps=False):
        self.occ_path = occ_path
        self.semantic = semantic
        self.use_ego = use_ego
        assert semantic and (not use_ego)
        self.use_sweeps = use_sweeps

        xyz = self.get_meshgrid([-50, -50, -5.0, 50, 50, 3.0], [200, 200, 16], 0.5)
        self.xyz = np.concatenate([xyz, np.ones_like(xyz[..., :1])], axis=-1) # x, y, z, 4

    def get_meshgrid(self, ranges, grid, reso):
        xxx = torch.arange(grid[0], dtype=torch.float) * reso + 0.5 * reso + ranges[0]
        yyy = torch.arange(grid[1], dtype=torch.float) * reso + 0.5 * reso + ranges[1]
        zzz = torch.arange(grid[2], dtype=torch.float) * reso + 0.5 * reso + ranges[2]

        xxx = xxx[:, None, None].expand(*grid)
        yyy = yyy[None, :, None].expand(*grid)
        zzz = zzz[None, None, :].expand(*grid)

        xyz = torch.stack([
            xxx, yyy, zzz
        ], dim=-1).numpy()
        return xyz # x, y, z, 3

    def __call__(self, results):
        # input is the occupancy annotation file generated by SurroundOcc
        # results['pts_filename'] is the path to the point cloud file
        label_file = os.path.join(self.occ_path, results['pts_filename'].split('/')[-1]+'.npy')
        if os.path.exists(label_file):
            label = np.load(label_file)

            # (200, 200, 16) is the shape of the grid
            # 17 is the number of classes
            new_label = np.ones((200, 200, 16), dtype=np.int64) * 17
            # Give the new label the value of the original label
            new_label[label[:, 0], label[:, 1], label[:, 2]] = label[:, 3]

            # Define a mask to see which grid cells are occupied
            # From SurroundOcc github, 0 is ignored class which is set to be 255. Here we use a mask.
            # In the head, we set empty label to be 17 (no annotation), and the regression number classes is 18: 0, 1-16, 17
            mask = new_label != 0

            # Update results
            results['occ_label'] = new_label if self.semantic else new_label != 17
            results['occ_cam_mask'] = mask
        elif self.use_sweeps:
            new_label = np.ones((200, 200, 16), dtype=np.int64) * 17
            mask = new_label != 0
            results['occ_label'] = new_label if self.semantic else new_label != 17
            results['occ_cam_mask'] = mask
        else:
            raise NotImplementedError
        
        """
        Since SurroundOcc's annotation is in the camera coordinate system, we need to convert the ego frame to the lidar coordinate system if we are using ego!!
        """
        if not self.use_ego:
            occ_xyz = self.xyz[..., :3]
        else:
            ego2lidar = np.linalg.inv(results['ego2lidar']) # 4, 4
            occ_xyz = ego2lidar[None, None, None, ...] @ self.xyz[..., None] # x, y, z, 4, 1
            occ_xyz = np.squeeze(occ_xyz, -1)[..., :3]
        
        results['occ_xyz'] = occ_xyz
        
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


@OPENOCC_TRANSFORMS.register_module()
class LoadOccupancyOcc3d(object):

    def __init__(self, occ_path, semantic=False, use_ego=False, use_occ3d_mask=False, pc_range=[-40.0, -40.0, -1.0, 40.0, 40.0, 5.4], use_lidar=True, use_mask_training=False):
        self.occ_path = occ_path
        self.semantic = semantic
        self.use_ego = use_ego
        assert semantic and use_ego
        self.use_occ3d_mask = use_occ3d_mask
        self.pc_range = pc_range
        self.use_lidar = use_lidar
        self.use_mask_training = use_mask_training

        xyz = self.get_meshgrid([-40, -40, -1, 40, 40, 5.4], [200, 200, 16], 0.4)
        self.xyz = np.concatenate([xyz, np.ones_like(xyz[..., :1])], axis=-1) # x, y, z, 4

    def get_meshgrid(self, ranges, grid, reso):
        xxx = torch.arange(grid[0], dtype=torch.float) * reso + 0.5 * reso + ranges[0]
        yyy = torch.arange(grid[1], dtype=torch.float) * reso + 0.5 * reso + ranges[1]
        zzz = torch.arange(grid[2], dtype=torch.float) * reso + 0.5 * reso + ranges[2]

        xxx = xxx[:, None, None].expand(*grid)
        yyy = yyy[None, :, None].expand(*grid)
        zzz = zzz[None, None, :].expand(*grid)

        xyz = torch.stack([
            xxx, yyy, zzz
        ], dim=-1).numpy()
        return xyz # x, y, z, 3

    def __call__(self, results):
        # First occ_path is to Occ3d folder
        # Second occ_path starts from gts/...
        label_file = os.path.join(self.occ_path, results['occ_path'])
        if os.path.exists(label_file):
            occ3d = np.load(label_file)
            label = occ3d['semantics'] # (200, 200, 16)
            occ3d_mask_camera = occ3d['mask_camera'] # (200, 200, 16)
            occ3d_mask_lidar = occ3d['mask_lidar'] # (200, 200, 16)

            new_label = np.ones((200, 200, 16), dtype=np.int64) * 17 # 17 is empty
            new_label = label

            # Here mask is for void / ignored class
            # mask = new_label != 0

            # Here mask is for invisible cells for the camera
            if self.use_occ3d_mask and self.use_mask_training:
                mask = occ3d_mask_camera
                results['occ3d_mask_camera'] = occ3d_mask_camera
                # print(mask.shape)
            elif self.use_occ3d_mask and not self.use_mask_training:
                mask = np.ones((200, 200, 16), dtype=np.int64)
                results['occ3d_mask_camera'] = occ3d_mask_camera

            # Update results
            results['occ_label'] = new_label
            results['occ_cam_mask'] = mask

            # Occ3d is in ego frame, so we need to convert lidar point cloud to ego frame
            if self.use_ego and self.use_lidar:
                # Convert lidar point cloud to ego frame
                lidar2ego = np.linalg.inv(results['ego2lidar'])
                lidar2ego_rot = lidar2ego[:3, :3]
                lidar2ego_tran = lidar2ego[:3, 3]
                points = results['points'].tensor[..., :3]
                points = points @ lidar2ego_rot.T + lidar2ego_tran
                results['points'].tensor[..., :3] = points

                # filter points outside the range
                new_points = results['points'].tensor
                new_points_mask = (new_points[..., 0] > self.pc_range[0]) & (new_points[..., 0] < self.pc_range[3]) & \
                                    (new_points[..., 1] > self.pc_range[1]) & (new_points[..., 1] < self.pc_range[4]) & \
                                    (new_points[..., 2] > self.pc_range[2]) & (new_points[..., 2] < self.pc_range[5])
                new_points = new_points[new_points_mask]
                results['points'].tensor = new_points

            occ_xyz = self.xyz[..., :3]
            results['occ_xyz'] = occ_xyz
        else:
            raise NotImplementedError
        
        
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


@OPENOCC_TRANSFORMS.register_module()
class LoadPointFromFileLiDARWildOcc(object):
    """Load LiDAR Points From File.

    Load sunrgbd and scannet points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int]): Which dimensions of the points to be used.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): Whether to use shifted height. Defaults to False.
        use_color (bool): Whether to use color features. Defaults to False.
    """

    def __init__(
        self,
        coord_type='LIDAR',
        load_dim=4, # for our case: 4
        use_dim=[0, 1, 2, 3], # for our case: 4
        shift_height=False,
        use_color=False,
        load_augmented=None,
        reduce_beams=None,
    ):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert (
            max(use_dim) < load_dim
        ), f"Expect all used dimensions < {load_dim}, got {use_dim}"
        assert coord_type in ["CAMERA", "LIDAR", "DEPTH"]

        self.coord_type = coord_type
        self.load_dim = load_dim
        assert load_dim == 5
        self.use_dim = use_dim
        self.load_augmented = load_augmented
        self.reduce_beams = reduce_beams

    def _load_points(self, lidar_path):
        """Private function to load point clouds data.

        Args:
            lidar_path (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        # mmcv.check_file_exist(lidar_path)
        mmengine.check_file_exist(lidar_path)
        if lidar_path.endswith(".npy"):
            points = np.load(lidar_path)
        else:
            points = np.fromfile(lidar_path, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        lidar_path = results["lidar_path"]
        # lidar_path = results["pts_filename"]
        points = self._load_points(lidar_path)
        points = points.reshape(-1, self.load_dim-1)
        if points.shape[1] == 4:
            points = np.pad(points, ((0, 0), (0, 1)), mode='constant', constant_values=0.0)
        points[:, 4] = 0
        # check reduced beams
        if self.reduce_beams and self.reduce_beams < 32:
            points = reduce_LiDAR_beams(points, self.reduce_beams)
        points = points[:, self.use_dim]
        attribute_dims = None

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims
        )
        results["points"] = points

        return results
    
    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


@OPENOCC_TRANSFORMS.register_module()
class LoadPointsFromMultiSweepsLiDARWildOcc(object):
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(
        self,
        sweeps_num=10,
        load_dim=5,
        use_dim=[0, 1, 2, 4],
        pad_empty_sweeps=False,
        remove_close=False,
        test_mode=False,
        load_augmented=None,
        reduce_beams=None,
    ):
        self.load_dim = load_dim
        assert load_dim == 5
        self.sweeps_num = sweeps_num
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        self.use_dim = use_dim
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode
        self.load_augmented = load_augmented
        self.reduce_beams = reduce_beams

    def _load_points(self, lidar_path):
        """Private function to load point clouds data.

        Args:
            lidar_path (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        # mmcv.check_file_exist(lidar_path)
        mmengine.check_file_exist(lidar_path)
        if self.load_augmented:
            assert self.load_augmented in ["pointpainting", "mvp"]
            virtual = self.load_augmented == "mvp"
            points = load_augmented_point_cloud(
                lidar_path, virtual=virtual, reduce_beams=self.reduce_beams
            )
        elif lidar_path.endswith(".npy"):
            points = np.load(lidar_path)
        else:
            points = np.fromfile(lidar_path, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point \
                    cloud arrays.
        """
        points = results["points"]
        if points.shape[1] == 4:
            points = np.pad(points, ((0, 0), (0, 1)), mode='constant', constant_values=0.0)
        points = points[:, self.use_dim]
        points.tensor[:, 4] = 0
        sweep_points_list = [points]
        ts = results["timestamp"] / 1e3 # swei: convert to ms, original is 1e6
        if self.pad_empty_sweeps and len(results["sweeps"]) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            if len(results["sweeps"]) <= self.sweeps_num:
                choices = np.arange(len(results["sweeps"]))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                # NOTE: seems possible to load frame -11?
                if not self.load_augmented:
                    choices = np.random.choice(
                        len(results["sweeps"]), self.sweeps_num, replace=False
                    )
                else:
                    # don't allow to sample the earliest frame, match with Tianwei's implementation.
                    choices = np.random.choice(
                        len(results["sweeps"]) - 1, self.sweeps_num, replace=False
                    )
            for idx in choices:
                sweep = results["sweeps"][idx]
                points_sweep = self._load_points(sweep["lidar_path"])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim-1)
                if points_sweep.shape[1] == 4:
                    points_sweep = np.pad(points_sweep, ((0, 0), (0, 1)), mode='constant', constant_values=0.0)

                if self.reduce_beams and self.reduce_beams < 32:
                    points_sweep = reduce_LiDAR_beams(points_sweep, self.reduce_beams)

                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                points_sweep = points_sweep[:, self.use_dim] # [num_points, 4]
                sweep_ts = sweep["timestamp"] / 1e3 # convert to ms, original is 1e6
                # TODO: swei: check whether this is correct
                sweep2lidar = results['lidar_pose'] @ np.linalg.inv(sweep['lidar_pose'])
                sweep2lidar_rotation = sweep2lidar[:3, :3]
                sweep2lidar_translation = sweep2lidar[:3, 3]
                points_sweep[:, :3] = (
                    points_sweep[:, :3] @ sweep2lidar_rotation
                )
                points_sweep[:, :3] += sweep2lidar_translation
                points_sweep[:, 4] = ts - sweep_ts
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)

        results["points"] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f"{self.__class__.__name__}(sweeps_num={self.sweeps_num})"


@OPENOCC_TRANSFORMS.register_module()
class LoadImageFromFilesWildOcc(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool, optional): Whether to convert the img to float32.
            Defaults to False.
        color_type (str, optional): Color type of the file.
            Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['image_path']
        # img is of shape (h, w, c)
        img = mmcv.imread(filename, self.color_type)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        results['img'] = [img] # single-view image
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@OPENOCC_TRANSFORMS.register_module()
class LoadOccupancyWildOcc(object):
    def __init__(self):
        # 3: 'grass', 4: 'tree', 19: 'bush', 31: 'puddle', 33: 'mud', 27: 'barrier', 34: 'rubble'
        # reference: https://github.com/unmannedlab/RELLIS-3D/blob/main/utils/label2color.ipynb
        self.label_wildocc = [3, 4, 19, 31, 33, 27, 34]

        xyz = self.get_meshgrid([-20, -10, -2.0, 0, 10, 6.0], [100, 100, 40], 0.2)
        self.xyz = np.concatenate([xyz, np.ones_like(xyz[..., :1])], axis=-1) # x, y, z, 4

    def get_meshgrid(self, ranges, grid, reso):
        # NOTE: Attention: the index of the grid is the start of the FOV of the camera
        '''
                     |
                     |------o (start the index of the grid)
                     |      |
                   \ | /    |  (\ / is the FOV of the camera)
        y <----------o----------------
                     |
                     |
                     v
                     x
        '''
        xxx = torch.arange(grid[0], dtype=torch.float) * reso + 0.5 * reso + ranges[0]
        yyy = torch.arange(grid[1], dtype=torch.float) * reso + 0.5 * reso + ranges[1]
        zzz = torch.arange(grid[2], dtype=torch.float) * reso + 0.5 * reso + ranges[2]

        xxx = xxx[:, None, None].expand(*grid)
        yyy = yyy[None, :, None].expand(*grid)
        zzz = zzz[None, None, :].expand(*grid)

        xyz = torch.stack([
            xxx, yyy, zzz
        ], dim=-1).numpy()
        return xyz # x, y, z, 3

    def __call__(self, results):
        # input is the occupancy annotation file generated by SurroundOcc
        # results['pts_filename'] is the path to the point cloud file
        label_file = results['occ_path']
        label = np.load(label_file)
        # The shape of each npy file is (n,4), where n is the number of non-empty occupancies. Four dimensions represent xyz and semantic label respectively.

        # 17 is the number of classes
        new_label = np.ones((100, 100, 40), dtype=np.int64) * 8
        # Give the new label the value of the original label
        for idx, label_id in enumerate(self.label_wildocc):
            indices = label[label[:, 3] == label_id][:, :3]  # (N, 3)
            new_label[indices[:, 0], indices[:, 1], indices[:, 2]] = idx + 1 
        indices = label[~np.isin(label[:, 3], self.label_wildocc)][:, :3]
        new_label[indices[:, 0], indices[:, 1], indices[:, 2]] = 0

        # Define a mask to see which grid cells are occupied
        # From SurroundOcc github, 0 is ignored class which is set to be 255. Here we use a mask.
        # In the head, we set empty label to be 17 (no annotation), and the regression number classes is 9: 0, 1-7, 17
        mask = new_label != 0

        # Update results
        results['occ_label'] = new_label
        results['occ_cam_mask'] = mask
        
        """
        Since SurroundOcc's annotation is in the camera coordinate system, we need to convert the ego frame to the lidar coordinate system if we are using ego!!
        """
        occ_xyz = self.xyz[..., :3]
        
        results['occ_xyz'] = occ_xyz
        
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


@OPENOCC_TRANSFORMS.register_module()
class LoadDepthFromFileWildOcc(object):
    """Load the gound truth depth map generated from BEVDepth using lidar.
    """

    def __init__(self, is_to_depth_map=True, map_size=None):
        self.is_to_depth_map = is_to_depth_map
        self.map_size = map_size

    def __call__(self, results):
        if self.map_size is None:
            self.map_size = results['img'][0].shape[:2] # List of (H, W, C) numpy array, here should be (1200, 1920)
        img_paths = results['img_filename']
        dpt_paths = []
        map_depths = []

        dpt_path = results['depth_path']
        point_depth = np.fromfile(dpt_path, dtype=np.float32, count=-1).reshape(-1, 3)
        dpt_paths.append(dpt_path)
        if self.is_to_depth_map:
            map_depth = self.to_depth_map(point_depth)
            map_depths.append(map_depth)
        
        # img is of shape (h, w, c, num_views)
        # map_depths is a List of depth maps, each of shape (H, W)
        # N * (H, W), N = num_views
        results['dpt'] = map_depths
        results['filename_dpt'] = dpt_paths
        return results
    
    def to_depth_map(self, point_depth):
        """Transform depth based on ida augmentation configuration.

        Args:
            cam_depth (np array): Nx3, 3: x,y,d.
            resize (float): Resize factor.
            resize_dims (list): Final dimension.
            crop (list): x1, y1, x2, y2
            flip (bool): Whether to flip.
            rotate (float): Rotation value.

        Returns:
            np array: [h/down_ratio, w/down_ratio, d]
        """

        # Here they assume the point depth coordinates are 900, 1600?
        # TODO: check the point depth coordinate is (H, W) or (W, H)
        depth_coords = point_depth[:, :2].astype(np.int16)

        # The loaded image shape is also 900, 1600?
        depth_map = np.zeros(self.map_size) # (H, W)
        valid_mask = ((depth_coords[:, 1] < self.map_size[0])
                    & (depth_coords[:, 0] < self.map_size[1])
                    & (depth_coords[:, 1] >= 0)
                    & (depth_coords[:, 0] >= 0))
        depth_map[depth_coords[valid_mask, 1],
                depth_coords[valid_mask, 0]] = point_depth[valid_mask, 2]

        return depth_map
    
    def __repr__(self):
        return self.__class__.__name__
