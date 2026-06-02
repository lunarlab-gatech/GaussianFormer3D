import numpy as np
from pyquaternion import Quaternion
import torch


# Not used in this project
def get_rm(angle, axis, deg=False):
    if deg:
        angle = np.deg2rad(angle)
    rm = np.eye(3)
    if axis == 'x':
        rm[1, 1] = np.cos(angle)
        rm[2, 2] = np.cos(angle)
        rm[1, 2] = - np.sin(angle)
        rm[2, 1] = np.sin(angle)
    elif axis == 'y':
        rm[0, 0] = np.cos(angle)
        rm[2, 2] = np.cos(angle)
        rm[0, 2] = np.sin(angle)
        rm[2, 0] = - np.sin(angle)
    elif axis == 'z':
        rm[0, 0] = np.cos(angle)
        rm[1, 1] = np.cos(angle)
        rm[0, 1] = - np.sin(angle)
        rm[1, 0] = np.sin(angle)
    return rm

# Not used in this project
def get_xyz(pose_dict):
    return np.array(pose_dict['translation'])

# Used in dataset/dataset.py
def get_img2global(calib_dict, pose_dict):
    
    cam2img = np.eye(4)
    cam2img[:3, :3] = np.asarray(calib_dict['camera_intrinsic'])
    img2cam = np.linalg.inv(cam2img)

    cam2ego = np.eye(4)
    cam2ego[:3, :3] = Quaternion(calib_dict['rotation']).rotation_matrix
    cam2ego[:3, 3] = np.asarray(calib_dict['translation']).T

    ego2global = np.eye(4)
    ego2global[:3, :3] = Quaternion(pose_dict['rotation']).rotation_matrix
    ego2global[:3, 3] = np.asarray(pose_dict['translation']).T

    img2global = ego2global @ cam2ego @ img2cam # transformation from image frame to camera frame to ego frame to global frame
    return img2global

# Used in dataset/dataset.py
def get_lidar2global(calib_dict, pose_dict):

    lidar2ego = np.eye(4)
    lidar2ego[:3, :3] = Quaternion(calib_dict['rotation']).rotation_matrix
    lidar2ego[:3, 3] = np.asarray(calib_dict['translation']).T

    ego2global = np.eye(4)
    ego2global[:3, :3] = Quaternion(pose_dict['rotation']).rotation_matrix
    ego2global[:3, 3] = np.asarray(pose_dict['translation']).T

    lidar2global = ego2global @ lidar2ego # transformation from lidar frame to ego frame to global frame
    return lidar2global


# Please update this function in the dataset
def get_lidar2cam(calib_dict, pose_dict):
    # Copilot code
    # lidar - global - ego - img - cam
    # cam2img = np.eye(4)
    # cam2img[:3, :3] = np.asarray(calib_dict['camera_intrinsic'])
    # img2cam = np.linalg.inv(cam2img)

    # cam2ego = np.eye(4)
    # cam2ego[:3, :3] = Quaternion(calib_dict['rotation']).rotation_matrix
    # cam2ego[:3, 3] = np.asarray(calib_dict['translation']).T

    # ego2global = np.eye(4)
    # ego2global[:3, :3] = Quaternion(pose_dict['rotation']).rotation_matrix
    # ego2global[:3, 3] = np.asarray(pose_dict['translation']).T

    # img2global = ego2global @ cam2ego @ img2cam # transformation from image frame to camera frame to ego frame to global frame

    # lidar2ego = np.eye(4)
    # lidar2ego[:3, :3] = Quaternion(calib_dict['rotation']).rotation_matrix
    # lidar2ego[:3, 3] = np.asarray(calib_dict['translation']).T

    # lidar2global = ego2global @ lidar2ego # transformation from lidar frame to ego frame to global frame

    # lidar2cam = img2cam @ np.linalg.inv(img2global) @ lidar2global # transformation from lidar frame to global frame to image frame

    # lidar - ego - cam
    lidar2ego = np.eye(4)
    lidar2ego[:3, :3] = Quaternion(calib_dict['rotation']).rotation_matrix
    lidar2ego[:3, 3] = np.asarray(calib_dict['translation']).T

    cam2ego = np.eye(4)
    cam2ego[:3, :3] = Quaternion(calib_dict['rotation']).rotation_matrix
    cam2ego[:3, 3] = np.asarray(calib_dict['translation']).T
    ego2cam = np.linalg.inv(cam2ego)

    lidar2cam = ego2cam @ lidar2ego

    return lidar2cam # should be 4 x 4 matrix


def lidar_to_cam_points(points, lidar2cam):
    """
    Transform lidar points to camera points
    :param points: (N, 3) lidar points, numpy array
    :param lidar2cam: (4, 4) lidar to camera transformation matrix
    """
    points = np.hstack([points, np.ones((points.shape[0], 1))])
    points = np.dot(lidar2cam, points.T).T
    points[:, :3] = points[:, :3] / points[:, 3].reshape(-1, 1)
    return points[:, :3] # should be (N, 3) numpy array


# might not need this function
# intrinsic already updated in ResizeCropFlipImage
def adjust_intrinsic(fW, fH, W, H, intrinsic):
    """
    Adjust intrinsic matrix for the new image size
    :param fW: float, scaling factor for width
    :param fH: float, scaling factor for height
    :param W: int, original width
    :param H: int, original height
    :param intrinsic: (4, 4) homogeneous intrinsic matrix
    """
    resize = (fW / W, fH / H)
    post_rot2 = torch.eye(2)
    rot_resize = torch.tensor([[resize[0], 0], [0, resize[1]]])
    post_rot2 = rot_resize @ post_rot2
    post_rot = torch.eye(3)
    post_rot[:2, :2] = post_rot2
    cam_p = post_rot @ intrinsic[:3, :3]

    return cam_p # should be (3, 3) tensor, here is in W, H


def project_pc_to_image(point_cloud, cam_p):
    """Projects a 3D point cloud to 2D points

    Args:
        point_cloud: (3, N) point cloud in camera coordinate frame
        cam_p: camera projection matrix

    Returns:
        pts_2d: (2, N) projected coordinates [u, v] of the 3D points
    """
    points = cam_p @ point_cloud # should be (3, N) numpy array
    points[:2] /= points[2:3]

    return points[0:2]


def project_depths(point_cloud, cam_p, image_shape, max_depth=100.0):
    """Projects a point cloud into image space and saves depths per pixel.

    Args:
        point_cloud: (5, N) Point cloud in each cam's coordinate frame
        cam_p: camera projection matrix
        image_shape: image shape [h, w] TODO: check if this is (h, w) or (w, h)!!
        max_depth: optional, max depth for inversion

    Returns:
        projected_depths: projected depth map
    """

    # Only keep points in front of the camera
    point_cloud_pos = point_cloud[:3, :]
    all_points = point_cloud_pos.T

    # Save the depth corresponding to each point
    points_in_img = project_pc_to_image(all_points.T, cam_p) # since cam_p is first w then h, points_in_img is also [w, h]
    points_in_img_int = np.int32(np.round(points_in_img))

    # Remove points outside image
    # image_shape = [h, w], so points_in_img_int is in the form [w, h]!
    valid_indices = \
        (points_in_img_int[0] >= 0) & (points_in_img_int[0] < image_shape[1]) & \
        (points_in_img_int[1] >= 0) & (points_in_img_int[1] < image_shape[0])

    all_points = all_points[valid_indices]
    points_in_img_int = points_in_img_int[:, valid_indices]

    # Invert depths
    all_points[:, 2] = max_depth - all_points[:, 2]

    # Only save valid pixels, keep closer points when overlapping
    # Here we swap the x and y coordinates to match the image shape in the form [h, w]
    projected_depths = np.zeros(image_shape)

    # Loop version
    # valid_indices = [points_in_img_int[1], points_in_img_int[0]]
    # projected_depths[tuple(valid_indices)] = [
    #     max(projected_depths[
    #         points_in_img_int[1, idx], points_in_img_int[0, idx]],
    #         all_points[idx, 2])
    #     for idx in range(points_in_img_int.shape[1])]
    
    # Vectorized version of the above loop
    valid_indices = (points_in_img_int[1], points_in_img_int[0])
    np.maximum.at(projected_depths, valid_indices, all_points[:, 2])

    projected_depths[tuple(valid_indices)] = \
        max_depth - projected_depths[tuple(valid_indices)]

    projected_depths[projected_depths < 0] = 0

    return projected_depths.astype(np.float32) # should be (h, w) numpy array


def project_intensity(point_cloud, cam_p, image_shape, max_intensity=255.0):
    """Projects a point cloud into image space and saves intensity per pixel.

    Args:
        point_cloud: (5, N) Point cloud in each cam's coordinate frame
        cam_p: camera projection matrix
        image_shape: image shape [h, w] TODO: check if this is (h, w) or (w, h)!!
        max_intensity: optional, max intensity for inversion

    Returns:
        projected_intensity: projected intensity map
    """

    # Only keep points in front of the camera
    point_cloud_pos = point_cloud[:3, :]
    all_points = point_cloud_pos.T
    all_points_property = point_cloud.T

    # Save the depth corresponding to each point
    points_in_img = project_pc_to_image(all_points.T, cam_p) # since cam_p is first w then h, points_in_img is also [w, h]
    points_in_img_int = np.int32(np.round(points_in_img))

    # Remove points outside image
    # image_shape = [h, w], so points_in_img_int is in the form [w, h]!
    valid_indices = \
        (points_in_img_int[0] >= 0) & (points_in_img_int[0] < image_shape[1]) & \
        (points_in_img_int[1] >= 0) & (points_in_img_int[1] < image_shape[0])

    all_points_property = all_points_property[valid_indices]
    points_in_img_int = points_in_img_int[:, valid_indices]

    # Only save valid pixels, keep closer points when overlapping
    # Here we swap the x and y coordinates to match the image shape in the form [h, w]
    projected_intensity = np.zeros(image_shape)
    valid_indices = (points_in_img_int[1], points_in_img_int[0])

    # projected_intensity[tuple(valid_indices)] = [
    #     max(projected_intensity[
    #         points_in_img_int[1, idx], points_in_img_int[0, idx]],
    #         all_points_property[idx, 3])
    #     for idx in range(points_in_img_int.shape[1])]

    # Vectorized version of the above loop
    np.maximum.at(projected_intensity, valid_indices, all_points_property[:, 3])

    projected_intensity[projected_intensity < 0] = 0 # value should be between 0 and 255
    projected_intensity[projected_intensity > max_intensity] = max_intensity

    return projected_intensity.astype(np.float32) # should be (h, w) numpy array


def project_height(point_cloud_raw, point_cloud, cam_p, image_shape, max_height=3.0, min_height=-5.0):
    """Projects a point cloud into image space and saves intensity per pixel.

    Args:
        point_cloud_raw: (5, N) Point cloud in each lidar coordinate frame
        point_cloud: (5, N) Point cloud in each cam's coordinate frame
        cam_p: camera projection matrix
        image_shape: image shape [h, w] TODO: check if this is (h, w) or (w, h)!!

    Returns:
        projected_height: projected height map
    """

    # Only keep points in front of the camera
    point_cloud_pos = point_cloud[:3, :]
    all_points = point_cloud_pos.T
    all_points_property_lidar = point_cloud_raw.T # these points should be in the lidar frame, which contains the height information which is z

    # Save the depth corresponding to each point
    points_in_img = project_pc_to_image(all_points.T, cam_p) # since cam_p is first w then h, points_in_img is also [w, h]
    points_in_img_int = np.int32(np.round(points_in_img))

    # Remove points outside image
    # image_shape = [h, w], so points_in_img_int is in the form [w, h]!
    valid_indices = \
        (points_in_img_int[0] >= 0) & (points_in_img_int[0] < image_shape[1]) & \
        (points_in_img_int[1] >= 0) & (points_in_img_int[1] < image_shape[0])

    all_points_property_lidar = all_points_property_lidar[valid_indices]
    points_in_img_int = points_in_img_int[:, valid_indices]

    # Only save valid pixels, keep closer points when overlapping
    # Here we swap the x and y coordinates to match the image shape in the form [h, w]
    projected_height = np.zeros(image_shape)
    valid_indices = (points_in_img_int[1], points_in_img_int[0])

    # projected_height[tuple(valid_indices)] = [
    #     max(projected_height[
    #         points_in_img_int[1, idx], points_in_img_int[0, idx]],
    #         all_points_property_lidar[idx, 2])
    #     for idx in range(points_in_img_int.shape[1])]

    # Vectorized version of the above loop
    np.maximum.at(projected_height, valid_indices, all_points_property_lidar[:, 2])

    projected_height[projected_height < min_height] = min_height # value should be between 0 and 255
    projected_height[projected_height > max_height] = max_height

    return projected_height.astype(np.float32) # should be (h, w) numpy array


def project_depths_intensity_height(point_cloud_raw, point_cloud, cam_p, image_shape, max_depth=100.0, max_intensity=255.0, max_height=3.0, min_height=-5.0):
    """Projects a point cloud into image space and saves depths, intensity and height per pixel.

    Args:
        point_cloud_raw: (5, N) Point cloud in each lidar coordinate frame
        point_cloud: (5, N) Point cloud in each cam's coordinate frame
        cam_p: camera projection matrix
        image_shape: image shape [h, w] TODO: check if this is (h, w) or (w, h)!!
        max_depth: optional, max depth for inversion
        max_intensity: optional, max intensity for filtering
        max_height: optional, max height for filtering

    Returns:
        projected_depths: projected depth map
        projected_intensity: projected intensity map
        projected_height: projected height map
    """
    
    # Only keep points in front of the camera
    point_cloud_pos = point_cloud[:3, :]
    all_points = point_cloud_pos.T
    all_points_property = point_cloud.T
    all_points_property_lidar = point_cloud_raw.T # these points should be in the lidar frame, which contains the height information which is z
    
    # Save the depth corresponding to each point
    points_in_img = project_pc_to_image(all_points.T, cam_p) # since cam_p is first w then h, points_in_img is also [w, h]
    points_in_img_int = np.int32(np.round(points_in_img))
    
    # Remove points outside image
    # image_shape = [h, w], so points_in_img_int is in the form [w, h]!
    valid_indices = \
        (points_in_img_int[0] >= 0) & (points_in_img_int[0] < image_shape[1]) & \
        (points_in_img_int[1] >= 0) & (points_in_img_int[1] < image_shape[0])
    
    all_points = all_points[valid_indices]
    all_points_property = all_points_property[valid_indices]
    all_points_property_lidar = all_points_property_lidar[valid_indices]
    points_in_img_int = points_in_img_int[:, valid_indices]
    
    # Invert depths
    all_points[:, 2] = max_depth - all_points[:, 2]

    # Only save valid pixels, keep closer points when overlapping
    # Here we swap the x and y coordinates to match the image shape in the form [h, w]
    projected_depths = np.zeros(image_shape)
    projected_intensity = np.zeros(image_shape)
    projected_height = np.zeros(image_shape)
    valid_indices = [points_in_img_int[1], points_in_img_int[0]]
    
    # projected_depths[tuple(valid_indices)] = [
    #     max(projected_depths[
    #         points_in_img_int[1, idx], points_in_img_int[0, idx]],
    #         all_points[idx, 2])
    #     for idx in range(points_in_img_int.shape[1])]
    
    # projected_intensity[tuple(valid_indices)] = [
    #     max(projected_intensity[
    #         points_in_img_int[1, idx], points_in_img_int[0, idx]],
    #         all_points_property[idx, 3])
    #     for idx in range(points_in_img_int.shape[1])]
    
    # projected_height[tuple(valid_indices)] = [
    #     max(projected_height[
    #         points_in_img_int[1, idx], points_in_img_int[0, idx]],
    #         all_points_property_lidar[idx, 2])
    #     for idx in range(points_in_img_int.shape[1])]

    for idx in range(points_in_img_int.shape[1]):
        # Extract pixel indices
        y_idx = points_in_img_int[1, idx]
        x_idx = points_in_img_int[0, idx]
            
        # Update projected depths: keep the maximum (closer point)
        projected_depths[y_idx, x_idx] = max(
            projected_depths[y_idx, x_idx], 
            all_points[idx, 2]  # z-coordinate (depth)
        )
            
        # Update projected intensity: keep the maximum intensity
        projected_intensity[y_idx, x_idx] = max(
            projected_intensity[y_idx, x_idx], 
            all_points_property[idx, 3]  # intensity property
        )
            
            
        # Update projected height: keep the maximum height (z from LiDAR)
        projected_height[y_idx, x_idx] = max(
            projected_height[y_idx, x_idx], 
            all_points_property_lidar[idx, 2]  # height (z-coordinate in LiDAR)
        )


    # Invert depths and filter values   
    projected_depths[tuple(valid_indices)] = \
    max_depth - projected_depths[tuple(valid_indices)]
    projected_depths[projected_depths < 0] = 0

    # filter intensity
    projected_intensity[projected_intensity < 0] = 0
    projected_intensity[projected_intensity > max_intensity] = max_intensity

    # filter height
    projected_height[projected_height < min_height] = min_height
    projected_height[projected_height > max_height] = max_height

    return projected_depths.astype(np.float32), projected_intensity.astype(np.float32), projected_height.astype(np.float32) # should be (h, w) numpy array


def custom_collate_fn_temporal(instances):
    return_dict = {}
    for k, v in instances[0].items():
        if isinstance(v, np.ndarray):
            return_dict[k] = torch.stack([
                torch.from_numpy(instance[k]) for instance in instances])
        elif isinstance(v, torch.Tensor):
            return_dict[k] = torch.stack([instance[k] for instance in instances])
        elif isinstance(v, (dict, str)):
            return_dict[k] = [instance[k] for instance in instances]
        elif isinstance(v, list):
            return_dict[k] = v
        elif v is None:
            return_dict[k] = [None] * len(instances)
        else:
            raise NotImplementedError
    return return_dict
