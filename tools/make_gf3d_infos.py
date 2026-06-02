import argparse
import os
import pickle

import numpy as np

# ---------------------------------------------------------------------------
# Path helper
# ---------------------------------------------------------------------------
DATA_ROOT_REL = "data/nuscenes"  # where data/nuscenes/{samples,sweeps} live


def relativize(path, data_root=DATA_ROOT_REL):
    """Turn an absolute nuScenes file path into one relative to the repo root.

    Works regardless of the original absolute prefix by anchoring on the
    ``samples/`` or ``sweeps/`` segment that every nuScenes file path contains.
    Already-relative paths are returned unchanged.
    """
    path = str(path)
    for marker in ("samples/", "sweeps/"):
        idx = path.find(marker)
        if idx != -1:
            return os.path.join(data_root, path[idx:])
    return path


def relativize_infos(data):
    """In-place relativize lidar_path + sweeps[*].data_path for every keyframe."""
    n_lidar = n_sweep = 0
    for scene, i in data["metadata"]:
        info = data["infos"][scene][i]
        if "lidar_path" in info:
            info["lidar_path"] = relativize(info["lidar_path"])
            n_lidar += 1
        for sweep in info.get("sweeps", []):
            sweep["data_path"] = relativize(sweep["data_path"])
            n_sweep += 1
    print(f"  relativized {n_lidar} lidar_path and {n_sweep} sweep data_path entries")
    return data


# ---------------------------------------------------------------------------
# Mode 1: derive from the existing *_new.pkl (no devkit needed)
# ---------------------------------------------------------------------------
def from_new(anno_root):
    for split in ("train", "val"):
        src = os.path.join(anno_root, f"nuscenes_infos_gaussian_{split}_new.pkl")
        dst = os.path.join(anno_root, f"nuscenes_infos_gf3d_{split}.pkl")
        print(f"[{split}] reading {src}")
        with open(src, "rb") as f:
            data = pickle.load(f)
        relativize_infos(data)
        with open(dst, "wb") as f:
            pickle.dump(data, f)
        print(f"[{split}] wrote {dst}")


# ---------------------------------------------------------------------------
# Mode 2: full regeneration from *_sweeps_occ.pkl via the nuScenes devkit
# ---------------------------------------------------------------------------
def obtain_sensor2top(nusc, sensor_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat,
                      sensor_type="lidar"):
    """Obtain the RT matrix from a sweep sensor to the top LiDAR (relative path)."""
    from pyquaternion import Quaternion

    sd_rec = nusc.get("sample_data", sensor_token)
    cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
    data_path = relativize(nusc.get_sample_data_path(sd_rec["token"]))
    sweep = {
        "data_path": data_path,
        "type": sensor_type,
        "sample_data_token": sd_rec["token"],
        "sensor2ego_translation": cs_record["translation"],
        "sensor2ego_rotation": cs_record["rotation"],
        "ego2global_translation": pose_record["translation"],
        "ego2global_rotation": pose_record["rotation"],
        "timestamp": sd_rec["timestamp"],
    }
    l2e_r_s = sweep["sensor2ego_rotation"]
    l2e_t_s = sweep["sensor2ego_translation"]
    e2g_r_s = sweep["ego2global_rotation"]
    e2g_t_s = sweep["ego2global_translation"]
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep["sensor2lidar_rotation"] = R.T
    sweep["sensor2lidar_translation"] = T
    return sweep


def regenerate(root_path, anno_root, max_sweeps=10, version="v1.0-trainval"):
    import mmengine
    from nuscenes.nuscenes import NuScenes
    from pyquaternion import Quaternion

    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)

    for split in ("train", "val"):
        src = os.path.join(anno_root, f"nuscenes_infos_{split}_sweeps_occ.pkl")
        dst = os.path.join(anno_root, f"nuscenes_infos_gf3d_{split}.pkl")
        print(f"[{split}] reading {src}")
        with open(src, "rb") as f:
            data = pickle.load(f)

        for scene, i in mmengine.track_iter_progress(data["metadata"]):
            info = data["infos"][scene][i]
            lidar_token = info["data"]["LIDAR_TOP"]["token"]
            lidar_path = nusc.get_sample_data(lidar_token)[0]
            mmengine.check_file_exist(lidar_path)
            info["lidar_path"] = relativize(lidar_path)

            sd_rec = nusc.get("sample_data", lidar_token)
            cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
            pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
            l2e_t = cs_record["translation"]
            e2g_t = pose_record["translation"]
            l2e_r_mat = Quaternion(cs_record["rotation"]).rotation_matrix
            e2g_r_mat = Quaternion(pose_record["rotation"]).rotation_matrix

            sweeps = []
            while len(sweeps) < max_sweeps and sd_rec["prev"] != "":
                sweeps.append(obtain_sensor2top(
                    nusc, sd_rec["prev"], l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, "lidar"))
                sd_rec = nusc.get("sample_data", sd_rec["prev"])
            info["sweeps"] = sweeps

        with open(dst, "wb") as f:
            pickle.dump(data, f)
        print(f"[{split}] wrote {dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--from-new", action="store_true",
                      help="derive from existing *_new.pkl (fast, no devkit)")
    mode.add_argument("--regenerate", action="store_true",
                      help="rebuild from *_sweeps_occ.pkl via the nuScenes devkit")
    parser.add_argument("--root-path", default="data/nuscenes",
                        help="nuScenes dataroot (use a RELATIVE path, e.g. data/nuscenes)")
    parser.add_argument("--out-path", default="data/nuscenes_cam",
                        help="directory holding the *.pkl info files")
    parser.add_argument("--max-sweeps", type=int, default=10)
    args = parser.parse_args()

    if args.from_new:
        from_new(args.out_path)
    else:
        regenerate(args.root_path, args.out_path, max_sweeps=args.max_sweeps)
    print("Done!")
