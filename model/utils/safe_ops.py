import torch


SIGMOID_MAX = 9.21024
LOGIT_MAX = 0.9999
pc_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
pc_range_2 = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
pc_range_3 = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
pc_range_4 = [-20.0, -10.0, -2.0, 0.0, 10.0, 6.0]
pc_range_5 = [0.0, -10.0, -2.0, 20.0, 10.0, 6.0]

def safe_sigmoid(tensor):
    tensor = torch.clamp(tensor, -9.21, 9.21)
    return torch.sigmoid(tensor)

def safe_inverse_sigmoid(tensor):
    tensor = torch.clamp(tensor, 1 - LOGIT_MAX, LOGIT_MAX)
    return torch.log(tensor / (1 - tensor))

def point_cloud_map(tensor, occ_annotation="surroundocc"):
    # tensor: (N, C)
    if occ_annotation == "surroundocc":
        pc_range_tensor = torch.tensor(pc_range, device=tensor.device, dtype=tensor.dtype)
    elif occ_annotation == "openocc":
        pc_range_tensor = torch.tensor(pc_range_2, device=tensor.device, dtype=tensor.dtype)
    elif occ_annotation == "occ3d":
        pc_range_tensor = torch.tensor(pc_range_3, device=tensor.device, dtype=tensor.dtype)
    elif occ_annotation == "wildocc":
        pc_range_tensor = torch.tensor(pc_range_4, device=tensor.device, dtype=tensor.dtype)
    elif occ_annotation == "wildocc_wrong":
        pc_range_tensor = torch.tensor(pc_range_5, device=tensor.device, dtype=tensor.dtype)
    raw_xyz = tensor[:, :3]
    map_xxx = (raw_xyz[:, 0] - pc_range_tensor[0]) / (pc_range_tensor[3] - pc_range_tensor[0])
    map_yyy = (raw_xyz[:, 1] - pc_range_tensor[1]) / (pc_range_tensor[4] - pc_range_tensor[1])
    map_zzz = (raw_xyz[:, 2] - pc_range_tensor[2]) / (pc_range_tensor[5] - pc_range_tensor[2])
    return torch.stack([map_xxx, map_yyy, map_zzz], dim=-1)
