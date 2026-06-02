try:
    from vis import save_occ, save_gaussian
except:
    print('Load Occupancy Visualization Tools Failed.')
import time, argparse, os.path as osp, os
import torch, numpy as np
import torch.distributed as dist

from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.logging import MMLogger
from mmseg.models import build_segmentor
from PIL import Image

from nuscenes.nuscenes import NuScenes
import matplotlib.pyplot as plt
from nuscenes.utils.data_classes import LidarPointCloud
import random
import mmengine
import pickle

import warnings
warnings.filterwarnings("ignore")


def pass_print(*args, **kwargs):
    pass

def main(local_rank, args):
    # global settings
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # load config
    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir

    # init DDP
    if args.gpus > 1:
        distributed = True
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", "20507")
        hosts = int(os.environ.get("WORLD_SIZE", 1))  # number of nodes
        rank = int(os.environ.get("RANK", 0))  # node id
        gpus = torch.cuda.device_count()  # gpus per node
        print(f"tcp://{ip}:{port}")
        dist.init_process_group(
            backend="nccl", init_method=f"tcp://{ip}:{port}", 
            world_size=hosts * gpus, rank=rank * gpus + local_rank)
        world_size = dist.get_world_size()
        cfg.gpu_ids = range(world_size)
        torch.cuda.set_device(local_rank)

        if local_rank != 0:
            import builtins
            builtins.print = pass_print
    else:
        distributed = False
        world_size = 1
    
    writer = None
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'{timestamp}.log')
    logger = MMLogger('gf3d', log_file=log_file)
    MMLogger._instance_dict['gf3d'] = logger
    logger.info(f'Config:\n{cfg.pretty_text}')

    # build model
    import model
    from dataset import get_dataloader

    my_model = build_segmentor(cfg.model)
    my_model.init_weights()
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')
    if distributed:
        if cfg.get('syncBN', True):
            my_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(my_model)
            logger.info('converted sync bn.')

        find_unused_parameters = cfg.get('find_unused_parameters', False)
        ddp_model_module = torch.nn.parallel.DistributedDataParallel
        my_model = ddp_model_module(
            my_model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
        raw_model = my_model.module
    else:
        my_model = my_model.cuda()
        raw_model = my_model
    logger.info('done ddp model')

    cfg.val_dataset_config.update({
        "vis_indices": args.vis_index,
        "num_samples": args.num_samples})

    if args.train_pkl == True:
        # Directly use train pkl for video visualization
        cfg.val_dataset_config = cfg.train_dataset_config
        cfg.val_dataset_config['phase'] = 'val'
        logger.info('Using train pkl for video visualization')
        logger.info('val_dataset_config: ' + str(cfg.val_dataset_config))


    train_dataset_loader, val_dataset_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_loader,
        cfg.val_loader,
        dist=distributed,
        val_only=True)
    
    # resume and load
    cfg.resume_from = ''
    if osp.exists(osp.join(args.work_dir, 'latest.pth')):
        cfg.resume_from = osp.join(args.work_dir, 'latest.pth')
    if args.resume_from:
        cfg.resume_from = args.resume_from
    
    logger.info('resume from: ' + cfg.resume_from)
    logger.info('work dir: ' + args.work_dir)

    if cfg.resume_from and osp.exists(cfg.resume_from):
        map_location = 'cpu'
        ckpt = torch.load(cfg.resume_from, map_location=map_location)
        raw_model.load_state_dict(ckpt['state_dict'], strict=True)
        # raw_model.load_state_dict(ckpt['state_dict'], strict=False)
        print(f'successfully resumed.')

    elif cfg.load_from:
        ckpt = torch.load(cfg.load_from, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        print(raw_model.load_state_dict(state_dict, strict=False))
        
    print_freq = cfg.print_freq

    my_model.eval()
    os.environ['eval'] = 'true'
    if args.vis_occ or args.vis_gaussian:
        os.makedirs(os.path.join(args.work_dir, 'vis'), exist_ok=True)

    with torch.no_grad():
        for i_iter_val, data in enumerate(val_dataset_loader):

            for k in list(data.keys()):
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].cuda()
                # if isinstance(data[k], list):
                #     data[k] = [d.cuda() for d in data[k]]
                if isinstance(data[k], dict):
                    for kk in data[k].keys():
                        if isinstance(data[k][kk], torch.Tensor):
                            data[k][kk] = data[k][kk].cuda()
                if isinstance(data[k], list):
                    for kk in range(len(data[k])):
                        # list element is tensor
                        if isinstance(data[k][kk], torch.Tensor):
                            data[k][kk] = data[k][kk].cuda()
                        # list element is dict
                        if isinstance(data[k][kk], dict):
                            for kkk in data[k][kk].keys():
                                if isinstance(data[k][kk][kkk], torch.Tensor):
                                    data[k][kk][kkk] = data[k][kk][kkk].cuda()
            
            input_imgs = data.pop('img')
            ## Visualization
            # Image
            print(data.keys())
            if args.vis_img:
                imgs_path_lst = data['img_filename']
                # destination_folder = os.path.join(args.work_dir, 'vis', f'val_{i_iter_val}_imgs')
                destination_folder = os.path.join(args.work_dir, 'vis', f'val_{i_iter_val}', 'imgs')
                os.makedirs(destination_folder, exist_ok=True)
                for img_path in imgs_path_lst:
                    try:
                        with Image.open(img_path) as img:
                            img_name = os.path.basename(img_path)
                            save_path = os.path.join(destination_folder, img_name)             
                            img.save(save_path)                 
                            print(f"Saved: {save_path}")
                    except Exception as e:
                        print(f"Failed to process {img_path}: {e}")
            
            # Visualize lidar point cloud
            if args.vis_lidar:
                # lidar_map_vis_path = os.path.join(args.work_dir, 'vis', f'val_{i_iter_val}_lidar_map.png')
                lidar_map_vis_path = os.path.join(args.work_dir, 'vis', f'val_{i_iter_val}', 'lidar_map.png')
                lidar_path = data['pts_filename'][0]
                mmengine.check_file_exist(lidar_path)
                # plot
                plt.switch_backend('Agg')
                fig, ax = plt.subplots(figsize=(9, 9))
                LidarPointCloud.from_file(lidar_path).render_height(ax)
                ax.axis('off')
                fig.savefig(lidar_map_vis_path, bbox_inches='tight', pad_inches=0, dpi=300)
                plt.close(fig)
                print(f"LiDAR visualization saved to {lidar_map_vis_path}")

            ##
            if 'points' in data:
                input_points = data.pop('points')
            else:
                input_points = None
            if 'lidar_feature_maps' in data:
                input_lidar_features = data.pop('lidar_feature_maps') # dict
            else:
                input_lidar_features = None
            if 'dpt' in data:
                input_dpt = data.pop('dpt')
            else:
                input_dpt = None
            if 'anchor_points' in data:
                input_anchor_points = data.pop('anchor_points')
            else:
                input_anchor_points = None

            result_dict = my_model(imgs=input_imgs, points=input_points, lidar_feature_maps=input_lidar_features, dpt=input_dpt, anchor_points=input_anchor_points, metas=data)

            for idx, pred in enumerate(result_dict['pred_occ'][-1]):
                pred_occ = pred.argmax(0)
                gt_occ = result_dict['sampled_label'][idx]
                os.makedirs(os.path.join(args.work_dir, 'vis', f'val_{i_iter_val}'), exist_ok=True)
                if args.vis_occ:
                    save_occ(
                        os.path.join(args.work_dir, 'vis', f'val_{i_iter_val}'),
                        pred_occ.reshape(1, 200, 200, 16),
                        f'val_{i_iter_val}_pred',
                        True, 0)
                if args.vis_gaussian:
                    save_gaussian(
                        os.path.join(args.work_dir, 'vis', f'val_{i_iter_val}'),
                        result_dict['gaussian'],
                        f'val_{i_iter_val}_gaussian')

                if args.vis_gaussian_init:
                    save_gaussian(
                        os.path.join(args.work_dir, 'vis', f'val_{i_iter_val}'),
                        result_dict['gaussian_init'],
                        f'val_{i_iter_val}_gaussian_init')
            
            if i_iter_val % print_freq == 0 and local_rank == 0:
                logger.info('[EVAL] Iter %5d'%(i_iter_val))
            
            torch.cuda.empty_cache()
    
    if writer is not None:
        writer.close()
        

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/tpv_lidarseg.py')
    parser.add_argument('--work-dir', type=str, default='./out/tpv_lidarseg')
    parser.add_argument('--resume-from', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-pkl', action='store_true', default=False)
    parser.add_argument('--vis-occ', action='store_true', default=False)
    parser.add_argument('--vis-gaussian', action='store_true', default=False)
    parser.add_argument('--vis-img', action='store_true', default=False)
    parser.add_argument('--vis-lidar', action='store_true', default=False)
    parser.add_argument('--vis-gaussian-init', action='store_true', default=False)
    parser.add_argument('--vis-index', type=int, nargs='+', default=[])
    parser.add_argument('--num-samples', type=int, default=1)
    args = parser.parse_args()
    
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    print(args)

    if ngpus > 1:
        torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
    else:
        main(0, args)