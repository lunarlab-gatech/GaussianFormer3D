import os
os.environ['QT_PLUGIN_PATH'] = '/usr/lib/x86_64-linux-gnu/qt5'
offscreen = False
if os.environ.get('DISP', 'f') == 'f':
    try:
        from pyvirtualdisplay import Display
        display = Display(visible=False, size=(2560, 1440))
        display.start()
        offscreen = True
    except:
        print("Failed to start virtual display.")

try:
    from mayavi import mlab
    import mayavi
    mlab.options.offscreen = offscreen
    print("Set mlab.options.offscreen={}".format(mlab.options.offscreen))
except:
    print("No Mayavi installation found.")

import torch, numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.style as mplstyle
mplstyle.use('fast')
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
from pyquaternion import Quaternion
import os
from PIL import Image


def get_grid_coords(dims, resolution):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """

    g_xx = np.arange(0, dims[0]) # [0, 1, ..., 256]
    # g_xx = g_xx[::-1]
    g_yy = np.arange(0, dims[1]) # [0, 1, ..., 256]
    # g_yy = g_yy[::-1]
    g_zz = np.arange(0, dims[2]) # [0, 1, ..., 32]

    # Obtaining the grid with coords...
    xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz)
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float32)
    resolution = np.array(resolution, dtype=np.float32).reshape([1, 3])

    coords_grid = (coords_grid * resolution) + resolution / 2

    return coords_grid

def save_occ(
        save_dir, 
        gaussian, 
        name,
        sem=False,
        cap=2,
        dataset='nusc'
    ):
    if dataset == 'nusc':
        voxel_size = [0.5] * 3
        vox_origin = [-50.0, -50.0, -5.0]
        vmin, vmax = 0, 16
    elif dataset == 'rellis3d':
        voxel_size = [0.2] * 3
        vox_origin = [-20.0, -10.0, -2.0]
        vmin, vmax = 0, 7
        # vmin, vmax = 0, 8
    elif dataset == 'kitti':
        voxel_size = [0.2] * 3
        vox_origin = [0.0, -25.6, -2.0]
        vmin, vmax = 1, 19
    elif dataset == 'kitti360':
        voxel_size = [0.2] * 3
        vox_origin = [0.0, -25.6, -2.0]
        vmin, vmax = 1, 18

    voxels = gaussian[0].cpu().to(torch.int) # gaussian shape: (1, 200, 200, 16), voxel shape: (200, 200, 16)
    voxels[0, 0, 0] = 1
    voxels[-1, -1, -1] = 1
    if not sem:
        voxels[..., (-cap):] = 0
        for z in range(voxels.shape[-1] - cap):
            mask = (voxels > 0)[..., z]
            voxels[..., z][mask] = z + 1 
    
    # Compute the voxels coordinates
    grid_coords = get_grid_coords(
        voxels.shape, voxel_size
    ) + np.array(vox_origin, dtype=np.float32).reshape([1, 3])

    grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T
    # Get the voxels inside FOV
    fov_grid_coords = grid_coords

    # Remove empty and unknown voxels
    if not sem:
        fov_voxels = fov_grid_coords[
            (fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 100)
        ]
    else:
        if dataset == 'nusc':
            fov_voxels = fov_grid_coords[
                (fov_grid_coords[:, 3] >= 0) & (fov_grid_coords[:, 3] < 17) # nuscenes has 17 classes
            ]
        elif dataset == 'rellis3d':
            fov_voxels = fov_grid_coords[
                (fov_grid_coords[:, 3] >= 0) & (fov_grid_coords[:, 3] < 8) # rellis3d has 8 classes
            ]
        elif dataset == 'kitti360':
            fov_voxels = fov_grid_coords[
                (fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 19)
            ]
        else:
            fov_voxels = fov_grid_coords[
                (fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 20)
            ]
    print(len(fov_voxels))
    
    figure = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))
    print("mlab figure created")
    # Draw occupied inside FOV voxels
    voxel_size = sum(voxel_size) / 3
    if not sem:
        plt_plot_fov = mlab.points3d(
            fov_voxels[:, 0],
            -fov_voxels[:, 1],
            fov_voxels[:, 2],
            fov_voxels[:, 3],
            colormap="jet",
            scale_factor=1.0 * voxel_size,
            mode="cube",
            opacity=1.0,
        )
    else:
        plt_plot_fov = mlab.points3d(
            fov_voxels[:, 0],
            -fov_voxels[:, 1],
            fov_voxels[:, 2],
            fov_voxels[:, 3],
            scale_factor=1.0 * voxel_size,
            mode="cube",
            opacity=1.0,
            vmin=vmin,
            vmax=vmax, # 16
        )
    print("enter plt_plot_fov.glyph.scale_mode")
    plt_plot_fov.glyph.scale_mode = "scale_by_vector"
    if sem:
        if dataset == 'nusc':
            colors = np.array(
                [
                    [  0,   0,   0, 255],       # others
                    [255, 120,  50, 255],       # barrier              orange
                    [255, 192, 203, 255],       # bicycle              pink
                    [255, 255,   0, 255],       # bus                  yellow
                    [  0, 150, 245, 255],       # car                  blue
                    [  0, 255, 255, 255],       # construction_vehicle cyan
                    [255, 127,   0, 255],       # motorcycle           dark orange
                    [255,   0,   0, 255],       # pedestrian           red
                    [255, 240, 150, 255],       # traffic_cone         light yellow
                    [135,  60,   0, 255],       # trailer              brown
                    [160,  32, 240, 255],       # truck                purple                
                    [255,   0, 255, 255],       # driveable_surface    dark pink
                    # [175,   0,  75, 255],       # other_flat           dark red
                    [139, 137, 137, 255],
                    [ 75,   0,  75, 255],       # sidewalk             dard purple
                    [150, 240,  80, 255],       # terrain              light green          
                    [230, 230, 250, 255],       # manmade              white
                    [  0, 175,   0, 255],       # vegetation           green
                    # [  0, 255, 127, 255],       # ego car              dark cyan
                    # [255,  99,  71, 255],       # ego car
                    # [  0, 191, 255, 255]        # ego car
                ]
            ).astype(np.uint8)
        elif dataset == 'rellis3d':
            print("enter define colors for rellis3d")
            colors = np.array(
                [
                    [  0,   0,   0, 255],       # others
                    [0, 128, 0, 255],       # grass
                    [0, 255, 0, 255],       # tree
                    [255, 153, 204, 255],       # bush
                    [133, 255, 240, 255],       # puddle
                    [99, 66, 33, 255],       # mud
                    [41, 128, 255, 255],       # barrier
                    [110, 20, 138, 255],       # rubble
                    # [255, 0, 0, 255],       # red empty
                ]
            ).astype(np.uint8)
        elif dataset == 'kitti360':
            colors = (get_kitti360_colormap()[1:, :] * 255).astype(np.uint8)
        else:
            colors = (get_kitti_colormap()[1:, :] * 255).astype(np.uint8)

        plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors
    
    print("begin scene camera")
    scene = figure.scene

    # Original
    # scene.camera.position = [118.7195754824976, 118.70290907014409, 120.11124225247899]
    scene.camera.position = [100 * 0.3, 100 * 0.3, 140 * 0.3]
    # scene.camera.focal_point = [0.008333206176757812, -0.008333206176757812, 1.399999976158142]
    scene.camera.focal_point = [-8, -8, 1.399999976158142]

    scene.camera.view_angle = 17.0
    scene.camera.view_up = [0.0, 0.0, 1.0]
    scene.camera.clipping_range = [114.42016931210819, 320.9039783052695]
    scene.camera.compute_view_plane_normal()
    scene.render()
    scene.camera.azimuth(-5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()
    scene.camera.azimuth(5)
    scene.render()

    scene.camera.elevation(5)
    scene.camera.orthogonalize_view_up()
    scene.render()
    scene.camera.elevation(-5)
    mlab.pitch(-8)
    mlab.move(up=15)
    scene.camera.orthogonalize_view_up()
    scene.render()



    filepath = os.path.join(save_dir, f'{name}.png')
    if offscreen:
        mlab.savefig(filepath)
        # Open and crop
        img = Image.open(filepath)
        width, height = img.size
        cropped = img.crop((0, 0, int(width * 0.85), height))
        cropped.save(filepath)
    else:
        mlab.show()
    mlab.close()

def get_nuscenes_colormap():
    colors = np.array(
        [
            [  0,   0,   0, 255],       # others               black
            [255, 120,  50, 255],       # barrier              orange
            [255, 192, 203, 255],       # bicycle              pink
            [255, 255,   0, 255],       # bus                  yellow
            [  0, 150, 245, 255],       # car                  blue
            [  0, 255, 255, 255],       # construction_vehicle cyan
            [255, 127,   0, 255],       # motorcycle           dark orange
            [255,   0,   0, 255],       # pedestrian           red
            [255, 240, 150, 255],       # traffic_cone         light yellow
            [135,  60,   0, 255],       # trailer              brown
            [160,  32, 240, 255],       # truck                purple                
            [255,   0, 255, 255],       # driveable_surface    dark pink
            # [175,   0,  75, 255],       # other_flat           dark red
            [139, 137, 137, 255],
            [ 75,   0,  75, 255],       # sidewalk             dard purple
            [150, 240,  80, 255],       # terrain              light green          
            [230, 230, 250, 255],       # manmade              white
            [  0, 175,   0, 255],       # vegetation           green
            # [  0, 255, 127, 255],       # ego car              dark cyan
            # [255,  99,  71, 255],       # ego car
            # [  0, 191, 255, 255]        # ego car
        ]
    ).astype(np.float32) / 255.
    return colors

def get_rellis3d_colormap():
    colors = np.array(
        [
            # [  0,   0,   0, 255],       # others
            [  0,   0,   0, 255],       # others
            [0, 128, 0, 255],       # grass
            [0, 255, 0, 255],       # tree
            [255, 153, 204, 255],       # bush
            [133, 255, 240, 255],       # puddle
            [99, 66, 33, 255],       # mud
            [41, 128, 255, 255],       # barrier
            [110, 20, 138, 255],       # rubble
            # [  255,   0,   0, 255],       # empty
        ]
    ).astype(np.float32) / 255.
    return colors

def save_gaussian(save_dir, gaussian, name, dataset='nusc'):
    if dataset == 'nusc':
        sem_cmap = get_nuscenes_colormap()
        empty_label = 17
    elif dataset == 'rellis3d':
        sem_cmap = get_rellis3d_colormap()
        empty_label = 8

    means = gaussian.means[0].detach().cpu().numpy() # g, 3
    scales = gaussian.scales[0].detach().cpu().numpy() # g, 3
    rotations = gaussian.rotations[0].detach().cpu().numpy() # g, 4
    opas = gaussian.opacities[0]
    if opas.numel() == 0:
        opas = torch.ones_like(gaussian.means[0][..., :1])
    opas = opas.squeeze().detach().cpu().numpy() # g, 1
    sems = gaussian.semantics[0].detach().cpu().numpy() # g, 18

    pred = np.argmax(sems, axis=-1)
    opas_threshold = [0.04, 0.06]

    for opa in opas_threshold:
        mask = (opas > opa)

        means = means[mask]
        scales = scales[mask]
        rotations = rotations[mask]
        opas = opas[mask]
        pred = pred[mask]

        ellipNumber = means.shape[0]
        norm = colors.Normalize(vmin=-1.0, vmax=5.4)
        cmap = cm.jet
        m = cm.ScalarMappable(norm=norm, cmap=cmap)

        fig = plt.figure(figsize=(9, 9), dpi=300)
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=60, azim=-90)
        scalar = 1

        # compute each and plot each ellipsoid iteratively
        if dataset=='nusc':
            border = np.array([
                [-50.0, -50.0, 0.0],
                [-50.0, 50.0, 0.0],
                [50.0, -50.0, 0.0],
                [50.0, 50.0, 0.0],
            ])
        elif dataset=='rellis3d':
            border = np.array([
                [-20.0, -10.0, 0.0],
                [-20.0, 10.0, 0.0],
                [0.0, -10.0, 0.0],
                [0.0, 10.0, 0.0],
            ])
        ax.plot_surface(border[:, 0:1], border[:, 1:2], border[:, 2:], 
            rstride=1, cstride=1, color=[0, 0, 0, 1], linewidth=0, alpha=0., shade=True)

        for indx in range(ellipNumber):
            
            center = means[indx]
            radii = scales[indx] * scalar
            rot_matrix = rotations[indx]
            rot_matrix = Quaternion(rot_matrix).rotation_matrix.T

            # calculate cartesian coordinates for the ellipsoid surface
            u = np.linspace(0.0, 2.0 * np.pi, 10)
            v = np.linspace(0.0, np.pi, 10)
            x = radii[0] * np.outer(np.cos(u), np.sin(v))
            y = radii[1] * np.outer(np.sin(u), np.sin(v))
            z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

            xyz = np.stack([x, y, z], axis=-1) # phi, theta, 3
            xyz = rot_matrix[None, None, ...] @ xyz[..., None]
            xyz = np.squeeze(xyz, axis=-1)

            xyz = xyz + center[None, None, ...]
            ax.plot_surface(
                xyz[..., 1], -xyz[..., 0], xyz[..., 2], 
                rstride=1, cstride=1, color=sem_cmap[pred[indx]], linewidth=0, alpha=1.0, shade=True) # set visible gaussians opacity directly to 1.0
            

        plt.axis("equal")

        ax.dist = 7

        ax.grid(False)
        ax.set_axis_off()    

        filepath = os.path.join(save_dir, f'{name}_{opa}.png')
        plt.savefig(filepath)

        plt.cla()
        plt.clf()

        # Open and crop
        img = Image.open(filepath)
        width, height = img.size
        cropped = img.crop((int(width * 0.3), int(height * 0.15), int(width * 0.9), int(height * 0.7)))
        cropped.save(filepath)