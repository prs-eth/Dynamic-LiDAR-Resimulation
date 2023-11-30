import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
import os, json, torch
import torch.nn as nn
from NFLStudio.libs.utils import estimate_normal
try:
    import open3d as o3d
except:
    print('fail to load open3d')

def get_colored_img_binary(image, min, max, mask):
    """
    Input:
        image:  [H, w]
        mask:   [H, w]
    """
    H, W = image.shape
    base_img = np.ones((H, W, 3)) * 169 / 256
    colors = np.array([[1,1,1],[0,0,0]])
    base_img[mask] = colors[image[mask].astype(int)]
    return base_img

def get_colored_img(image, min, max, mask, cmap):
    """
    Input:
        image:  [H, w]
        mask:   [H, w]
    """
    n_colors = cmap.shape[0] -1 
    H, W = image.shape
    image = np.clip(image, min, max)
    image_idx = np.floor((image - min) / (max - min) * n_colors).astype(int)
    image_colors = cmap[image_idx.reshape(-1)].reshape(H, W, 3)
    base_colors = np.ones_like(image_colors) * 169 / 256
    base_colors[mask.astype(bool)] = image_colors[mask.astype(bool)]
    return base_colors

def lidar_to_range_img(inputs, outputs, extent, save_path, H = 64, W = 2650):
    """
    In Waymo dataset, the inclination range is [-17.6, 2.4] degrees
    """
    row_idx, col_idx = inputs['row_idx'], inputs['col_idx']
    first_mask_gt = inputs['first_mask']
    second_mask_gt = inputs['second_mask']
    first_mask_est = outputs['coarse']['ray_drop_prob']
    second_mask_est = outputs['coarse']['two_return_prob']
    normals = inputs['rays_normal']
    ray_dirs = inputs['rays_d']
    cos = -(normals * ray_dirs).sum(1)
    cos = torch.clamp(cos, 0, 1).cpu().numpy()
    
    if first_mask_est.ndim == 1:
        first_mask_est = first_mask_est < 0.5
    else:
        first_mask_est = first_mask_est.argmin(1).bool()
    
    if second_mask_est.ndim == 1:
        second_mask_est = second_mask_est > 0.5
    else:
        second_mask_est = second_mask_est.argmax(1).bool()
        # softmax = nn.Softmax(dim = 1)
        # second_mask_est = softmax(second_mask_est)[:,1] > 0.6


    first_mask_gt_img = np.zeros((H, W)).astype(np.float32).reshape(-1)
    eval_mask = deepcopy(first_mask_gt_img)
    first_mask_est_img = deepcopy(first_mask_gt_img)
    second_mask_gt_img = deepcopy(first_mask_gt_img)
    second_mask_est_img = deepcopy(first_mask_gt_img)
    range_gt_img = deepcopy(first_mask_gt_img)
    range_est_img = deepcopy(first_mask_gt_img)
    range_diff_img = deepcopy(first_mask_gt_img)
    intensity_gt_img = deepcopy(first_mask_gt_img)
    intensity_est_img = deepcopy(first_mask_gt_img)
    intensity_diff_img = deepcopy(first_mask_gt_img)
    range_cos_img = deepcopy(first_mask_gt_img)
    range_max_weight_img = deepcopy(first_mask_gt_img)

    idx = row_idx * W + col_idx
    idx = idx.cpu().numpy().astype(int)

    eval_mask = np.ones((H, W)).astype(bool)

    eval_mask_1 = np.zeros((H, W)).astype(np.float32).reshape(-1)
    eval_mask_1[idx[first_mask_gt.bool().cpu().numpy()].astype(int)] = 1.0
    eval_mask_1 = eval_mask_1.astype(bool).reshape(H, W)

    eval_mask_2 = np.zeros((H, W)).astype(np.float32).reshape(-1)
    eval_mask_2[idx[second_mask_gt.bool().cpu().numpy()].astype(int)] = 1.0
    eval_mask_2 = eval_mask_2.astype(bool).reshape(H, W)

    fig = plt.figure(figsize = (25, 30))
    colorbar_frac = 0.02
    n_images = 10
    CMAP='bwr'
    cmap = plt.get_cmap(CMAP)
    cmap = cmap(np.arange(256))[:,:3]  # [blue to red ---> small to big]
    
    
    ##########################
    # plot ray drop
    first_mask_gt_img[idx] = first_mask_gt.cpu().numpy().astype(np.float32)
    first_mask_gt_img = first_mask_gt_img.reshape(H, W)
    first_mask_est_img[idx] = first_mask_est.cpu().numpy().astype(np.float32)
    first_mask_est_img = first_mask_est_img.reshape(H, W)

    plt.subplot(n_images, 1, 1)
    img = get_colored_img_binary(first_mask_gt_img, 0, 1, eval_mask)
    plt.imshow(img, aspect='auto')
    plt.gca().set_title('Ray drop gt')
    # plt.colorbar(fraction = colorbar_frac)

    plt.subplot(n_images, 1, 2)
    img = get_colored_img_binary(first_mask_est_img, 0, 1, eval_mask)
    plt.imshow(img, aspect='auto')
    plt.gca().set_title('Ray drop est')
    # plt.colorbar(fraction = colorbar_frac)
    
    ##########################
    # plot two returns
    second_mask_gt_img[idx] = second_mask_gt.cpu().numpy().astype(np.float32)
    second_mask_gt_img = second_mask_gt_img.reshape(H, W)
    second_mask_est_img[idx] = second_mask_est.cpu().numpy().astype(np.float32)
    second_mask_est_img = second_mask_est_img.reshape(H, W)

    plt.subplot(n_images, 1, 3)
    img = get_colored_img_binary(second_mask_gt_img, 0, 1, eval_mask)
    plt.imshow(img, aspect='auto')
    plt.gca().set_title('two return gt')
    # plt.colorbar(fraction = colorbar_frac)

    plt.subplot(n_images, 1, 4)
    img = get_colored_img_binary(second_mask_est_img, 0, 1, eval_mask)
    plt.imshow(img, aspect='auto')
    plt.gca().set_title('two return est')
    # plt.colorbar(fraction = colorbar_frac)
    
    
    ##########################
    # plot range and intensity of the first return
    range_gt = inputs['first_dist'].cpu().numpy() * extent
    range_est = outputs['coarse']['depth_final_1'].cpu().numpy() * extent
    range_est[~first_mask_gt.cpu().numpy().astype(bool)] = 0.
    range_gt[~first_mask_gt.cpu().numpy().astype(bool)] = 0.
    range_gt_img[idx] = range_gt
    range_est_img[idx] = range_est
    range_diff_img[idx] = range_gt - range_est

    plt.subplot(n_images, 1, 5)
    max_range_diff = 0.4
    img = get_colored_img(range_diff_img.reshape(H, W), -max_range_diff, max_range_diff, eval_mask_1, cmap)
    plt.imshow(img, aspect='auto')
    plt.gca().set_title('First return: Range_gt - range_est')
    # plt.colorbar(fraction = colorbar_frac)

    # plot the intensity 
    intensity_gt = inputs['first_intensity'].cpu().numpy()
    intensity_est = outputs['coarse']['intensity'].cpu().numpy()
    intensity_est[~first_mask_gt.cpu().numpy().astype(bool)] = 0.
    intensity_gt[~first_mask_gt.cpu().numpy().astype(bool)] = 0.

    intensity_gt_img[idx] = intensity_gt
    intensity_est_img[idx] = intensity_est
    intensity_diff_img[idx] = intensity_gt - intensity_est
    
    plt.subplot(n_images, 1, 6)
    max_intensity_diff = 0.1
    img = get_colored_img(intensity_diff_img.reshape(H, W), -max_intensity_diff, max_intensity_diff, eval_mask_1, cmap)
    plt.imshow(img, aspect='auto')
    plt.gca().set_title('First return: Intensity gt - intensity est')
    # plt.colorbar(fraction = colorbar_frac)
    
    ##########################
    # plot range and intensity of the second return
    range_gt_img = np.zeros((H, W)).astype(np.float32).reshape(-1)
    range_est_img = np.zeros((H, W)).astype(np.float32).reshape(-1)
    range_diff_img = np.zeros((H, W)).astype(np.float32).reshape(-1)
    
    range_gt = inputs['second_dist'].cpu().numpy() * extent
    range_est = outputs['coarse']['depth_final_2'].cpu().numpy() * extent
    range_est[~second_mask_gt.cpu().numpy().astype(bool)] = 0.
    range_gt[~second_mask_gt.cpu().numpy().astype(bool)] = 0.
    range_gt_img[idx] = range_gt
    range_est_img[idx] = range_est
    range_diff_img[idx] = range_gt - range_est


    plt.subplot(n_images, 1, 7)
    max_range_diff = 0.8
    img = get_colored_img(range_diff_img.reshape(H, W), -max_range_diff, max_range_diff, eval_mask_2, cmap)
    plt.imshow(img, aspect='auto')
    plt.gca().set_title('Second return: Range_gt - range_est')
    # plt.colorbar(fraction = colorbar_frac)

    # plot the intensity 
    intensity_gt_img = np.zeros((H, W)).astype(np.float32).reshape(-1)
    intensity_est_img = np.zeros((H, W)).astype(np.float32).reshape(-1)
    intensity_diff_img = np.zeros((H, W)).astype(np.float32).reshape(-1)
    intensity_gt = inputs['second_intensity'].cpu().numpy()
    intensity_est = outputs['coarse']['second_intensity'].cpu().numpy()
    intensity_est[~second_mask_gt.cpu().numpy().astype(bool)] = 0.
    intensity_gt[~second_mask_gt.cpu().numpy().astype(bool)] = 0.

    intensity_gt_img[idx] = intensity_gt
    intensity_est_img[idx] = intensity_est
    intensity_diff_img[idx] = intensity_gt - intensity_est

    plt.subplot(n_images, 1, 8)
    max_intensity_diff = 0.1
    img = get_colored_img(intensity_diff_img.reshape(H, W), -max_intensity_diff, max_intensity_diff, eval_mask_2, cmap)
    plt.imshow(img, aspect='auto')
    plt.gca().set_title('Second return: Intensity gt - intensity est')
    
    
    max_weights = outputs['coarse']['max_weights'].cpu().numpy()
    max_weights[~first_mask_gt.cpu().numpy().astype(bool)] = 0.
    range_max_weight_img[idx] = max_weights
    
    plt.subplot(n_images, 1,9)
    img = get_colored_img(range_max_weight_img.reshape(H, W), 0, 1, eval_mask, cmap)
    plt.imshow(img, aspect='auto')
    plt.gca().set_title('max weight')
    
    # # plot max weights and incidence angles
    # cos[~first_mask_gt.cpu().numpy().astype(bool)] = 0.
    # range_cos_img[idx] = cos

    # plt.subplot(n_images, 1, 8)
    # img = get_colored_img(range_cos_img.reshape(H, W), 0, 1, eval_mask, cmap)
    # plt.imshow(img, aspect='auto')
    # plt.gca().set_title('Incidence angle')
    
    plt.savefig(save_path, pad_inches=0.1, bbox_inches='tight')
    plt.close('all')


def render_pcd_with_bbox(pcds, bbox_groups, path_render_option, path_camera, save_path):
    # load render option
    assert os.path.exists(path_render_option)
    assert os.path.exists(path_camera)

    # load camera parameters
    cam_params = json.load(open(path_camera))
    extrinsic = np.array(cam_params['extrinsic']).reshape(4,4).T
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic_params = cam_params['intrinsic']
    intrinsic.height = intrinsic_params['height']
    intrinsic.width = intrinsic_params['width']
    intrinsic.intrinsic_matrix = np.array(intrinsic_params['intrinsic_matrix']).reshape(3, 3).T
    cam = o3d.camera.PinholeCameraParameters()
    cam.extrinsic = extrinsic
    cam.intrinsic = intrinsic  

    # initialise the windows
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=intrinsic_params['width'], height=intrinsic_params['height'], left=0, top=0)
    for ele in pcds:
        estimate_normal(ele, 0.3, 50)
    for eachpcd in pcds:
        vis.add_geometry(eachpcd)

    color_maps = [[0,0,0],[1,0.647,0], [0,0,1], [0,1,0], [1,1,0]]
    lines = [[0, 1], [1, 2], [2, 3], [0, 3],
        [4, 5], [5, 6], [6, 7], [4, 7],
        [0, 4], [1, 5], [2, 6], [3, 7]]
    for i in range(len(bbox_groups)):
        colors = [color_maps[i] for _ in range(len(lines))]
        for j in range(len(bbox_groups[i])):
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(bbox_groups[i][j])
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            vis.add_geometry(line_set) 
    # # set the parameters
    vis.get_view_control().convert_from_pinhole_camera_parameters(cam, allow_arbitrary=True)
    vis.get_render_option().load_from_json(path_render_option)
    vis.update_renderer()
    vis.capture_screen_image(save_path, do_render=True)

    # while True:
    #     vis.update_geometry(pcd)
    #     if not vis.poll_events():
    #         break
    #     vis.update_renderer()
    
    vis.destroy_window()

def render_pcd(pcds, path_render_option, path_camera, save_path):
    # load render option
    assert os.path.exists(path_render_option)
    assert os.path.exists(path_camera)

    # load camera parameters
    cam_params = json.load(open(path_camera))
    extrinsic = np.array(cam_params['extrinsic']).reshape(4,4).T
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic_params = cam_params['intrinsic']
    intrinsic.height = intrinsic_params['height']
    intrinsic.width = intrinsic_params['width']
    intrinsic.intrinsic_matrix = np.array(intrinsic_params['intrinsic_matrix']).reshape(3, 3).T
    cam = o3d.camera.PinholeCameraParameters()
    cam.extrinsic = extrinsic
    cam.intrinsic = intrinsic  

    # initialise the windows
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=intrinsic_params['width'], height=intrinsic_params['height'], left=0, top=0)
    for ele in pcds:
        estimate_normal(ele, 0.3, 50)
    for eachpcd in pcds:
        vis.add_geometry(eachpcd)

    # # set the parameters
    vis.get_view_control().convert_from_pinhole_camera_parameters(cam, allow_arbitrary=True)
    vis.get_render_option().load_from_json(path_render_option)
    vis.update_renderer()
    vis.capture_screen_image(save_path, do_render=True)

    # while True:
    #     vis.update_geometry(pcd)
    #     if not vis.poll_events():
    #         break
    #     vis.update_renderer()
    
    vis.destroy_window()
    


def lidar_to_range_img_geometry(inputs, outputs, extent, save_path, H = 64, W = 2650):
    """
    In Waymo dataset, the inclination range is [-17.6, 2.4] degrees
    """
    row_idx, col_idx = inputs['row_idx'], inputs['col_idx']
    first_mask_gt = inputs['first_mask']
    normals = inputs['rays_normal']
    ray_dirs = inputs['rays_d']
    cos = -(normals * ray_dirs).sum(1)
    cos = torch.clamp(cos, 0, 1).cpu().numpy()

    first_mask_gt_img = np.zeros((H, W)).astype(np.float32).reshape(-1)
    eval_mask = deepcopy(first_mask_gt_img)
    range_gt_img = deepcopy(first_mask_gt_img)
    range_vol_est_img = deepcopy(first_mask_gt_img)
    range_peak_est_img = deepcopy(first_mask_gt_img)
    range_vol_diff_img = deepcopy(first_mask_gt_img)
    range_peak_diff_img = deepcopy(first_mask_gt_img)
    range_max_weight_img = deepcopy(first_mask_gt_img)
    range_sum_weight_img = deepcopy(first_mask_gt_img)
    range_cos_img = deepcopy(first_mask_gt_img)
    range_sample_dist_img = deepcopy(first_mask_gt_img)

    idx = row_idx * W + col_idx

    eval_mask[idx[first_mask_gt.bool()].cpu().numpy().astype(int)] = 1.0
    eval_mask = eval_mask.astype(bool).reshape(H, W)
    idx = idx.cpu().numpy().astype(int)

    first_mask_gt_img[idx] = first_mask_gt.cpu().numpy().astype(np.float32)
    first_mask_gt_img = first_mask_gt_img.reshape(H, W)

    fig = plt.figure(figsize = (25, 26))
    colorbar_frac = 0.02
    n_images = 8
    CMAP='bwr'
    cmap = plt.get_cmap(CMAP)
    cmap = cmap(np.arange(256))[:,:3]  # [blue to red ---> small to big]

    # plot range values
    max_range = 80
    range_gt = inputs['first_dist'].cpu().numpy() * extent
    range_est_vol = outputs['coarse']['depth_final_1'].cpu().numpy() * extent
    range_est_vol[~first_mask_gt.cpu().numpy().astype(bool)] = 0.
    range_gt[~first_mask_gt.cpu().numpy().astype(bool)] = 0.
    range_gt_img[idx] = range_gt
    range_vol_est_img[idx] = range_est_vol
    range_vol_diff_img[idx] = range_gt - range_est_vol

    range_est_peak = outputs['coarse']['depth_from_peak_1'].cpu().numpy() * extent
    range_est_peak[~first_mask_gt.cpu().numpy().astype(bool)] = 0.
    range_gt[~first_mask_gt.cpu().numpy().astype(bool)] = 0.
    range_gt_img[idx] = range_gt
    range_peak_est_img[idx] = range_est_peak
    range_peak_diff_img[idx] = range_gt - range_est_peak

    max_weights = outputs['coarse']['max_weights'].cpu().numpy()
    max_weights[~first_mask_gt.cpu().numpy().astype(bool)] = 0.
    range_max_weight_img[idx] = max_weights

    sum_weights = outputs['coarse']['weights_1'].sum(1).cpu().numpy()
    sum_weights[~first_mask_gt.cpu().numpy().astype(bool)] = 0.
    range_sum_weight_img[idx] = sum_weights

    sample_dist = outputs['coarse']['sample_dist'].cpu().numpy() * extent
    range_sample_dist_img[idx] = sample_dist

    cos[~first_mask_gt.cpu().numpy().astype(bool)] = 0.
    range_cos_img[idx] = cos

    plt.subplot(n_images, 1, 1)
    img = get_colored_img(range_gt_img.reshape(H,W), 0, max_range, eval_mask, cmap)
    plt.imshow(img, aspect='auto')
    plt.gca().set_title('Range gt')

    plt.subplot(n_images, 1, 2)
    img = get_colored_img(range_vol_est_img.reshape(H, W), 0, max_range, eval_mask, cmap)
    plt.imshow(img, aspect='auto')
    plt.gca().set_title('Range est vol')

    plt.subplot(n_images, 1, 3)
    img = get_colored_img(range_peak_est_img.reshape(H, W), 0, max_range, eval_mask, cmap)
    plt.imshow(img, aspect='auto')
    plt.gca().set_title('Range est peak')

    plt.subplot(n_images, 1, 4)
    max_range_diff = 0.3
    img = get_colored_img(range_vol_diff_img.reshape(H, W), -max_range_diff, max_range_diff, eval_mask, cmap)
    plt.imshow(img, aspect='auto')
    plt.gca().set_title('Range_gt - range_est_vol')

    plt.subplot(n_images, 1, 5)
    max_range_diff = 0.3
    img = get_colored_img(range_peak_diff_img.reshape(H, W), -max_range_diff, max_range_diff, eval_mask, cmap)
    plt.imshow(img, aspect='auto')
    plt.gca().set_title('Range_gt - range_est_peak')

    plt.subplot(n_images, 1, 6)
    img = get_colored_img(range_max_weight_img.reshape(H, W), 0, 1, eval_mask, cmap)
    plt.imshow(img, aspect='auto')
    plt.gca().set_title('max weight')

    plt.subplot(n_images, 1, 7)
    img = get_colored_img(range_cos_img.reshape(H, W), 0, 1, eval_mask, cmap)
    plt.imshow(img, aspect='auto')
    plt.gca().set_title('Incidence angle')

    plt.subplot(n_images, 1, 8)
    img = get_colored_img(range_sample_dist_img.reshape(H, W), 0.1, 0.2, eval_mask, cmap)
    plt.imshow(img, aspect='auto')
    plt.gca().set_title('Sample distance')

    plt.savefig(save_path, pad_inches=0.1, bbox_inches='tight')
    plt.close('all')