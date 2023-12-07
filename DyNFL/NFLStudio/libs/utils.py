"""
General utility functions

Author: Shengyu Huang, Hanfeng Wu
Last modified: 30.11.2023
"""

import os,re,sys,json,yaml,random, argparse, torch, pickle, imageio
import numpy as np
try:
    import open3d as o3d
except:
    print('fail to import open3d')
try:
    import cv2
except:
    print('fail to import cv2')
import multiprocessing as mp
from tqdm import tqdm
count_job = 0
import multiprocessing as mp

_EPS = 1e-20  # To prevent division by zero
PI = np.pi

def get_cpu_count():
    return mp.cpu_count()


def play_video(path, name):
    # Create a VideoCapture object and read from input file
    cap = cv2.VideoCapture(path)
    
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video  file")
    
    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # Display the resulting frame
            cv2.imshow(name, frame)
        
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
        # Break the loop
        else: 
            break
    
    # When everything done, release 
    # the video capture object
    cap.release()
    
    # Closes all the frames
    cv2.destroyAllWindows()


def partial_load(pretrained_dict, model):
    model_dict = model.state_dict()
    
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].size()==v.size()}

    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    

def mp_process(func, task):
    p = mp.Pool(processes=mp.cpu_count())
    p.map(func, task)
    p.close()
    p.join()

def dict_to_array(c_dict):
    """
    Convert a dictionary to array, both key and value are numeric values
    Return:
        c_array:    c_array[key] = c_dict[key]
    """
    n_elements = max([ele for ele,_ in c_dict.items()])+1
    c_array = np.zeros([n_elements]).astype(int32)
    for key, value in c_dict.items():
        c_array[key] = int(value)
    return c_array
    

def lighter(color, percent):
    '''assumes color is rgb between (0, 0, 0) and (1,1,1)'''
    if percent.ndim == 1:
        percent = percent[:,None].repeat(3,1)

    percent = 1 - percent
    color = np.array(color)
    white = np.array([1, 1, 1])
    vector = white-color
    return color + vector * percent


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

class Logger:
    def __init__(self, path):
        self.path = path
        self.fw = open(self.path+'/log','a')

    def write(self, text):
        self.fw.write(text)
        self.fw.flush()

    def close(self):
        self.fw.close()

def load_pcd(file_name):
    pcd = o3d.io.read_point_cloud(file_name)
    return pcd


def save_pcd(pcd, file_name):
    """
    save a point cloud to ply file
    """
    o3d.io.write_point_cloud(file_name, pcd)


def load_pkl(path):
    """
    Load a .pkl object
    """
    file = open(path ,'rb')
    return pickle.load(file)

def save_pkl(obj, path ):
    """
    save a dictionary to a pickle file
    """
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_yaml(path):
    """
    Loads configs from .yaml file

    Args:
        path (str): path to the config file

    Returns: 
        config (dict): dictionary of the configuration parameters, merge sub_dicts

    """
    with open(path,'r') as f:
        cfg = yaml.safe_load(f)

    return cfg


def setup_seed(seed):
    """
    fix random seed for deterministic training
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def square_distance(src, dst, normalised = False):
    """
    Calculate Euclid distance between each two points.
    Args:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Returns:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    if(normalised):
        dist += 2
    else:
        dist += torch.sum(src ** 2, dim=-1)[:, :, None]
        dist += torch.sum(dst ** 2, dim=-1)[:, None, :]

    dist = torch.clamp(dist, min=1e-12, max=None)
    return dist
    

def validate_gradient(model):
    """
    Confirm all the gradients are non-nan and non-inf
    """
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.numel() > 0:
            # print(param.grad.max(), torch.quantile(param.grad, 0.995))    
            if torch.any(torch.isnan(param.grad)):
                return False
            if torch.any(torch.isinf(param.grad)):
                return False
    return True


def natural_key(string_):
    """
    Sort strings by numbers in the name
    """
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


def makedirs(folder):
    if(os.path.exists(folder)):
        return 
    else:
        os.makedirs(folder)

def get_percentile_numpy(data):
    percentiles = dict()
    keys = [5,10,25,50,75,90,95]
    for x in keys:
        percentiles[x] = np.percentile(data, x)
    return percentiles

def get_percentile_torch(data):
    percentiles = dict()
    keys = [5,10,25,50,75,90,95]
    for x in keys:
        percentiles[x] = torch.quantile(data, x)
    return percentiles


def get_gray():
    return [169 / 255, 169 / 255, 169 / 255]

def get_blue():
    """
    Get color blue for rendering
    """
    return [0, 0.651, 0.929]

def get_red():
    return [1,0,0]

def get_yellow():
    """
    Get color yellow for rendering
    """
    return [1, 0.706, 0]

def to_tensor(array):
    """
    Convert array to tensor
    """
    if(not isinstance(array,torch.Tensor)):
        return torch.from_numpy(array)
    else:
        return array

def to_array(tensor):
    """
    Conver tensor to array
    """
    if(not isinstance(tensor,np.ndarray)):
        if(tensor.device == torch.device('cpu')):
            return tensor.numpy()
        else:
            return tensor.cpu().numpy()
    else:
        return tensor

def to_tsfm(rot,trans):
    tsfm = np.eye(4)
    tsfm[:3,:3]=rot
    tsfm[:3,3]=trans.flatten()
    return tsfm

def to_o3d_vec(vec):
    """
    Create open3d array objects
    """
    return o3d.utility.Vector3dVector(to_array(vec))
    
def to_o3d_pcd(xyz):
    """
    Convert tensor/array to open3d PointCloud
    xyz:       [N, 3]
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(to_array(xyz).astype(np.float64))
    return pcd

def to_o3d_feats(embedding):
    """
    Convert tensor/array to open3d features
    embedding:  [N, 3]
    """
    feats = o3d.pipelines.registration.Feature()
    feats.data = to_array(embedding).T
    return feats

def value_to_color_idx(values, n_colors, percentile = 95, max_value = None, min_value = 0):
    if max_value is None:
        max_value = np.percentile(values, percentile)
    
    diff = max_value - min_value
    values = (values - min_value) / diff
    values = np.clip(values, 0, 1)
    color_idx = (values // (1 / n_colors)).astype(int)
    return color_idx 

def canonicalise_random_indice(indice):
    """
    Convert randomly scattered indice(unordered) to canonical spcae.
    For example: [1,4,4,6,10] ----> [0,1,1,2,3]
    Input:
        indice: list
    """
    unique_ids = sorted(set(indice))  # make sure -1 is mapped to 0
    mapping_dict = dict()
    for idx,ele in enumerate(unique_ids):
        mapping_dict[ele] = idx
    
    mapped_list = [mapping_dict[ele] for ele in indice]
    return mapped_list


def get_cdf(data, min_val, max_val, intervals = 100):
    data = to_array(data)
    sampled_data = np.linspace(min_val, max_val, intervals)
    x_range = max_val - min_val
    ratios = []
    for ele in sampled_data:
        c_ratio = (data < ele).mean()
        ratios.append(c_ratio)
    
    return sampled_data, np.array(ratios)

def cat_movies_side_by_side(path_1, path_2, save_path):
    from moviepy.editor import VideoFileClip, clips_array
    clip1 = VideoFileClip(path_1).margin(10) # add 10px contour
    clip2 = VideoFileClip(path_2).margin(10) # add 10px contour
    final_clip = clips_array([[clip1, clip2]])
    final_clip.write_videofile(save_path)

def compose_figures_to_movie(filenames, save_path, fps = 10):
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(filenames, fps=fps)
    clip.write_videofile(save_path)

def compose_figures_to_gif(filenames, save_path):
    images = []
    for filename in tqdm(filenames):
        images.append(imageio.imread(filename))
    imageio.mimsave(save_path, images,duration=0.1)


def vis_o3d(pcds, render=True, window_name = None, add_frame=True):
    """
    Input:
        pcds:   a list of open3d objects
    """
    width=1600
    height = 1440
    
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    if(render):
        for eachpcd in pcds:
            try:
                eachpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
            except:
                pass
    if add_frame:
        pcds.append(mesh_frame)
    if(window_name is not None):
        o3d.visualization.draw_geometries(pcds, window_name=window_name, width= width, height= height)
    else:
        o3d.visualization.draw_geometries(pcds, width= width, height= height)


def vis_ray_cast(pts_gt, pts_est):
    flow = pts_gt - pts_est
    base_pcd = o3d.geometry.PointCloud()
    base_pcd.points = o3d.utility.Vector3dVector(pts_gt)
    base_pcd.paint_uniform_color([1.0, 0., 0.])
    corres_lineset = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.vstack([pts_gt, pts_est])),
        lines=o3d.utility.Vector2iVector(np.arange(2 * pts_gt.shape[0]).reshape((2, -1)).T))
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(pts_est)
    target_pcd.paint_uniform_color([0,0,1])
    vis_o3d([base_pcd, target_pcd, corres_lineset], window_name='GT: red ------ Est: Blue')

def vis_point_flow(base_pc, base_flow, target_pc):
    """
    Visualise the scene flows
    """
    print("Start from red, go to green, target is blue.")
    base_pcd = o3d.geometry.PointCloud()
    base_pcd.points = o3d.utility.Vector3dVector(base_pc)
    base_pcd.paint_uniform_color([1.0, 0., 0.])
    final_pcd = o3d.geometry.PointCloud()
    final_pcd.points = o3d.utility.Vector3dVector(base_pc+base_flow)
    final_pcd.paint_uniform_color([0., 1.0, 0.])
    corres_lineset = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.vstack([base_pc, base_pc + base_flow])),
        lines=o3d.utility.Vector2iVector(np.arange(2 * base_flow.shape[0]).reshape((2, -1)).T))
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_pc)
    target_pcd.paint_uniform_color([0,0,1])
    vis_o3d([base_pcd, final_pcd, target_pcd, corres_lineset], window_name='GT: red,    Est: blue')
    
def estimate_normal(pcd, radius, max_nn):
    if isinstance(pcd, o3d.geometry.PointCloud):
        if not pcd.has_normals():
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))

def multi_vis_with_bbox(pcds, names, bboxs, render = True, width = 960, height = 540, shift = 100,add_frame = True):
    def add_geometry(vis, pcds, mesh_frame, bbox_groups):
        color_maps = [[1,0,0], [0,0,1], [0,1,0], [1,1,0]]
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
        for ele in pcds:
            vis.add_geometry(ele)
        if add_frame:
            vis.add_geometry(mesh_frame)
    
    def update_geometry(vis,pcds, mesh_frame):
        for ele in pcds:
            vis.update_geometry(ele)
        if add_frame:
            vis.update_geometry(mesh_frame)

    """
    Visulise point clouds in multiple windows, we allow at most 4 windows

    Input:
        pcds:   a list of pcds
        names:  a list of window names
    """
    import pyautogui
    width, height = pyautogui.size()
    width = (width - 100) // 2
    height = (height - 100) // 2

    assert len(pcds) == len(names) == len(bboxs)
    n_windows = len(pcds)

    window_corners = [
        [0,0],
        [0,height+shift],
        [width+shift, 0],
        [width+shift, height+shift]
    ]
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])

    # estimate normals for better visualisation
    if(render):
        for each_pcd in pcds:
            for ele in each_pcd:
                estimate_normal(ele, 0.3, 50)

    # initialise the windows
    vis_1 = o3d.visualization.Visualizer()
    vis_1.create_window(window_name=names[0], width=width, height=height, left=window_corners[0][0], top=window_corners[0][1])
    add_geometry(vis_1, pcds[0], mesh_frame, bboxs[0])

    if(n_windows >= 2):
        vis_2 = o3d.visualization.Visualizer()
        vis_2.create_window(window_name=names[1], width=width, height=height, left=window_corners[1][0], top=window_corners[1][1])
        add_geometry(vis_2, pcds[1], mesh_frame, bboxs[1])

        if(n_windows >=3):
            vis_3 = o3d.visualization.Visualizer()
            vis_3.create_window(window_name=names[2], width=width, height=height, left=window_corners[2][0], top=window_corners[2][1])
            add_geometry(vis_3, pcds[2], mesh_frame,bboxs[2])

            if(n_windows>=4):
                vis_4 = o3d.visualization.Visualizer()
                vis_4.create_window(window_name=names[3], width=width, height=height, left=window_corners[3][0], top=window_corners[3][1])
                add_geometry(vis_4, pcds[3], mesh_frame, bboxs[3])

    # start rendering
    while True:
        update_geometry(vis_1, pcds[0], mesh_frame)
        if not vis_1.poll_events():
            break
        vis_1.update_renderer()

        if(n_windows>=2):
            update_geometry(vis_2, pcds[1], mesh_frame)
            if not vis_2.poll_events():
                break
            vis_2.update_renderer()

            cam = vis_1.get_view_control().convert_to_pinhole_camera_parameters()
            cam2 = vis_2.get_view_control().convert_to_pinhole_camera_parameters()
            cam2.extrinsic = cam.extrinsic
            vis_2.get_view_control().convert_from_pinhole_camera_parameters(cam2)


            if(n_windows>=3):
                update_geometry(vis_3, pcds[2], mesh_frame)
                if not vis_3.poll_events():
                    break
                vis_3.update_renderer()

                cam3 = vis_3.get_view_control().convert_to_pinhole_camera_parameters()
                cam3.extrinsic = cam.extrinsic
                vis_3.get_view_control().convert_from_pinhole_camera_parameters(cam3)

                if(n_windows>=4):
                    update_geometry(vis_4, pcds[3], mesh_frame)
                    if not vis_4.poll_events():
                        break
                    vis_4.update_renderer()

                    cam4 = vis_4.get_view_control().convert_to_pinhole_camera_parameters()
                    cam4.extrinsic = cam.extrinsic
                    vis_4.get_view_control().convert_from_pinhole_camera_parameters(cam4)
    
    vis_1.destroy_window()
    if(n_windows>=2):
        vis_2.destroy_window()

        if(n_windows>=3):
            vis_3.destroy_window()

            if(n_windows>=4):
                vis_4.destroy_window()
def multi_vis(pcds, names, render = True, width = 960, height = 540, shift = 100,add_frame = True):
    def add_geometry(vis, pcds, mesh_frame):
        for ele in pcds:
            vis.add_geometry(ele)
        if add_frame:
            vis.add_geometry(mesh_frame)
    
    def update_geometry(vis,pcds, mesh_frame):
        for ele in pcds:
            vis.update_geometry(ele)
        if add_frame:
            vis.update_geometry(mesh_frame)

    """
    Visulise point clouds in multiple windows, we allow at most 4 windows

    Input:
        pcds:   a list of pcds
        names:  a list of window names
    """
    import pyautogui
    width, height = pyautogui.size()
    width = (width - 100) // 2
    height = (height - 100) // 2

    assert len(pcds) == len(names)
    n_windows = len(pcds)

    window_corners = [
        [0,0],
        [0,height+shift],
        [width+shift, 0],
        [width+shift, height+shift]
    ]
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])

    # estimate normals for better visualisation
    if(render):
        for each_pcd in pcds:
            for ele in each_pcd:
                estimate_normal(ele, 0.3, 50)

    # initialise the windows
    vis_1 = o3d.visualization.Visualizer()
    vis_1.create_window(window_name=names[0], width=width, height=height, left=window_corners[0][0], top=window_corners[0][1])
    add_geometry(vis_1, pcds[0], mesh_frame)

    if(n_windows >= 2):
        vis_2 = o3d.visualization.Visualizer()
        vis_2.create_window(window_name=names[1], width=width, height=height, left=window_corners[1][0], top=window_corners[1][1])
        add_geometry(vis_2, pcds[1], mesh_frame)

        if(n_windows >=3):
            vis_3 = o3d.visualization.Visualizer()
            vis_3.create_window(window_name=names[2], width=width, height=height, left=window_corners[2][0], top=window_corners[2][1])
            add_geometry(vis_3, pcds[2], mesh_frame)

            if(n_windows>=4):
                vis_4 = o3d.visualization.Visualizer()
                vis_4.create_window(window_name=names[3], width=width, height=height, left=window_corners[3][0], top=window_corners[3][1])
                add_geometry(vis_4, pcds[3], mesh_frame)

    # start rendering
    while True:
        update_geometry(vis_1, pcds[0], mesh_frame)
        if not vis_1.poll_events():
            break
        vis_1.update_renderer()

        if(n_windows>=2):
            update_geometry(vis_2, pcds[1], mesh_frame)
            if not vis_2.poll_events():
                break
            vis_2.update_renderer()

            cam = vis_1.get_view_control().convert_to_pinhole_camera_parameters()
            cam2 = vis_2.get_view_control().convert_to_pinhole_camera_parameters()
            cam2.extrinsic = cam.extrinsic
            vis_2.get_view_control().convert_from_pinhole_camera_parameters(cam2)


            if(n_windows>=3):
                update_geometry(vis_3, pcds[2], mesh_frame)
                if not vis_3.poll_events():
                    break
                vis_3.update_renderer()

                cam3 = vis_3.get_view_control().convert_to_pinhole_camera_parameters()
                cam3.extrinsic = cam.extrinsic
                vis_3.get_view_control().convert_from_pinhole_camera_parameters(cam3)

                if(n_windows>=4):
                    update_geometry(vis_4, pcds[3], mesh_frame)
                    if not vis_4.poll_events():
                        break
                    vis_4.update_renderer()

                    cam4 = vis_4.get_view_control().convert_to_pinhole_camera_parameters()
                    cam4.extrinsic = cam.extrinsic
                    vis_4.get_view_control().convert_from_pinhole_camera_parameters(cam4)
    
    vis_1.destroy_window()
    if(n_windows>=2):
        vis_2.destroy_window()

        if(n_windows>=3):
            vis_3.destroy_window()

            if(n_windows>=4):
                vis_4.destroy_window()


import torch

def interpolate_rotation(rot1, rot2, t):
    """
    Interpolate rotation matrices using spherical linear interpolation (slerp).

    Arguments:
    - rot1: The starting rotation matrix of shape (batch_size, 3, 3)
    - rot2: The ending rotation matrix of shape (batch_size, 3, 3)
    - t: The interpolation factor between 0 and 1

    Returns:
    - interpolated_rot: The interpolated rotation matrix of shape (batch_size, 3, 3)
    """
    # Convert rotation matrices to quaternions
    quat1 = torch.nn.functional.normalize(rotation_matrix_to_quaternion(rot1), dim=-1)
    quat2 = torch.nn.functional.normalize(rotation_matrix_to_quaternion(rot2), dim=-1)

    # Perform spherical linear interpolation (slerp) between quaternions
    dot = torch.sum(quat1 * quat2, dim=-1, keepdim=True)
    dot = torch.clamp(dot, -1.0, 1.0)


    theta = torch.acos(dot) * t[:, None]
    sin_theta = torch.sin(theta)
    sin_theta_complement = torch.sin((1.0 - t[:, None]) * theta)

    small_angle_mask = (sin_theta <= 1e-6)
    quat_interpolated = torch.where(small_angle_mask, quat1, (quat1 * sin_theta_complement + quat2 * sin_theta) / sin_theta)

    # Normalize the interpolated quaternions
    interpolated_quat = torch.nn.functional.normalize(quat_interpolated, dim=-1)

    # Convert interpolated quaternions back to rotation matrices
    interpolated_rot = quaternion_to_rotation_matrix(interpolated_quat)

    return interpolated_rot


def interpolate_translation(trans1, trans2, t):
    """
    Interpolate translations using linear interpolation.

    Arguments:
    - trans1: The starting translation vector of shape (batch_size, 3)
    - trans2: The ending translation vector of shape (batch_size, 3)
    - t: The interpolation factor between 0 and 1 of shape (batch_size,)

    Returns:
    - interpolated_trans: The interpolated translation vector of shape (batch_size, 3)
    """
    interpolated_trans = trans1 + (trans2 - trans1) * t[:, None]
    return interpolated_trans


def interpolate_rigid_transformation(transform1, transform2, t):
    """
    Interpolate rigid body transformations.

    Arguments:
    - transform1: The starting transformation matrix of shape (batch_size, 4, 4)
    - transform2: The ending transformation matrix of shape (batch_size, 4, 4)
    - t: The interpolation factor between 0 and 1 of shape (batch_size,)

    Returns:
    - interpolated_transform: The interpolated transformation matrix of shape (batch_size, 4, 4)
    """
    rot1 = transform1[:, :3, :3]
    trans1 = transform1[:, :3, 3]
    rot2 = transform2[:, :3, :3]
    trans2 = transform2[:, :3, 3]

    interpolated_rot = interpolate_rotation(rot1, rot2, t)
    interpolated_trans = interpolate_translation(trans1, trans2, t)

    interpolated_transform = torch.cat([interpolated_rot, interpolated_trans.unsqueeze(-1)], dim=-1)
    interpolated_transform = torch.cat([interpolated_transform, torch.tensor([[[0, 0, 0, 1]]], dtype=torch.float32).expand(transform1.shape[0], -1, -1)], dim=1)

    return interpolated_transform


def rotation_matrix_to_quaternion(rot):
    """
    Convert rotation matrices to quaternions.

    Arguments:
    - rot: The rotation matrix of shape (batch_size, 3, 3)

    Returns:
    - quat: The quaternion representation of shape (batch_size, 4)
    """
    m00 = rot[:, 0, 0]
    m01 = rot[:, 0, 1]
    m02 = rot[:, 0, 2]
    m10 = rot[:, 1, 0]
    m11 = rot[:, 1, 1]
    m12 = rot[:, 1, 2]
    m20 = rot[:, 2, 0]
    m21 = rot[:, 2, 1]
    m22 = rot[:, 2, 2]


  
    trace = m00 + m11 + m22


    if trace > 0: 
      S = torch.sqrt(trace+1.0) * 2 # S=4*qw 
      qw = 0.25 * S
      qx = (m21 - m12) / S
      qy = (m02 - m20) / S
      qz = (m10 - m01) / S
    elif ((m00 > m11) and (m00 > m22)):
      S = torch.sqrt(1.0 + m00 - m11 - m22) * 2 # S=4*qx 
      qw = (m21 - m12) / S
      qx = 0.25 * S
      qy = (m01 + m10) / S
      qz = (m02 + m20) / S
    elif (m11 > m22):
      S = torch.sqrt(1.0 + m11 - m00 - m22) * 2 # S=4*qy
      qw = (m02 - m20) / S
      qx = (m01 + m10) / S
      qy = 0.25 * S
      qz = (m12 + m21) / S
    else:
      S = torch.sqrt(1.0 + m22 - m00 - m11) * 2 # S=4*qz
      qw = (m10 - m01) / S
      qx = (m02 + m20) / S
      qy = (m12 + m21) / S
      qz = 0.25 * S
    
    # r = torch.sqrt(1 + trace) + 1e-5
    # s = 0.5 / r

    # x = (m21 - m12) * s
    # y = (m02 - m20) * s
    # z = (m10 - m01) * s
    # w = 0.5 * r

    quat = torch.stack([qx, qy, qz, qw], dim=-1)
    return quat


def quaternion_to_rotation_matrix(quat):
    """
    Convert quaternions to rotation matrices.

    Arguments:
    - quat: The quaternion representation of shape (batch_size, 4)

    Returns:
    - rot: The rotation matrix of shape (batch_size, 3, 3)
    """
    x = quat[:, 0]
    y = quat[:, 1]
    z = quat[:, 2]
    w = quat[:, 3]

    x2 = x * x
    y2 = y * y
    z2 = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    m00 = 1 - 2 * (y2 + z2)
    m01 = 2 * (xy - wz)
    m02 = 2 * (xz + wy)
    m10 = 2 * (xy + wz)
    m11 = 1 - 2 * (x2 + z2)
    m12 = 2 * (yz - wx)
    m20 = 2 * (xz - wy)
    m21 = 2 * (yz + wx)
    m22 = 1 - 2 * (x2 + y2)

    rot = torch.stack([torch.stack([m00, m01, m02], dim=-1),
                      torch.stack([m10, m11, m12], dim=-1),
                      torch.stack([m20, m21, m22], dim=-1)], dim=-2)
    return rot

def visulize_with_bbox(points3D, bbox, colors=None):

    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points3D)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        pcd.colors = o3d.utility.Vector3dVector(np.zeros_like(points3D))    
    pcd.paint_uniform_color(get_gray())
    estimate_normal(pcd,0.003,50)
    viewer.add_geometry(pcd)
    lines = [[0, 1], [1, 2], [2, 3], [0, 3],
             [4, 5], [5, 6], [6, 7], [4, 7],
             [0, 4], [1, 5], [2, 6], [3, 7]]

    # Use the same color for all lines
    colors = [[1, 0, 0] for _ in range(len(lines))]
    for i in range(len(bbox)):
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(bbox[i])
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        viewer.add_geometry(line_set)
    viewer.run()
    viewer.destroy_window()

def visulize_with_2_bbox(points3D, bbox1, bbox2, colors=None):

    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points3D)
    # if colors is not None:
    #     pcd.colors = o3d.utility.Vector3dVector(colors)
    # else:
    #     pcd.colors = o3d.utility.Vector3dVector(np.zeros_like(points3D))    
    pcd.paint_uniform_color(get_gray())
    estimate_normal(pcd,0.003,50)
    viewer.add_geometry(pcd)
    lines = [[0, 1], [1, 2], [2, 3], [0, 3],
             [4, 5], [5, 6], [6, 7], [4, 7],
             [0, 4], [1, 5], [2, 6], [3, 7]]

    # Use the same color for all lines
    colors = [[1, 0, 0] for _ in range(len(lines))]
    for i in range(len(bbox1)):
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(bbox1[i])
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        viewer.add_geometry(line_set)
    colors = [[0, 0, 1] for _ in range(len(lines))]
    for i in range(len(bbox2)):
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(bbox2[i])
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        viewer.add_geometry(line_set)
    viewer.run()
    viewer.destroy_window()

def multi_vis_6(pcds, names, render = True, width = 960, height = 540, shift = 100,add_frame = True):
    def add_geometry(vis, pcds, mesh_frame):
        for ele in pcds:
            vis.add_geometry(ele)
        if add_frame:
            vis.add_geometry(mesh_frame)
    
    def update_geometry(vis,pcds, mesh_frame):
        for ele in pcds:
            vis.update_geometry(ele)
        if add_frame:
            vis.update_geometry(mesh_frame)

    """
    Visulise point clouds in multiple windows, we allow at most 4 windows

    Input:
        pcds:   a list of pcds
        names:  a list of window names
    """
    import pyautogui
    width, height = pyautogui.size()
    width = (width - 100) // 2
    height = (height - 100) // 2

    assert len(pcds) == len(names)
    n_windows = len(pcds)

    window_corners = [
        [0,0],
        [0,height+shift],
        [width+shift, 0],
        [width+shift, height+shift]
    ]
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])

    # estimate normals for better visualisation
    if(render):
        for each_pcd in pcds:
            for ele in each_pcd:
                estimate_normal(ele, 0.3, 50)

    # initialise the windows
    vis_1 = o3d.visualization.Visualizer()
    vis_1.create_window(window_name=names[0], width=width, height=height, left=window_corners[0][0], top=window_corners[0][1])
    add_geometry(vis_1, pcds[0], mesh_frame)

    if(n_windows >= 2):
        vis_2 = o3d.visualization.Visualizer()
        vis_2.create_window(window_name=names[1], width=width, height=height, left=window_corners[1][0], top=window_corners[1][1])
        add_geometry(vis_2, pcds[1], mesh_frame)

        if(n_windows >=3):
            vis_3 = o3d.visualization.Visualizer()
            vis_3.create_window(window_name=names[2], width=width, height=height, left=window_corners[2][0], top=window_corners[2][1])
            add_geometry(vis_3, pcds[2], mesh_frame)

            if(n_windows>=4):
                vis_4 = o3d.visualization.Visualizer()
                vis_4.create_window(window_name=names[3], width=width, height=height, left=window_corners[3][0], top=window_corners[3][1])
                add_geometry(vis_4, pcds[3], mesh_frame)
                if(n_windows >=5):
                    vis_5 = o3d.visualization.Visualizer()
                    vis_5.create_window(window_name=names[4], width=width, height=height, left=window_corners[2][0], top=window_corners[2][1])
                    add_geometry(vis_5, pcds[4], mesh_frame)

                    if(n_windows>=6):
                        vis_6 = o3d.visualization.Visualizer()
                        vis_6.create_window(window_name=names[5], width=width, height=height, left=window_corners[3][0], top=window_corners[3][1])
                        add_geometry(vis_6, pcds[5], mesh_frame)

    # start rendering
    while True:
        update_geometry(vis_1, pcds[0], mesh_frame)
        if not vis_1.poll_events():
            break
        vis_1.update_renderer()

        if(n_windows>=2):
            update_geometry(vis_2, pcds[1], mesh_frame)
            if not vis_2.poll_events():
                break
            vis_2.update_renderer()

            cam = vis_1.get_view_control().convert_to_pinhole_camera_parameters()
            cam2 = vis_2.get_view_control().convert_to_pinhole_camera_parameters()
            cam2.extrinsic = cam.extrinsic
            vis_2.get_view_control().convert_from_pinhole_camera_parameters(cam2)


            if(n_windows>=3):
                update_geometry(vis_3, pcds[2], mesh_frame)
                if not vis_3.poll_events():
                    break
                vis_3.update_renderer()

                cam3 = vis_3.get_view_control().convert_to_pinhole_camera_parameters()
                cam3.extrinsic = cam.extrinsic
                vis_3.get_view_control().convert_from_pinhole_camera_parameters(cam3)

                if(n_windows>=4):
                    update_geometry(vis_4, pcds[3], mesh_frame)
                    if not vis_4.poll_events():
                        break
                    vis_4.update_renderer()

                    cam4 = vis_4.get_view_control().convert_to_pinhole_camera_parameters()
                    cam4.extrinsic = cam.extrinsic
                    vis_4.get_view_control().convert_from_pinhole_camera_parameters(cam4)
                    
                    if(n_windows>=5):
                        update_geometry(vis_5, pcds[4], mesh_frame)
                        if not vis_5.poll_events():
                            break
                        vis_5.update_renderer()

                        cam5 = vis_5.get_view_control().convert_to_pinhole_camera_parameters()
                        cam5.extrinsic = cam.extrinsic
                        vis_5.get_view_control().convert_from_pinhole_camera_parameters(cam5)

                        if(n_windows>=6):
                            update_geometry(vis_6, pcds[5], mesh_frame)
                            if not vis_6.poll_events():
                                break
                            vis_6.update_renderer()

                            cam6 = vis_6.get_view_control().convert_to_pinhole_camera_parameters()
                            cam6.extrinsic = cam.extrinsic
                            vis_6.get_view_control().convert_from_pinhole_camera_parameters(cam6)
    
    vis_1.destroy_window()
    if(n_windows>=2):
        vis_2.destroy_window()

        if(n_windows>=3):
            vis_3.destroy_window()

            if(n_windows>=4):
                vis_4.destroy_window()
                
                if(n_windows>=5):
                    vis_5.destroy_window()

                    if(n_windows>=6):
                        vis_6.destroy_window()