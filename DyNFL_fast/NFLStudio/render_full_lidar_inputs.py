import matplotlib.pyplot as plt
import numpy as np
from NFLStudio.libs.utils import to_o3d_pcd, get_blue, get_yellow, vis_o3d, multi_vis, natural_key, get_gray,get_red,get_blue,get_yellow, to_array, to_o3d_vec, makedirs
import open3d as o3d
from glob import glob
from tqdm import tqdm
from NFLStudio.libs.waymo import load_pc_dat, parse_waymo_data
from NFLStudio.libs.sim_utils import voxel_downsample
from NFLStudio.libs.render_results import render_pcd
import torch
import sys

my_cmap = plt.cm.get_cmap('tab10')
my_cmap = my_cmap(np.arange(5))[:,:3]
color_first_return = my_cmap[0]
color_second_return = my_cmap[1]


def color_intensity(intensity):
    min_intensity = 0
    max_intensity = 0.25
    CMAP='coolwarm'
    cmap = plt.get_cmap(CMAP)
    cmap = cmap(np.arange(256))[:,:3] 
    n_colors = cmap.shape[0] -1 
    intensity = np.clip(intensity, 0, max_intensity)
    color_idx = np.floor((intensity - min_intensity) / (max_intensity - min_intensity) * n_colors).astype(int)
    color = cmap[color_idx]
    return color

if __name__=='__main__':
    

    context_name = '1005081002024129653_5313_150_5333_150' #choose to diplay context name
    dir = '/path/to/pcd_out' #choose saved dir
    c_data = torch.load(f'{dir}/{context_name}/batch_full.pt')
    outputs_full = torch.load(f'{dir}/{context_name}/outputs_full_active_intensity_raydrop_60000.pt')
    first_dist_vol = outputs_full['depth_vol_c'] # resimulated distance
    intensity = outputs_full['intensity'] # resimulated intensity

    # load GT data
    rays_o, rays_d = c_data['rays_o'],c_data['rays_d']
    first_dist_gt = c_data['first_dist']
    first_mask = torch.logical_or(c_data['static_mask'], c_data['vehicle_mask']) 
    static_mask = c_data['static_mask'] 
    first_intensity = c_data['first_intensity']
    vehicle_mask = c_data['vehicle_mask']
    static_vehicle_mask = c_data['static_vehicle_mask']

        
    # compute gt LiDAR points
    points_gt = rays_o + rays_d * first_dist_gt[:,None]
    # compute resimulated LiDAR points
    points_vol = rays_o + rays_d * first_dist_vol[:,None]

    
    i=0 # select the index of LiDAR scans to display from 0 to 10
    sel = torch.logical_or(static_mask, vehicle_mask)[2650*64*i:2650*64*(i+1)] # ground truth mask

    # compute ray drop mask
    ray_drop_prob = outputs_full['ray_drop_prob']
    ray_hit_mask = ray_drop_prob[:,1] < ray_drop_prob[:,0] # 1 is drop 0 is hit
    pred_vehicle_mask = outputs_full['predicted_vehicle_mask']
    sel_pred = torch.logical_or(ray_hit_mask,pred_vehicle_mask)[2650*64*i:2650*64*(i+1)] 
    
    pcd_gt = points_gt[2650*64*i:2650*64*(i+1)][sel]
    pcd_vol_raydrop= points_vol[2650*64*i:2650*64*(i+1)][sel_pred]
        
    mean = pcd_gt.mean(0)[None]
    
    # subtract the mean
    pcd_gt -= mean
    pcd_vol_raydrop -=mean
    
        
    
    pcd_gt_o3d = to_o3d_pcd(pcd_gt.numpy())
    pcd_vol_raydrop = to_o3d_pcd(pcd_vol_raydrop.numpy())

    # paint LiDAR points based on intensity colors
    pcd_gt_o3d.colors = to_o3d_vec(color_intensity((first_intensity[2650*64*i:2650*64*(i+1)][sel]).numpy()))
    pcd_vol_raydrop.colors = to_o3d_vec(color_intensity(intensity[2650*64*i:2650*64*(i+1)][sel_pred].numpy()))

    # visulize gt and resimulation LiDAR scans
    multi_vis([[pcd_gt_o3d], [pcd_vol_raydrop]], ['gt', 'raydrop_pred_mask'],add_frame=True)
