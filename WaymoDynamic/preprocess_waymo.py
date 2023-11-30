#@title Initial setup
from typing import Optional
import numpy as np
import warnings
# Disable annoying warnings from PyArrow using under the hood.
warnings.simplefilter(action='ignore', category=FutureWarning)


import tensorflow as tf
import dask.dataframe as dd
from waymo_open_dataset import v2
from waymo_open_dataset.v2.perception.utils import lidar_utils
# from bind.build_my import waymo_data_preprocess
from libs.box_np_ops import points_in_rbbox
from libs.register_utils import kabsch_transformation_estimation, convert_rot_trans_to_tsfm, apply_tsfm
from libs.bbox_utils import center_to_corner_box3d
from libs.utils import to_tensor, to_array
import os
from pathlib import Path
import point_cloud_utils as pcu
import open3d as o3d
# Path to the directory with all components
dataset_dir = './waymo/training'

context_name = '1005081002024129653_5313_150_5333_150' 
context_name = '1083056852838271990_4080_000_4100_000' 
context_name = '13271285919570645382_5320_000_5340_000' 
context_name = '10072140764565668044_4060_000_4080_000' 
context_name = '10500357041547037089_1474_800_1494_800' 

def read(tag: str) -> dd.DataFrame:
  """Creates a Dask DataFrame for the component specified by its tag."""
  paths = tf.io.gfile.glob(f'{dataset_dir}/{tag}/{context_name}.parquet')
  return dd.read_parquet(paths)

def visulize3D(points3D ,colors=None):
    """Visualize the 3d lidar points"""
    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points3D)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    viewer.add_geometry(pcd)
    viewer.run()
    viewer.destroy_window()

def visulize_with_bbox(points3D, bbox, colors=None):

    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points3D)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        pcd.colors = o3d.utility.Vector3dVector(np.zeros_like(points3D))    
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

# def normal_estimate(points3D):
#     """Compute the normal for every point in the Points3D
#     Param Points3D: np.array (N, 3)
#     """
#     # colors = colors.astype(np.float64)/255
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points3D)
#     normal_bool = pcd.estimate_normals(
#         search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2,
#                                                           max_nn=30)
#     )
#     print(normal_bool)
#     # pcd.colors = o3d.utility.Vector3dVector(colors)
#     # o3d.visualization.draw_geometries([pcd])
#     # o3d.visualization.draw_geometries([pcd],point_show_normal=True)
#     return np.asarray(pcd.normals)

def from_euler(roll, pitch, yaw):
    """Compute the transform matrix from roll pitch yaw
    Param: roll np.array (H,W,1)
    Param: pitch np.array (H,W,1)
    Param: yaw np.array (H,W,1)

    Return: transform matrix np.array(H,W,4,4)
    """
    zero = np.zeros_like(roll)
    one = np.ones_like(roll)
    yaw_matrix = np.concatenate(
        [
            np.concatenate([np.cos(yaw),-np.sin(yaw),zero], axis=2)[:,:,None,:],
            np.concatenate([np.sin(yaw), np.cos(yaw),zero], axis=2)[:,:,None,:],
            np.concatenate([zero, zero, one], axis=2)[:,:,None,:],
        ],
        axis=2
    )
    pitch_matrix = np.concatenate(
        [
            np.concatenate([np.cos(pitch),zero,np.sin(pitch)], axis=2)[:,:,None,:],
            np.concatenate([zero,one,zero], axis=2)[:,:,None,:],
            np.concatenate([-np.sin(pitch), zero, np.cos(pitch)], axis=2)[:,:,None,:],
        ],
        axis=2
    )
    roll_matrix = np.concatenate(
        [
            np.concatenate([one,zero,zero], axis=2)[:,:,None,:],
            np.concatenate([zero, np.cos(roll),-np.sin(roll)], axis=2)[:,:,None,:],
            np.concatenate([zero, np.sin(roll), np.cos(roll)], axis=2)[:,:,None,:],
        ],
        axis=2
    )
    # result = np.einsum("...ab, ...bd -> ...ad", roll_matrix, pitch_matrix)
    # result = np.einsum("...ab, ...bd -> ...ad", result, yaw_matrix)
    result = np.einsum("...ab, ...bd -> ...ad", yaw_matrix, pitch_matrix)
    result = np.einsum("...ab, ...bd -> ...ad", result, roll_matrix)
    return result #(h,w,3,3)


def interpolate_ray_origin(origin_ray_world, mask):
    """Interpolate the ray origins vertically for the ray drop places
    Param: origin_ray_world np.array((H,W,3), dtype=float64)
    Param: mask np.array((H,W), dtype=bool) False indicates ray drop
    Return: origin_ray_world_interpolated np.array((H,W,3), dtype=float64) interpolated ray_origin
    """

    count = np.where(mask, 1, 0)
    count = np.sum(count,axis=0)
    origin_ray_world_copy = np.copy(origin_ray_world)
    origin_ray_world_copy[~mask] = np.array([0,0,0])
    origin_ray_world_sum = np.sum(origin_ray_world_copy,axis=0)
    origin_ray_world_mean = origin_ray_world_sum/count[:,None]
    origin_ray_world_drop = np.copy(origin_ray_world)
    origin_ray_world_drop[mask] = np.array([0,0,0])
    origin_ray_world_drop[~mask] = np.array([1,1,1])
    origin_ray_world_drop *= origin_ray_world_mean[None,:,:]
    origin_ray_world_interpolated = origin_ray_world_copy + origin_ray_world_drop
    return origin_ray_world_interpolated



def interpolate_ray_dir(ray_dir_world, inclination, mask):
    """Interpolate the ray directions vertically for the ray drop places
    Param: ray_dir_world np.array((H,W,3), dtype=float64)
    Param: inclination np.array (H,) for each row
    Param: mask np.array((H,W), dtype=bool) False indicates ray drop
    Return: ray_dir_world np.array((H,W,3), dtype=float64) interpolated ray_direction
    """
    from sklearn.linear_model import LinearRegression
    for i in range(ray_dir_world.shape[1]):
        if(len(inclination[~mask[:,i]]) == 0):continue
        ray_dir_world[:,i,:][~mask[:,i]] = np.array([0,0,0])
        X = inclination[mask[:,i]][:,None]
        y = ray_dir_world[:,i,:][mask[:,i]]
        reg = LinearRegression().fit(X,y)
        result = reg.predict(inclination[~(mask[:,i])][:,None])
        ray_dir_world[:,i,:][~mask[:,i]] = result/np.sqrt(np.sum(result*result, axis=1))[:,None]
    return ray_dir_world


def apply_tsfm(src, tsfm):
    """
    tsfm:   [4,4]
    src:    [N,3]
    """
    R, t = tsfm[:3,:3], tsfm[:3,3][:,None]
    src = (R @ src.T + t).T
    return src


lidar_df = read('lidar')
lidar_pose_df = read('lidar_pose')
lidar_calibration_df = read('lidar_calibration')
vehicle_pose_df = read('vehicle_pose')


lidar_lidar_pose_df = v2.merge(lidar_df, lidar_pose_df)
lidar_pose_lidar_transform = v2.merge(lidar_lidar_pose_df, lidar_calibration_df, key_prefix='key.laser_name')
lidar_vehicle_df = v2.merge(lidar_pose_lidar_transform, vehicle_pose_df, key_prefix='key.frame_timestamp_micros')


##group lidar boxes from one frame
lidar_box_df = read("lidar_box")
lidar_vehicle_df = v2.merge(lidar_vehicle_df, lidar_box_df, right_group=True)


first = True
p3d_tmp = []
objects_id_2_tsfm = {} #dict of object ids to a list of transformations wrt the anchor corner
objects_id_types_per_frame = []#for each frame stores the indices to object types
objects_id_2_corners = {} #for each object stores the global corners in each occurring frame
objects_id_2_anchors = {} #for each object stores the global corners in its anchor box
objects_id_2_frameidx = {}# dict of object ids to a list of occuring frame indices
objects_id_2_dynamic_flag = {}# dict of object ids to a list of dynamic_flags False is static True is dynamic
object_ids_per_frame = []# for each frame stores the indices to object ids
masks_for_frames = []
ray_origins = []
ray_dirs = []
range_images1 = []
range_images2 = []
ray_object_indices = []

context_root_dir = Path('./processed_data_dynamic/'+context_name)
try:
    os.mkdir('./processed_data_dynamic/')
except FileExistsError:
    pass
try:
    os.mkdir(context_root_dir)
except FileExistsError:
    pass
frame_index = 0
frame=0
for _,row in lidar_vehicle_df.iterrows():
    if frame<50:
        frame += 1
        continue
    lidar = v2.LiDARComponent.from_dict(row)

    range_image_return1 = lidar.range_image_return1.values.reshape(lidar.range_image_return1.shape)
    range_image_return2 = lidar.range_image_return2.values.reshape(lidar.range_image_return2.shape)
    num_row = range_image_return1.shape[0]
    num_column = range_image_return1.shape[1]
    lidar_pose = v2.LiDARPoseComponent.from_dict(row)

    rpy_xyz = lidar_pose.range_image_return1
    rpy_xyz = rpy_xyz.values.reshape(rpy_xyz.shape)
    roll_angle = rpy_xyz[:,:,0][:,:,None]
    pitch_angle = rpy_xyz[:,:,1][:,:,None]
    yaw_angle = rpy_xyz[:,:,2][:,:,None]
    rotation_pixel_vehicle_frame = from_euler(roll_angle, pitch_angle, yaw_angle)
    translation_pixel_vehicle_frame = rpy_xyz[:,:,3:6]

    transform_lidar_pixel_vehicle_world = np.concatenate([rotation_pixel_vehicle_frame, translation_pixel_vehicle_frame[:,:,:,None]],axis=3)

    amendment = np.array([[[0,0,0,1]]])
    amendment = np.repeat(amendment, num_row, axis=0)
    amendment = np.repeat(amendment, num_column, axis=1)
    transform_lidar_pixel_vehicle_world = np.concatenate([transform_lidar_pixel_vehicle_world, amendment[:,:,None,:]], axis=2)


    lidar_calibration = v2.LiDARCalibrationComponent.from_dict(row)
    vehicle_pose = v2.VehiclePoseComponent.from_dict(row)
    vehicle_frame_pose = vehicle_pose.world_from_vehicle

     
    transform_laser_vehicle = lidar_calibration.extrinsic.transform.reshape(4,4)
    transform_vehicle_world = vehicle_frame_pose.transform.reshape(4,4)
    inclination = lidar_calibration.beam_inclination.values
    if inclination is None: exit('wtf')
    inclination = np.flip(inclination ,axis=-1)
    width = range_image_return1.shape[1]
    az_correction = np.arctan2(transform_laser_vehicle[1,0], transform_laser_vehicle[0,0])
    ratios = (np.arange(width, 0, -1) - 0.5) / width
    azimuth = (ratios * 2 - 1)*np.pi - az_correction
    
    
    cos_azimuth = np.cos(azimuth)
    cos_azimuth = np.repeat(cos_azimuth[None,:], num_row,axis=0)
    sin_azimuth = np.sin(azimuth)
    sin_azimuth = np.repeat(sin_azimuth[None,:], num_row,axis=0)

    cos_incl = np.cos(inclination)
    cos_incl = np.repeat(cos_incl[:,None], num_column, axis=1)
    sin_incl = np.sin(inclination)
    sin_incl = np.repeat(sin_incl[:,None], num_column, axis=1)
    #####ray drop mask 1 is not drop 0 is drop (h,w,1)
    mask = np.where(range_image_return1[:,:,0]>=0, True, False)

    range_images1.append(range_image_return1[None,:,:,:])
    range_images2.append(range_image_return2[None,:,:,:])

    masks_for_frames.append(mask)
    mask2 = np.where(range_image_return2[:,:,0]>=0, True, False)

    #####compute the ray origin in world frame origin_world (h,w,3)
    origin_laser_frame = np.zeros((num_row, num_column, 4))
    origin_laser_frame[:,:,3] = 1
    origin_vehicle_frame = np.einsum("ij,...j -> ...i", transform_laser_vehicle, origin_laser_frame)
    origin_ray_world = np.einsum("...ij,...j -> ...i", transform_lidar_pixel_vehicle_world, origin_vehicle_frame)[:,:,:3]
    origin_ray_world = interpolate_ray_origin(origin_ray_world, mask)
    ray_origins.append(origin_ray_world[None,:,:,:])
    #####

    #####Compute the ray direction in world frame ray_dir_world(h,w,3)
    # Note that, in order to take into the rolling shutter effect into account, we can't use one transforamtion matrix to 
    # map the ray origin or ray direction from vehicle frame to world frame, we have to use pixel level distinct transformation
    # to map each ray_origin/ray_dir represented by each pixel to the world frame. This is stored in lidar_pose
    ray_dir_laser_frame = np.concatenate([(cos_azimuth * cos_incl)[:,:,None], (sin_azimuth * cos_incl)[:,:,None], sin_incl[:,:,None], np.ones((num_row, num_column, 1))], axis=2)
    ray_dir_laser_frame[:,:,:3] *= range_image_return1[:,:,0][:,:,None] ## pcd in laser frame
    ray_dir_vehicle_frame = np.einsum("ij,...j -> ...i", transform_laser_vehicle , ray_dir_laser_frame)
    ray_dir_world = np.einsum("...ij,...j -> ...i", transform_lidar_pixel_vehicle_world , ray_dir_vehicle_frame)
     
    pcd_world = ray_dir_world

    ray_dir_world = ray_dir_world[:,:,:3] - origin_ray_world
    ray_dir_world /= np.sqrt(np.sum(ray_dir_world * ray_dir_world, axis=2))[:,:,None]
    ray_dir_world = interpolate_ray_dir(ray_dir_world, inclination, mask)
    ray_dirs.append(ray_dir_world[None,:,:,:])
    #####

    p3d_world = origin_ray_world + ray_dir_world*range_image_return1[:,:,0][:,:,None]
    p3d_world = p3d_world[mask]
    p3d_masked = p3d_world.reshape(-1,3)
    color = np.zeros_like(p3d_masked)
    p3d_masked = np.concatenate([p3d_masked, np.ones((p3d_masked.shape[0],1))], axis=1) 
    p3d_masked = p3d_masked[:,:3]
    p3d_tmp.append(p3d_masked)#store the pointclouds of each frame for later norm estimation
    # visulize3D(p3d_masked)


    ###lidar boxes
    points_frame = np.einsum("ij, ...j -> ...i", np.linalg.inv(transform_vehicle_world), pcd_world)[:,:,:3]
    points_frame = points_frame[mask].reshape(-1,3)

    lidar_box = v2.LiDARBoxComponent.from_dict(row)
    # print(len(lidar_box.key.laser_object_id))
    bboxes = []
    names = [] ##object ids
    labels = [] ## object types
    center = lidar_box.box.center
    size = lidar_box.box.size
    speed = lidar_box.speed
    yaw_angle = lidar_box.box.heading
     
    pose_rot = transform_vehicle_world[:3, :3]

    laser_indice = np.zeros(points_frame.shape[0])

    speed_np_frame = np.concatenate(
        [
            np.array(speed.x)[:,None], np.array(speed.y)[:,None], np.array(speed.z)[:,None]
        ],
        axis=1
    )
    speed_np_world = np.einsum("ij, ...j -> ...i", pose_rot, speed_np_frame).T ##N*3
    speed_np_world_norm = np.linalg.norm(speed_np_world, axis=0)
    center_np_frame = np.concatenate(
        [
            np.array(center.x)[:,None], np.array(center.y)[:,None], np.array(center.z)[:,None],
        ],
        axis=1
    )
    center_np_world = np.einsum("ij, ...j -> ...i", pose_rot, center_np_frame).T ##N*3

    bboxes = np.concatenate([
        center_np_frame,
        np.array(size.x)[:,None], np.array(size.y)[:,None], np.array(size.z)[:,None],
        speed_np_frame,
        np.array(yaw_angle)[:,None]
    ],axis=1)

    
    #################### compute each lidar point belong to which object type, -1 is background 
    if len(bboxes) > 0:
        indices = points_in_rbbox(points_frame, bboxes).astype(np.int)  
        indices = np.hstack([np.ones((indices.shape[0],1)) * 0.5,indices])
        ind_bbox = indices.argmax(1) 
        ind_bbox -= 1  
        ind_bbox = ind_bbox.astype(np.int)
        assert ind_bbox.min() == -1
    else:
        ind_bbox = np.zero_like(laser_indice) - 1    
         

    laser_indice = laser_indice[:,None]
    ind_bbox = ind_bbox[:,None].astype(int)


    assert  ind_bbox.shape[0] == laser_indice.shape[0] == points_frame.shape[0]
    laser_data = np.hstack([points_frame, laser_indice, ind_bbox])

    
    
    ########################################## Uncomment to see the segmentation of the lidar points
    # color = np.zeros_like(p3d_masked)
    # color += 1
    # color[:,2]=0
    # color_mask = np.where(ind_bbox<0, 0, 1)
    # color *= color_mask
    # visulize3D(p3d_masked, color)
    ##########################################

    time_indice = frame_index
    object_ids_per_frame.append(lidar_box.key.laser_object_id + ['background'])
    
    ids_per_ray = np.zeros((num_row, num_column)) - 1
    ids_per_ray[mask] = ind_bbox.squeeze(1)
    ray_object_indices.append(ids_per_ray[None,:,:].astype(int))
    objects_id_types_per_frame.append(np.array(lidar_box.type))

    
    ###############################################
    ## In each frame we capture a set of objects with its bounding boxes
    ## We create several dictionaries map the object id to its bounding boxes,
    ## occurences as well as the transformation wrt. the anchor frame 
    corners = center_to_corner_box3d(bboxes[:,:3], bboxes[:,3:6], -bboxes[:,-1])#(N, 8, 3) N is the number of objects in the frame
    for i, object_id in enumerate(lidar_box.key.laser_object_id):
        velocity_flag = speed_np_world_norm[i] > 1.0
        if object_id in objects_id_2_tsfm:##occured before
            anchor_corners = objects_id_2_anchors[object_id]
            curr_corners = corners[i]
            curr_corners = apply_tsfm(curr_corners, transform_vehicle_world)
            rotation_matrix, translation_matrix, res, _ = kabsch_transformation_estimation(to_tensor(curr_corners).float()[None],to_tensor(anchor_corners).float()[None])
            curr_tsfm = convert_rot_trans_to_tsfm(to_array(rotation_matrix[0]), to_array(translation_matrix[0]))

            objects_id_2_tsfm[object_id].append(curr_tsfm) ## record the transformation
            objects_id_2_corners[object_id].append(curr_corners) ## record the object bounding box (8,3) of the current frame
            objects_id_2_frameidx[object_id].append(frame_index) ## record the frame idx when the object occurred
            objects_id_2_dynamic_flag[object_id] = objects_id_2_dynamic_flag[object_id] or velocity_flag

        else:##object never occured in previous frame
            first_corners = corners[i]
            first_corners = apply_tsfm(first_corners, transform_vehicle_world)
            x = np.linalg.norm(first_corners[0,:] - first_corners[4,:], axis=-1)
            y = np.linalg.norm(first_corners[0,:] - first_corners[3,:], axis=-1)
            z = np.linalg.norm(first_corners[0,:] - first_corners[1,:], axis=-1)
            # print(x,y,z)
            anchor_corners = np.array([[0,0,0], [0,0,z],[0,y,z], [0,y,0],[x,0,0],[x,0,z],[x,y,z],[x,y,0]]) + np.mean(first_corners, axis=0)
            ##transform to aabb corners
            rotation_matrix, translation_matrix, res, _  = kabsch_transformation_estimation(to_tensor(first_corners).float()[None], to_tensor(anchor_corners).float()[None])
            curr_tsfm = convert_rot_trans_to_tsfm(to_array(rotation_matrix[0]), to_array(translation_matrix[0]))
            objects_id_2_tsfm.update({object_id: [curr_tsfm]})
            objects_id_2_corners.update({object_id: [first_corners]})
            objects_id_2_anchors.update({object_id: anchor_corners})
            objects_id_2_frameidx.update({object_id: [frame_index]})
            objects_id_2_dynamic_flag[object_id] = velocity_flag

        
    if sum([1 if speed_np_world_norm[i] > 1.0 else 0 for i in range(len(corners)) ]) >= 1 :
        # ############################################## Uncomment to visulize the bounding boxes
        # corners_world = []
        # for i in range(len(corners)):
            # corner_world = apply_tsfm(corners[i], transform_vehicle_world)
            # corners_world.append(corner_world[None,:,:])
        # corners_world = np.concatenate(corners_world, axis=0)            
        # visulize_with_bbox(p3d_masked, corners_world)
        # ###############################################

        ############################################## Uncomment to visulize the dynamic bounding boxes
        # corners_world = []
        # for i in range(len(corners)):
        #     if speed_np_world_norm[i] > 1.0:
        #         corner_world = apply_tsfm(corners[i], transform_vehicle_world)
        #         corners_world.append(corner_world[None,:,:])
        #         print(f'dynamic object type: {objects_id_types_per_frame[frame_index][i]}, speed: {speed_np_world_norm[i]}')
        # if len(corners_world) > 0:   
        #     print(f'detect dynamic object in the frame')
        #     corners_world = np.concatenate(corners_world, axis=0)            
        #     #visualize the anchor bbox
        #     for object_id in objects_id_2_dynamic_flag.keys():
        #         if objects_id_2_dynamic_flag[object_id]:
        #             corners_world = np.concatenate([corners_world, objects_id_2_anchors[object_id][None,:,:]], axis=0)
        #             tsfm = objects_id_2_tsfm[object_id][-1]
        #             print(np.einsum("ij, bj -> bi", tsfm[:3,:3], objects_id_2_corners[object_id][-1]) + tsfm[:3,3]  - objects_id_2_anchors[object_id])

        #     visulize_with_bbox(p3d_masked, corners_world)
        # else: print('no dynamic object in the frame')    
        ###############################################
        pass

    print(frame_index)
    frame_index += 1
    if(frame_index == 50): break

range_images1 = np.concatenate(range_images1, axis=0)    
range_images2 = np.concatenate(range_images2, axis=0)     
ray_origins = np.concatenate(ray_origins, axis=0)    
ray_dirs = np.concatenate(ray_dirs, axis=0)    
ray_object_indices = np.concatenate(ray_object_indices, axis=0)    

np.save(context_root_dir/'range_images1.npy', range_images1, allow_pickle=True)
np.save(context_root_dir/'range_images2.npy', range_images2, allow_pickle=True)
np.save(context_root_dir/'ray_origins.npy', ray_origins, allow_pickle=True)
np.save(context_root_dir/'ray_dirs.npy', ray_dirs, allow_pickle=True)
np.save(context_root_dir/'ray_object_indices.npy', ray_object_indices, allow_pickle=True)



np.save(context_root_dir/'objects_id_2_tsfm.npy', objects_id_2_tsfm, allow_pickle=True)
np.save(context_root_dir/'objects_id_types_per_frame.npy', objects_id_types_per_frame, allow_pickle=True)
np.save(context_root_dir/'objects_id_2_corners.npy', objects_id_2_corners, allow_pickle=True)
np.save(context_root_dir/'objects_id_2_anchors.npy', objects_id_2_anchors, allow_pickle=True)
np.save(context_root_dir/'objects_id_2_frameidx.npy', objects_id_2_frameidx, allow_pickle=True)
np.save(context_root_dir/'objects_id_2_dynamic_flag.npy', objects_id_2_dynamic_flag, allow_pickle=True)
np.save(context_root_dir/'object_ids_per_frame.npy', object_ids_per_frame, allow_pickle=True)


p3d_all = np.concatenate(p3d_tmp, axis=0)
normals_all = np.zeros_like(p3d_all)
valid_normal_mask_all = np.zeros(p3d_all.shape[0]).astype(np.bool)
# ind, normals_estimated = pcu.estimate_point_cloud_normals_ball(p3d_all, 0.2)
ind, normals_estimated = pcu.estimate_point_cloud_normals_knn(p3d_all, 30)
normals_all[ind] = normals_estimated
valid_normal_mask_all[ind]=1


start_index = 0
normals = []

valid_normal_flags = []
for index in range(len(p3d_tmp)):
    normals_frame_full = np.zeros_like(p3d_tmp[index])
    normal_pcd_i = normals_all[start_index : start_index+p3d_tmp[index].shape[0]]
    valid_normal_i = valid_normal_mask_all[start_index : start_index+p3d_tmp[index].shape[0]].astype(np.bool)
    start_index += p3d_tmp[index].shape[0]
    
    valid_normal_flag = np.zeros_like(masks_for_frames[index]).astype(np.bool)

    tmp_flag = np.zeros_like(masks_for_frames[index]).astype(np.bool)
    tmp_flag = tmp_flag[masks_for_frames[index]]
    tmp_flag[valid_normal_i] = True
    valid_normal_flag[masks_for_frames[index]] = tmp_flag
    valid_normal_flags.append(valid_normal_flag[None,:,:])
    normals_map = np.zeros_like(ray_dir_world)
    normals_map[masks_for_frames[index]] = normal_pcd_i
    normals.append(normals_map[None,:,:,:])
normals = np.concatenate(normals, axis=0)    
np.save(context_root_dir/'normals.npy', normals, allow_pickle=True)
valid_normal_flags = np.concatenate(valid_normal_flags, axis=0)    
np.save(context_root_dir/'valid_normal_flags.npy', valid_normal_flags, allow_pickle=True)