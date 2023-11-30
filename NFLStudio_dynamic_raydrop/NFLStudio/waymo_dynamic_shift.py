import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from NFLStudio.libs.utils import interpolate_rigid_transformation, visulize_with_2_bbox, visulize_with_bbox
class WaymoDynamic(Dataset):
    center: None
    bound_z: 1.0
    extent: None

    def __init__(self, split, config, device=None, center=None, extent=None, bound_z=None, dtype=np.float32):
        self.scene_size = 50
        self.center = None
        self.bound_z = None
        self.extent = None
        self.split = split
        self.device = device
        self.context_name = config.context_name
        self.context_dir = Path(config.root_dir)/self.context_name
        self.margin = config.extra_margin

        self.dtype = dtype

        self.splix_idx = np.arange(self.scene_size)
        sample_every = 5
        if split == 'val':
            self.splix_idx = self.splix_idx[sample_every-1::sample_every]
        elif split == 'test':
            # self.splix_idx = self.splix_idx[sample_every-1::sample_every]
            pass
        elif split == 'train':
            self.splix_idx = np.array([idx for idx in self.splix_idx if idx not in self.splix_idx[sample_every-1::sample_every]])
            # self.splix_idx = self.splix_idx[sample_every-1::sample_every]

        range_images1 = np.load(self.context_dir/'range_images1.npy', allow_pickle=True).astype(dtype)[:self.scene_size,:,:]
        range_images2 = np.load(self.context_dir/'range_images2.npy', allow_pickle=True).astype(dtype)[:self.scene_size,:,:]
        self.frame_idx = torch.arange(self.scene_size)
        self.first_masks = np.where(range_images1[:,:,:,0]>=0, True, False)
        self.first_dist = range_images1[:,:,:,0]
        self.first_intensity = np.tanh(range_images1[:,:,:,1])
        self.first_elongation = range_images1[:,:,:,2]
        self.second_masks = np.where(range_images2[:,:,:,0]>=0, True, False)
        self.second_dist = range_images2[:,:,:,0]
        self.second_intensity = np.tanh(range_images2[:,:,:,1])
        self.second_elongation = range_images2[:,:,:,2]
        self.ray_object_indices = np.load(self.context_dir/'ray_object_indices.npy', allow_pickle=True)[:self.scene_size,:,:]
        self.normals = np.load(self.context_dir/'normals.npy', allow_pickle=True).astype(dtype)[:self.scene_size,:,:,:]
        self.ray_origins = np.load(self.context_dir/'ray_origins.npy', allow_pickle=True).astype(dtype)[:self.scene_size,:,:,:]
        self.ray_dirs = np.load(self.context_dir/'ray_dirs.npy', allow_pickle=True).astype(dtype)[:self.scene_size,:,:,:]
        self.valid_normal_flag = np.load(self.context_dir/'valid_normal_flags.npy', allow_pickle=True)[:self.scene_size,:,:]
        self.objects_id_2_tsfm = np.load(self.context_dir/'objects_id_2_tsfm.npy', allow_pickle=True).item()
        self.objects_id_types_per_frame = np.load(self.context_dir/'objects_id_types_per_frame.npy', allow_pickle=True)
        self.objects_id_2_corners = np.load(self.context_dir/'objects_id_2_corners.npy', allow_pickle=True).item()
        self.objects_id_2_anchors = np.load(self.context_dir/'objects_id_2_anchors.npy', allow_pickle=True).item()
        self.objects_id_2_frameidx = np.load(self.context_dir/'objects_id_2_frameidx.npy', allow_pickle=True).item()
        self.objects_id_2_dynamic_flag = np.load(self.context_dir/'objects_id_2_dynamic_flag.npy', allow_pickle=True).item()
        self.object_ids_per_frame = np.load(self.context_dir/'object_ids_per_frame.npy', allow_pickle=True)
        self.valid_first_return = np.where((self.second_dist - self.first_dist) > 1.75, True, False) 
        # if self.center is None and self.split=='train':
        self.get_spatial_extent(config.normalize)

        print("extent: ", self.extent)
        print("center: ", self.center)

        self.ray_origins = (self.ray_origins - self.center[None]) / self.extent
        self.first_dist = self.first_dist  / self.extent
        self.second_dist = self.second_dist  / self.extent
        



        self.map_id_2_type()
        self.convert_dynamic_object_id_2_global_idx()
        self.create_aabb_of_anchor_boxes_of_dynamic_vehicles()
        self.create_tsfm_for_object_idx_at_every_frame()
        self.interpolate_vehicle_tsfm_for_test_frames()

        # self.save_vehicle_data(13)
        # self.load_vehicle_data(to_replace_index=3, to_load_index=13, context_name_other='13271285919570645382_5320_000_5340_000')
        # self.visualize_traj()
        self.manipulate_sensor()
        self.manipulate_vehicle()


        

        print(f"split: {split}, split_idx: {self.splix_idx}, len: {self.__len__()}")

        self.float_args = ['rays_o', 'rays_d', 'rays_normal', 'first_dist', 'first_intensity', 'first_elongation', 'object_bbox', 'object_tsfm']

    def map_id_2_type(self):
        ## map a string id to int types
        self.object_id_2_type = {}
        for frame_idx in range(self.scene_size):
            for object_idx in range(len(self.objects_id_types_per_frame[frame_idx])):
                object_id = self.object_ids_per_frame[frame_idx][object_idx]
                object_type = self.objects_id_types_per_frame[frame_idx][object_idx]
                self.object_id_2_type.update({object_id:object_type})


    def convert_dynamic_object_id_2_global_idx(self):
        ###mapping string ids to int indices
        dynamic_object_counter=0
        object_id_2_global_idx = {}
        for frame_idx in range(self.scene_size):
            for object_id in self.object_ids_per_frame[frame_idx]:
                dynamic_flag = self.objects_id_2_dynamic_flag[object_id] if object_id in self.objects_id_2_dynamic_flag.keys() else False
                object_type = self.object_id_2_type[object_id] if object_id in self.object_id_2_type.keys() else -1
                if object_id not in object_id_2_global_idx.keys() and dynamic_flag and object_type==1:
                    object_id_2_global_idx.update({object_id:dynamic_object_counter})
                    dynamic_object_counter+=1
        self.object_id_2_global_idx = object_id_2_global_idx   
        self.dynamic_object_counter = dynamic_object_counter   
        print(f"detect {self.dynamic_object_counter} dynamic vehicles in the dataset")

    def create_aabb_of_anchor_boxes_of_dynamic_vehicles(self):
        self.aabb_vehicle = [torch.randn(6) for i in range(self.dynamic_object_counter)]
        for object_id in self.object_id_2_global_idx.keys():
            # anchor_bbox = self.objects_id_2_corners[object_id][0]
            anchor_bbox = self.objects_id_2_anchors[object_id]
            min = np.min(anchor_bbox,axis=0)
            max = np.max(anchor_bbox,axis=0)
            aabb = torch.tensor([min[0], min[1], min[2], max[0], max[1], max[2]])    
            self.aabb_vehicle[self.object_id_2_global_idx[object_id]] = aabb

    def create_tsfm_for_object_idx_at_every_frame(self):
        self.tsfm_vehicle = torch.randn(self.scene_size, self.dynamic_object_counter,4,4)
        self.mask_tsfm_vehicle = torch.zeros(self.scene_size, self.dynamic_object_counter).bool()
        for object_id in self.object_id_2_global_idx.keys():
            vehicle_idx = self.object_id_2_global_idx[object_id]
            occurred_frames = self.objects_id_2_frameidx[object_id]
            # print("occurred frames for vehicle ", vehicle_idx, " is: ", len(occurred_frames))
            tsfms = self.objects_id_2_tsfm[object_id]
            for i in range(len(occurred_frames)):
                occurred_frame_idx = occurred_frames[i]
                tsfm = tsfms[i]
                self.tsfm_vehicle[occurred_frame_idx, vehicle_idx,:4,:4] = torch.tensor(tsfm)
                self.mask_tsfm_vehicle[occurred_frame_idx, vehicle_idx] = True
        # print(self.tsfm_vehicle[:,2,:,:])

            
    def interpolate_vehicle_tsfm_for_test_frames(self):
        if self.split == 'train': return
        accum_fnorm = 0
        
        for object_idx in range(self.dynamic_object_counter):
            occurred_frames = torch.arange(self.scene_size)[self.mask_tsfm_vehicle[:,object_idx]]
            if len(occurred_frames) <= 2: continue
            for frame_idx in self.splix_idx:
                if frame_idx not in occurred_frames: continue
                occurred_idx = (occurred_frames==frame_idx).nonzero().squeeze()

                if frame_idx == occurred_frames[0]:
                    frameafter = occurred_frames[occurred_idx+1]
                    framesecondafter = occurred_frames[occurred_idx+2]
                    t = ((frame_idx-frameafter)/(framesecondafter - frameafter))[None]
                    tsfm_after = self.tsfm_vehicle[frameafter, object_idx][None]
                    tsfm_secondafter = self.tsfm_vehicle[framesecondafter, object_idx][None]
                    extrapolated = interpolate_rigid_transformation(tsfm_after, tsfm_secondafter, t).squeeze()
                    extrapolated[:3,:3] = tsfm_after[:,:3,:3]
                    diff= self.tsfm_vehicle[frame_idx, object_idx] - extrapolated
                    fnorm = torch.sqrt((diff * diff).sum())
                    accum_fnorm += fnorm
                    self.tsfm_vehicle[frame_idx, object_idx] = extrapolated
                    #extrapolate
                elif frame_idx == occurred_frames[-1]:
                    framebefore = occurred_frames[occurred_idx-1]
                    framesecondbefore = occurred_frames[occurred_idx-2]
                    t = ((frame_idx-framesecondbefore)/(framebefore - framesecondbefore))[None]
                    tsfm_before = self.tsfm_vehicle[framebefore, object_idx][None]
                    tsfm_secondbefore = self.tsfm_vehicle[framesecondbefore, object_idx][None]
                    extrapolated = interpolate_rigid_transformation(tsfm_secondbefore, tsfm_before, t).squeeze()
                    extrapolated[:3,:3] = tsfm_before[:,:3,:3]
                    diff= self.tsfm_vehicle[frame_idx, object_idx] - extrapolated
                    fnorm = torch.sqrt((diff * diff).sum())
                    accum_fnorm += fnorm
                    self.tsfm_vehicle[frame_idx, object_idx] = extrapolated
                    #extrapolate
                else:
                    framebefore = occurred_frames[occurred_idx-1]
                    frameafter = occurred_frames[occurred_idx+1]
                    t = ((frame_idx-framebefore)/(frameafter - framebefore))[None]
                    tsfm_before = self.tsfm_vehicle[framebefore, object_idx][None]
                    tsfm_after = self.tsfm_vehicle[frameafter, object_idx][None]
                    interpolated = interpolate_rigid_transformation(tsfm_before,tsfm_after,t).squeeze()
                    diff= self.tsfm_vehicle[frame_idx, object_idx] - interpolated
                    fnorm = torch.sqrt((diff * diff).sum())
                    if torch.isnan(fnorm):
                        print("nan")
                        interpolated = interpolate_rigid_transformation(tsfm_before,tsfm_after,t).squeeze()
                    accum_fnorm += fnorm
                    self.tsfm_vehicle[frame_idx, object_idx] = interpolated
                    #interpolate
        dynamic_mask = self.mask_tsfm_vehicle[self.splix_idx]
        count = dynamic_mask[dynamic_mask].shape[0]
        accum_fnorm /= count
        print("mean fnorm is: ", accum_fnorm)


    def visualize_traj(self):
        if self.split == 'train': return
        import open3d as o3d
        split_idx_np = np.array(self.splix_idx).astype(np.int64)
        points3D = (self.ray_origins[split_idx_np,:,:,:] + self.ray_dirs[split_idx_np,:,:,:]*self.first_dist[:,:,:,None][split_idx_np,:,:,:])[self.first_masks[split_idx_np,:,:]]
        bbox_ori =[]
        bbox_tsfm = []
        for object_id in self.object_id_2_global_idx.keys():
            for frame in self.splix_idx:
                if frame in self.objects_id_2_frameidx[object_id]:
                    idx = self.objects_id_2_frameidx[object_id].index(frame)
                    bbox_ori.append(self.objects_id_2_corners[object_id][idx][None])
                    anchor = self.objects_id_2_anchors[object_id]
                    dynamic_object_idx = self.object_id_2_global_idx[object_id]
                    tsfm = self.tsfm_vehicle[frame, dynamic_object_idx].numpy()
                    R = tsfm[:3,:3]
                    T = tsfm[:3,3]
                    bbox_interpolated = np.einsum("ij, bj -> bi", np.linalg.inv(R), anchor - T[None])
                    bbox_tsfm.append(bbox_interpolated[None])
                    
        bbox_ori = np.concatenate(bbox_ori, axis=0)
        bbox_tsfm = np.concatenate(bbox_tsfm, axis=0)
        # visulize_with_2_bbox(points3D, bbox_ori, bbox_tsfm)
        visulize_with_bbox(points3D, bbox_ori)
        exit(1)
    
    def save_vehicle_data(self, index=13):
        if self.split != "train": return

        import os
        aabb = self.aabb_vehicle[index]
        tsfm = self.tsfm_vehicle[:,index,:,:]
        extent = self.extent
        try:
            os.mkdir(f"/scratch/hanfeng/LidarSimStudio/NFLStudio_dynamic_raydrop/NFLStudio/vehicle_data/{self.context_name}")
        except FileExistsError:
            pass
        torch.save(aabb, f"/scratch/hanfeng/LidarSimStudio/NFLStudio_dynamic_raydrop/NFLStudio/vehicle_data/{self.context_name}/aabb_{index}.pt")
        torch.save(tsfm, f"/scratch/hanfeng/LidarSimStudio/NFLStudio_dynamic_raydrop/NFLStudio/vehicle_data/{self.context_name}/tsfm_{index}.pt")
        torch.save(torch.tensor(extent), f"/scratch/hanfeng/LidarSimStudio/NFLStudio_dynamic_raydrop/NFLStudio/vehicle_data/{self.context_name}/extent_{index}.pt")
        print(f'save vehicle {index} in context {self.context_name}')
        exit(1)

    def load_vehicle_data(self, to_replace_index=3, to_load_index=13, context_name_other='13271285919570645382_5320_000_5340_000'):
        aabb = torch.load(f"/scratch/hanfeng/LidarSimStudio/NFLStudio_dynamic_raydrop/NFLStudio/vehicle_data/{context_name_other}/aabb_{to_load_index}.pt")
        # tsfm = torch.load(f"/cluster/work/igp_psr/hanfeng/LidarSimStudio/NFLStudio_dynamic_raydrop/NFLStudio/vehicle_data/{context_name_other}/tsfm_{to_load_index}.pt")
        # extent = torch.load(f"/cluster/work/igp_psr/hanfeng/LidarSimStudio/NFLStudio_dynamic_raydrop/NFLStudio/vehicle_data/{context_name_other}/extent_{to_load_index}.pt").cpu().item()

        # index = torch.tensor([3,1,2]).long()
        # aabb_shift =  aabb[index] - self.aabb_vehicle[to_replace_index][index]
        self.aabb_vehicle[to_replace_index] = aabb
        # aabb[1] -=  5/self.extent
        # aabb[4] -=  5/self.extent
        aabb_shift =  (aabb[:3] - self.aabb_vehicle[0][:3])
        aabb_shift[0] +=  8/self.extent
        # aabb[3] += (aabb[0]-aabb[3])/2
        # tsfm[:,:3,3] = tsfm[:,:3,3] + aabb_shift[None]
        self.tsfm_vehicle[:,to_replace_index,:,:] = self.tsfm_vehicle[:,0,:,:]
        self.tsfm_vehicle[:,to_replace_index,:3,3] += aabb_shift[None]
        print('load vehicle')
    

    def manipulate_sensor(self):
        SHIFT = False
        LIFT = False
        Tilt = False
        DOWNSAMPLE = False
        UPSAMPLE = False
        if self.split == 'test':
            print("SHIFT:", SHIFT)
            print("LIFT:", LIFT)
            print("Tilt:", Tilt)
            print("DOWNSAMPLE:", DOWNSAMPLE)
            print("UPSAMPLE:", UPSAMPLE)
            if SHIFT:
                self.shift = np.array([1.0, 1.0, 0])[None]/self.extent
                self.ray_origins +=self.shift
            if LIFT:
                self.shift = np.array([0.0, 0.0, 1.0])[None]/self.extent
                self.ray_origins +=self.shift
            if Tilt:
                tilt_angle = 5
                alpha = np.radians(tilt_angle)
                R_y = np.array([
                        [np.cos(alpha), 0, np.sin(alpha)],
                        [0, 1, 0],
                        [-np.sin(alpha), 0, np.cos(alpha)]
                    ])
                self.ray_dirs = np.einsum("ij, abcj -> abci", R_y, self.ray_dirs)
            if DOWNSAMPLE:
                self.ray_dirs = self.ray_dirs[:,::2,:,:]
                self.ray_origins = self.ray_origins[:,::2,:,:]
                self.ray_object_indices = self.ray_object_indices[:,::2,:]
                self.normals = self.normals[:,::2,:,:]#(3,)
                self.first_dist = self.first_dist[:,::2,:]
                self.first_masks = self.first_masks[:,::2,:]#False for ray drop
                self.first_intensity = self.first_intensity[:,::2,:]
                self.first_elongation = self.first_elongation[:,::2,:]
                self.valid_normal_flag = self.valid_normal_flag[:,::2,:]
                self.valid_first_return = self.valid_first_return[:,::2,:]
            if UPSAMPLE:
                ray_dirs_upsampled = np.zeros((50,127,2650,3))
                ray_dirs_a = self.ray_dirs[:,:-1,:,:]
                ray_dirs_b = self.ray_dirs[:,1:,:,:]
                ray_dirs_interpolated = (ray_dirs_a+ray_dirs_b) / 2
                norm = np.linalg.norm(ray_dirs_interpolated,axis=3)
                ray_dirs_interpolated = ray_dirs_interpolated/norm[:,:,:,None]
                ray_dirs_upsampled[:,::2,:,:] = self.ray_dirs
                ray_dirs_upsampled[:,1::2,:,:] = ray_dirs_interpolated
                self.ray_dirs = ray_dirs_upsampled
                

                ray_origins_upsampled = np.zeros((50,127,2650,3))
                ray_origins_a = self.ray_origins[:,:-1,:,:]
                ray_origins_b = self.ray_origins[:,1:,:,:]
                ray_origins_interpolated = (ray_origins_a+ray_origins_b) / 2
                ray_origins_upsampled[:,::2,:,:] = self.ray_origins
                ray_origins_upsampled[:,1::2,:,:] = ray_origins_interpolated
                self.ray_origins = ray_origins_upsampled

                self.ray_object_indices = np.zeros((50,127,2650)).astype(np.int64) 
                self.normals = np.ones((50,127,2650,3)) 
                self.first_dist = np.ones((50,127,2650))
                self.first_masks = np.ones((50,127,2650)).astype(np.bool_)
                self.first_intensity = np.ones((50,127,2650))
                self.first_elongation = np.ones((50,127,2650))
                self.valid_normal_flag = np.ones((50,127,2650)).astype(np.bool_)
                self.valid_first_return = np.ones((50,127,2650)).astype(np.bool_)
            
    def manipulate_vehicle(self, index=0):
        REMOVAL = False
        TRAJSHIFT = False
        INSERT = True
        if self.split == 'test':
            print("REMOVAL:", REMOVAL)
            print("TRAJSHIFT:", TRAJSHIFT)
            print("INSERT:", INSERT)
            if REMOVAL:
                self.aabb_vehicle[index] *= 0
            if TRAJSHIFT:
                self.tsfm_vehicle[:,index,1,3] -= 12/self.extent
            if INSERT:
                self.load_vehicle_data(to_replace_index=5)

        







                





    def get_spatial_extent(self, normalize = True):
        """
        Get spatial extent of the scene [-height, -width/2, -width/2, height, width/2, width/2]
        """
        # assert self.split == 'train'
        # end_pts = self.lidar_data['rays_o'] + self.lidar_data['rays_d'] * self.lidar_data['first_dist'][:,None]
        end_pts = self.ray_origins + self.ray_dirs * self.first_dist[:,:,:,None]
        end_pts = end_pts[np.logical_and(self.valid_normal_flag, self.first_masks)]
        center = center = (np.min(end_pts, axis=0) + np.max(end_pts, axis=0)) /2
        end_pts = end_pts - center[None]
        self.extent = np.abs(end_pts).max() + self.margin if normalize else 1.0
        self.center = center

        height = np.abs(end_pts[:,2]).max() + self.margin
        self.bound_z = height / self.extent

        # normalise the bbox and tsfm after normalizing the coordinates 
        for object_id in self.objects_id_2_corners.keys():
            corners = self.objects_id_2_corners[object_id]
            for i in range(len(corners)):
                corners[i] = ((corners[i] - center[None])/self.extent).astype(self.dtype)
            self.objects_id_2_corners[object_id] = corners

        for object_id in self.objects_id_2_anchors.keys():
            corners = self.objects_id_2_anchors[object_id]
            corners = ((corners - center[None])/self.extent).astype(self.dtype)
            self.objects_id_2_anchors[object_id] = corners

        for object_id in self.objects_id_2_tsfm.keys():
            tsfm = self.objects_id_2_tsfm[object_id]
            for i in range(0,len(tsfm)):
                object_tsfm = tsfm[i]
                object_tsfm[:3,3] = (object_tsfm[:3,3] + object_tsfm[:3,:3] @ self.center - self.center)/self.extent #normalize the translation
                tsfm[i] = object_tsfm.astype(self.dtype)
            self.objects_id_2_tsfm[object_id] = tsfm


      
        
    def _get_frame_num(self):
        return self.first_masks.shape[0]    

    def _get_ray_num_per_frame(self):
        return self.first_masks.shape[1] * self.first_masks.shape[2]

    def _get_col_num(self):
        return self.first_masks.shape[2]    
          
    def __len__(self):
        return len(self.splix_idx) * self.first_masks.shape[1] * self.first_masks.shape[2]
        # return len(self.splix_idx) * self.first_masks.shape[1] * self.first_masks.shape[2]

    def __getitem__(self, idx):
        #               mask
        #          1 /       \0
        #       object_idx    raydrop
        #   ==-1   /  \  != -1
        # background  object with bounding box
        #
        #
        frame_idx = self.splix_idx[idx // self._get_ray_num_per_frame()] #timestamp
        offset = idx % self._get_ray_num_per_frame()
        row_idx = offset // self._get_col_num()
        col_idx = offset % self._get_col_num()
        object_idx = self.ray_object_indices[frame_idx, row_idx, col_idx]
        object_id = self.object_ids_per_frame[frame_idx][object_idx]#string
        ray_origin = self.ray_origins[frame_idx, row_idx, col_idx]#(3,) in global frame
        ray_dir = self.ray_dirs[frame_idx, row_idx, col_idx]#(3,) in global frame
        ray_normal = self.normals[frame_idx, row_idx, col_idx]#(3,)
        first_distance = self.first_dist[frame_idx, row_idx, col_idx]
        first_mask = self.first_masks[frame_idx, row_idx, col_idx]#False for ray drop
        first_intensity = self.first_intensity[frame_idx, row_idx, col_idx]
        first_elongation = self.first_elongation[frame_idx, row_idx, col_idx]
        # second_distance = self.second_dist[frame_idx, row_idx, col_idx]
        # second_mask = self.second_masks[frame_idx, row_idx, col_idx]#False for ray drop
        # second_intensity = self.second_intensity[frame_idx, row_idx, col_idx]
        # second_elongation = self.second_elongation[frame_idx, row_idx, col_idx]



        ### When object_idx == -1, that means the object is either background or ray drop, 
        ### the following 4 terms are then meaningless, hence assign a default value
        object_type = self.objects_id_types_per_frame[frame_idx][object_idx] if object_idx != -1 else -1 #int
        #   TYPE_UNKNOWN = 0
        #   TYPE_VEHICLE = 1
        #   TYPE_PEDESTRIAN = 2
        #   TYPE_SIGN = 3
        #   TYPE_CYCLIST = 4
        occurred_idx = self.objects_id_2_frameidx[object_id].index(frame_idx) if object_idx != -1 else int(-1) # is 0 if the first occurence
        object_bbox = self.objects_id_2_corners[object_id][occurred_idx] if object_idx != -1 else np.zeros((8,3))#(8,3) in world coordinates
        object_tsfm = self.objects_id_2_tsfm[object_id][occurred_idx] if object_idx != -1 else np.eye(4)#(4,4) wrt. the first occurrence

        # test:
        # if object_idx != -1:
        #     if occurred_idx !=0:
        #         object_bbox_anchor = self.objects_id_2_corners[object_id][0]
        #         estimated_anchor = np.einsum("ij, ...j -> ...i", object_tsfm[:3,:3], object_bbox) + object_tsfm[:3,3][None]
        #         print(estimated_anchor - object_bbox_anchor)



        dynamic_flag = self.objects_id_2_dynamic_flag[object_id] if object_id in self.objects_id_2_dynamic_flag.keys() else False

        valid_normal_flag = self.valid_normal_flag[frame_idx, row_idx, col_idx]


        valid_first_return = self.valid_first_return[frame_idx, row_idx, col_idx]

        dynamic_only_mask = first_mask and dynamic_flag and valid_normal_flag

        vehicle_mask = dynamic_only_mask and object_type==1

        static_vehicle_mask = object_type==1 and  (not dynamic_flag) and first_mask and valid_normal_flag

        static_mask = (not dynamic_only_mask) and first_mask and valid_normal_flag

        dynamic_vehicle_global_idx = self.object_id_2_global_idx[object_id] if object_id in self.object_id_2_global_idx.keys() else -1

        # anchor_bbox = self.aabb_vehicle[dynamic_vehicle_global_idx] if object_id in self.object_id_2_global_idx.keys() else torch.zeros(6)
        
        # tsfm_ray_origin = object_tsfm[:3,:3] @ ray_origin + object_tsfm[:3,3]
        # in_anchor = tsfm_ray_origin[0] > anchor_bbox[0] and tsfm_ray_origin[1] > anchor_bbox[1] and tsfm_ray_origin[2] > anchor_bbox[2] and tsfm_ray_origin[0] < anchor_bbox[3] and tsfm_ray_origin[1] < anchor_bbox[4] and tsfm_ray_origin[2] < anchor_bbox[5] 



        return_dict= {
            'rays_o': ray_origin,
            'rays_d': ray_dir,
            'rays_normal': ray_normal,
            'first_mask': first_mask and valid_normal_flag,
            # 'second_mask': second_mask,
            'first_dist': first_distance,
            # 'second_dist': second_distance,
            'first_intensity': first_intensity, 
            # 'second_intensity': second_intensity,
            'first_elongation': first_elongation,
            # 'second_elongation': second_elongation,
            'row_idx': row_idx,
            'col_idx': col_idx,
            # 'object_id': object_id,
            'dynamic_vehicle_idx': dynamic_vehicle_global_idx,
            'type': object_type,
            'object_bbox': object_bbox,
            # 'anchor_bbox': anchor_bbox,
            'object_tsfm': object_tsfm,
            'time_stamp': frame_idx,
            # 'dynamic_only_mask': dynamic_only_mask,
            'vehicle_mask': vehicle_mask,
            'static_mask': static_mask,
            'static_vehicle_mask': static_vehicle_mask
            # 'dynamic_mask': dynamic_only_mask or static_mask
            # 'in_anchor': in_anchor,
        }

        # for key in self.float_args:
        #     return_dict[key] = torch.tensor(return_dict[key])


        return return_dict     
        



        

