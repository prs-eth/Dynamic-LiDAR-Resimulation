from typing import Dict, List, Union
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.models.base_model import *
from torch.nn import Parameter
from NFLStudio.nflfield import NFLField


import torch, math
import torch.nn as nn
import torch.nn.functional as F

import tinycudann as tcnn
from NFLStudio.libs.activation import trunc_exp
from NFLStudio import raymarching
from NFLStudio.libs.utils import _EPS
from copy import deepcopy
from NFLStudio.datamanager import (
    NFLDataManagerConfig,
    NFLDataManager,
)

from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.model_components.ray_samplers import PDFSampler, UniformSampler, NeuSSampler

from nerfstudio.cameras.rays import Frustums, RaySamples
from NFLStudio.loss import NerfLoss
from NFLStudio.datamanager import (
    NFLDataManagerConfig,
    NFLDataParserConfig
)

# from NFLStudio.libs.utils import interpolate_rigid_transformation


@dataclass
class TrainConfig:
    lr: float = 0.005
    verbose_interval: int = 1
    max_iters: int = 100000
    iter_size: int = 1  # gradient accumulation 
    ema_decay: float = -1  # 0.95
    clip: float =  1.0  # gradient clip
    metric: str='d_l1'
    weight_threshold: float = 0.1
    max_epoch: int = 1

    # d_url: float = 3.0   # depth 
    # clamp_eps: float = 0.01  # used in GaussianNLLLoss
    # reweight: bool = True


@dataclass
class NFLModelConfig(ModelConfig):
    _target: Type = field(default_factory=lambda: NFLModel)
    nerf_model: str='ngp'
    peak_detection: bool = False  # run peak detection over the sigma values
    peak_threshold: float = 25
    num_coarse_steps: int = 768  # num steps sampled per ray 
    num_fine_steps: int = 64  # num steps up-sampled per ray,[currently let's just dis-able this]
    encoding_sdf:str =  'HashGrid'
    encoding_dir: str='SphericalHarmonics'
    activation: str='relu'  #[relu or trunc_exp]
    per_level_scale: float = 1.6
    max_std: float = 1.6
    min_std: float = 0.3
    device: str = 'cuda:0'
    center: Union[torch.Tensor, None] = None
    extent: Union[torch.Tensor, None] = None
    num_vehicles: int = 0
    aabb_vehicle: List[torch.tensor] = field(default_factory=list)
    tsfm_vehicle_train: torch.tensor = None
    tsfm_vehicle_eval: torch.tensor = None
    save_dir: str = './save_dir/'
    # train_config: TrainConfig = TrainConfig()
    loss: Dict[str, object] = field(default_factory=lambda:{
                                                            'd_url': 3.0, 'clamp_eps': 0.01, 'reweight': True, 
                                                            'eikonal': 0.3, "sdf": 1.0, "surface_sdf":3.0, "depth_vol":3,
                                                            'i_l2': 50.0, 'rdrop_bce': 0.15, 'rdrop_ls': 0.15})
    datamanager_config: NFLDataManagerConfig = NFLDataManagerConfig()


class NFLModel(Model):
    # model_cfg: NFLModelConfig = NFLModelConfig()

    def __init__(self, config: NFLModelConfig, scene_box: SceneBox, num_train_data: int,**kwargs) -> None:

        super().__init__(config, scene_box, num_train_data, **kwargs)

        self.loss = NerfLoss(self.config)


        # self.aabb_vehicle = aabb_vehicle
        # print("load aabb vehicle into NFLModel: ",aabb_vehicle)


    def populate_modules(self):
        super().populate_modules()
        self.aabb_vehicle = self.config.aabb_vehicle
        self.tsfm_vehicle_train = self.config.tsfm_vehicle_train.to('cuda')
        self.tsfm_vehicle_eval = self.config.tsfm_vehicle_eval.to('cuda')
        for i in range(len(self.aabb_vehicle)):
            self.aabb_vehicle[i] = self.aabb_vehicle[i].float().to('cuda')
        self.min_near = self.config.datamanager_config.dataparser_config.min_near / self.config.extent.item()
        self.nfl_field = NFLField(
            encoding_sdf= self.config.encoding_sdf,
            encoding_dir= self.config.encoding_dir,
            per_level_scale = self.config.per_level_scale,
            extent = self.config.extent,
        )

        self.vehicle_fields = nn.ModuleList([NFLField(
            encoding_sdf= self.config.encoding_sdf,
            encoding_dir= self.config.encoding_dir,
            per_level_scale = self.config.per_level_scale,
            extent = self.config.extent,
        ) for i in range(self.config.num_vehicles)])
        print('initialize vehicle fields')

        self.sampler_neus = NeuSSampler(num_samples=256, num_samples_importance=256, num_upsample_steps=8)
        self.sampler_neus_dynamic = NeuSSampler(num_samples=64, num_samples_importance=64, num_upsample_steps=4)
        self.anneal_end = 40000

    def get_outputs(self, ray_bundle_list: list[RayBundle], ray_batch_list=None, intersection_ray_bundle_list=None, intersection_ray_batch_list=None, vehicle_mask_list=None):
        ray_samples_static = self.sampler_neus(ray_bundle=ray_bundle_list[0], sdf_fn=self.nfl_field.get_sdf)
        outputs_static = self.nfl_field.forward(ray_samples_static, rays_batch=ray_batch_list[0])
        outputs_list = [outputs_static]
        ray_batch_full = ray_batch_list[0]
        true_vehicle_mask = ray_batch_full['vehicle_mask']

        outputs_static['ray_drop_prob'][true_vehicle_mask] = outputs_static['ray_drop_prob'][true_vehicle_mask].detach()#detach dynamic ray
        # outputs_static['ray_drop_prob'] = outputs_static['ray_drop_prob'][~true_vehicle_mask]
        raydrop_prob_all = [outputs_static['ray_drop_prob'][~true_vehicle_mask]]

        static_ray_hit_mask = ray_batch_full['static_mask'].bool()
        hit_mask_all = [static_ray_hit_mask[~true_vehicle_mask]]
        #put dynamic entries back to the full batch
        for i in range(1, len(ray_bundle_list)):
            if intersection_ray_batch_list[i]['rays_o'].shape[0] < 1: continue
            ray_samples_dynamic = self.sampler_neus_dynamic(ray_bundle=intersection_ray_bundle_list[i], sdf_fn=self.vehicle_fields[i-1].get_sdf)
            outputs_dynamic = self.vehicle_fields[i-1].forward(ray_samples_dynamic, rays_batch=intersection_ray_batch_list[i])
            mask = vehicle_mask_list[i-1]
            for key in outputs_static.keys():
                tmp = outputs_static[key]
                tmp[mask] = outputs_dynamic[key][intersection_ray_batch_list[i]['hit_mask']]
                outputs_static[key] = tmp
            raydrop_prob_all.append(outputs_dynamic['ray_drop_prob'])
            hit_mask_all.append(intersection_ray_batch_list[i]['hit_mask'])

        raydrop_prob_all = torch.cat(raydrop_prob_all,dim=0)
        hit_mask_all = torch.cat(hit_mask_all,dim=0)

        outputs_static.update({
            'raydrop_prob_all': raydrop_prob_all,
            'hit_mask_all': hit_mask_all
        })   
        return outputs_static
    
    def get_outputs_eval(self, ray_bundle_list: list[RayBundle], ray_batch_list=None, vehicle_mask_list=None):
        ray_samples_static = self.sampler_neus(ray_bundle=ray_bundle_list[0], sdf_fn=self.nfl_field.get_sdf)
        outputs_static = self.nfl_field.forward(ray_samples_static, rays_batch=ray_batch_list[0])
        ray_drop_prob = outputs_static['ray_drop_prob']
        ray_drop_mask = ray_drop_prob[:,1] > ray_drop_prob[:,0] # 1 is drop 0 is hit
        outputs_list = [outputs_static]
        ray_batch_full = ray_batch_list[0]
        # put dynamic entries back to the full batch
        valid_hit_mask = torch.zeros((vehicle_mask_list[0].shape[0], len(vehicle_mask_list)), dtype=torch.bool).cuda() #keep track of all intersections
        valid_hit_distance = torch.ones_like(valid_hit_mask).cuda() * 3.1028e38

        valid_dynamic_mask_all = torch.zeros_like(vehicle_mask_list[0]).bool()
        dynamic_outputs_all = []
        for i in range(1, len(ray_bundle_list)):
            if ray_batch_list[i]['rays_o'].shape[0] < 1: 
                dynamic_outputs_all.append(None)
                continue
            ray_samples_dynamic = self.sampler_neus_dynamic(ray_bundle=ray_bundle_list[i], sdf_fn=self.vehicle_fields[i-1].get_sdf)
            outputs_dynamic = self.vehicle_fields[i-1].forward(ray_samples_dynamic, rays_batch=ray_batch_list[i])
            dynamic_outputs_all.append(outputs_dynamic)
            dynamic_ray_drop_prob = outputs_dynamic['ray_drop_prob']
            dynamic_ray_drop_mask = dynamic_ray_drop_prob[:,1]>dynamic_ray_drop_prob[:,0]
            mask = vehicle_mask_list[i-1].clone()
            mask_clone = mask.clone()
        
            wrong_dynamic = dynamic_ray_drop_mask 
            tmp = mask_clone[mask]
            tmp[wrong_dynamic] = False
            mask_clone[mask] = tmp
            # vehicle_mask_list[i-1] = mask_clone


            # mask = vehicle_mask_list[i-1]

            mask = mask_clone

            static_distance = outputs_static['depth_vol_c'][mask]
            dynamic_distance = outputs_dynamic['depth_vol_c'][~wrong_dynamic]
            if dynamic_distance.shape[0] < 1 : continue

            # compare if the dynamic rendered distance is closer than the static rendered distance
            valid_dynamic = torch.logical_or(dynamic_distance < static_distance, ray_drop_mask[mask])


            mask_clone = mask.clone()
            mask_tmp = mask_clone[mask]
            mask_tmp[~valid_dynamic] = False
            mask_clone[mask] = mask_tmp
            valid_dynamic_mask_all = torch.logical_or(valid_dynamic_mask_all, mask_clone)



            valid_hit_mask[:,i-1] = mask_clone # mask_clone marks for ray 1. hit the vehicle 2. smaller distance than static distance
            valid_hit_distance[mask_clone,i-1] = outputs_dynamic['depth_vol_c'][~wrong_dynamic][valid_dynamic] # keep track of the dynamic distance

        


        min_dynamic_distance, argmin_dynamic_distance = torch.min(valid_hit_distance, dim=1)
        argmin_mask = torch.zeros_like(valid_hit_mask,dtype=torch.bool)
        argmin_mask[torch.arange(argmin_mask.shape[0]), argmin_dynamic_distance] = True

        true_dynamic_distance_mask = torch.logical_and(argmin_mask, valid_hit_mask) # filter out the case where minimum is 3.1028e38, B x N_v

        true_dynamic_distance = valid_hit_distance[true_dynamic_distance_mask] #[N_valid_dynamic_distance]
        true_dynamic_distance_mask_overall = torch.any(true_dynamic_distance_mask, dim=1)


        for i in range(len(ray_bundle_list)-1):
            outputs_dynamic = dynamic_outputs_all[i]
            if outputs_dynamic is None: continue
            dynamic_vehicle_mask_from_allbatch = true_dynamic_distance_mask[:,i]
            dynamic_vehicle_mask_from_dynamicbatch = true_dynamic_distance_mask[:,i][vehicle_mask_list[i]]
            # replace the dynamic distance with static distance
            mask = vehicle_mask_list[i]
            for key in outputs_static.keys():
                outputs_static[key][dynamic_vehicle_mask_from_allbatch] = outputs_dynamic[key][dynamic_vehicle_mask_from_dynamicbatch]


        outputs_static.update({"predicted_vehicle_mask": true_dynamic_distance_mask_overall})
        return outputs_static  
    

    def get_metrics_dict(self, outputs, batch):
        """Returns metrics dictionary which will be plotted with wandb or tensorboard."""
        return {}


    def get_loss_dict(self, outputs, batch, phase, iter, epoch, metrics_dict=None):
        """Returns a dictionary of losses to be summed which will be your loss."""  

        results = self.loss(outputs, batch, phase, iter, epoch)
        return results
    
    @staticmethod
    def scale_aabb(aabb, ratio=1.2):
        center = (aabb[:3] + aabb[3:])/2
        min = (aabb[:3] - center)*(ratio-1) + aabb[:3]
        max = (aabb[3:] - center)*(ratio-1) + aabb[3:]
        return torch.cat([min,max],dim=0)


    def _forwards_train(self, ray_bundle_list: list[RayBundle], ray_batch_list=None) -> Dict[str, torch.Tensor]:
        # assert self.train
        aabb_static = torch.tensor([-1.,-1.,-1/2,1.,1.,1/2]).float().to(self.device)
        nears, fars = raymarching.near_far_from_aabb(ray_batch_list[0]['rays_o'].float(), 
                                                     ray_batch_list[0]['rays_d'].float(), 
                                                     aabb_static, 
                                                     self.min_near)

        
        nears = nears[:,None]
        fars = fars[:,None]
        ray_bundle_list[0].nears = nears
        ray_bundle_list[0].fars = fars

        intersection_ray_bundle_list = [ray_bundle_list[0]]
        intersection_ray_batch_list = [ray_batch_list[0]] + [{} for i in range(self.config.num_vehicles)]
        
        frame_idx = ray_batch_list[0]['time_stamp']
        ##calculate the intersection of ray and the anchor box
        tsfm_for_all_objects = self.tsfm_vehicle_train[frame_idx]
        for i in range(1, len(ray_bundle_list)):
            object_tsfm = tsfm_for_all_objects[:,i-1,:,:]
            # if ray_batch_list[i]['rays_o'].shape[0] < 1: continue
            aabb_dynamic = self.aabb_vehicle[i-1]
            R = object_tsfm[:,:3,:3]
            T = object_tsfm[:,:3,3]
            ray_o_transformed = torch.einsum("bij,bj->bi",R, ray_batch_list[0]['rays_o']) + T
            ray_d_transformed = torch.einsum("bij,bj->bi",R, ray_batch_list[0]['rays_d'])
            nears, fars = raymarching.near_far_from_aabb(ray_o_transformed, 
                                                            ray_d_transformed, 
                                                            aabb_dynamic.float().to(self.device), 
                                                            self.min_near)

        
            nears = nears[:,None].to(self.device)
            fars = fars[:,None].to(self.device)
            intersection_ray_bundle_list.append(RayBundle(
                nears=nears,
                fars=fars,
                origins=ray_o_transformed,
                directions=ray_d_transformed,
                times=None,
                pixel_area=None
            ))

        ##compare nears, 
        all_nears_intersect_dynamic_objects = torch.cat([intersection_ray_bundle_list[i].nears for i in range(1, len(intersection_ray_bundle_list))], dim=1)
        all_fars_intersect_dynamic_objects = torch.cat([intersection_ray_bundle_list[i].fars for i in range(1, len(intersection_ray_bundle_list))], dim=1)
        negative_direction = all_fars_intersect_dynamic_objects<0
        all_nears_intersect_dynamic_objects[negative_direction] = 3.1028e38
        min_all_nears, argmin_all_nears = torch.min(all_nears_intersect_dynamic_objects, dim=1)
        dynamic_mask = min_all_nears <=2.0

        vehicle_mask_list = []
        for i in range(1, len(ray_bundle_list)):
            intersected_mask = torch.where(argmin_all_nears==(i-1), True, False)
            intersected_mask = torch.logical_and(intersected_mask, dynamic_mask)
            intersection_ray_bundle_list[i].nears = intersection_ray_bundle_list[i].nears[intersected_mask]
            intersection_ray_bundle_list[i].fars = intersection_ray_bundle_list[i].fars[intersected_mask]
            intersection_ray_bundle_list[i].origins = intersection_ray_bundle_list[i].origins[intersected_mask]
            intersection_ray_bundle_list[i].directions = intersection_ray_bundle_list[i].directions[intersected_mask]
            for key in ray_batch_list[0].keys():
                intersection_ray_batch_list[i].update({key:ray_batch_list[0][key][intersected_mask]})
            vehicle_true_mask = torch.logical_and(ray_batch_list[0]['dynamic_vehicle_idx'] == i-1, ray_batch_list[0]['vehicle_mask'].bool()) 
            intersection_ray_batch_list[i].update({'hit_mask':vehicle_true_mask[intersected_mask]}) ##hit mask for vehicle
            vehicle_mask_list.append(torch.logical_and(vehicle_true_mask, intersected_mask)) #intersected and is true vehicle

        return self.get_outputs(ray_bundle_list, ray_batch_list, intersection_ray_bundle_list, intersection_ray_batch_list, vehicle_mask_list)

    def _forwards_eval(self, ray_bundle_list: list[RayBundle], ray_batch_list=None) -> Dict[str, torch.Tensor]:
        aabb_static = torch.tensor([-1.,-1.,-1/2,1.,1.,1/2]).float().to(self.device)
        ray_batch_full = ray_batch_list[0]
        frame_idx = ray_batch_full['time_stamp']
        
        nears, fars = raymarching.near_far_from_aabb(ray_batch_list[0]['rays_o'].float(), 
                                                     ray_batch_list[0]['rays_d'].float(), 
                                                     aabb_static, 
                                                     self.min_near)

        
        nears = nears[:,None].to(self.device)
        fars = fars[:,None].to(self.device)
        ray_bundle_list[0].nears = nears
        ray_bundle_list[0].fars = fars

        # calculate the intersection of ray and the anchor box
        tsfm_for_all_objects = self.tsfm_vehicle_eval[frame_idx]
        for i in range(1, len(ray_bundle_list)):
            object_tsfm = tsfm_for_all_objects[:,i-1,:,:]
            # if ray_batch_list[i]['rays_o'].shape[0] < 1: continue
            aabb_dynamic = self.aabb_vehicle[i-1]
            R = object_tsfm[:,:3,:3]
            T = object_tsfm[:,:3,3]
            ray_o_transformed = torch.einsum("bij,bj->bi",R, ray_batch_list[0]['rays_o']) + T
            ray_d_transformed = torch.einsum("bij,bj->bi",R, ray_batch_list[0]['rays_d'])
            nears, fars = raymarching.near_far_from_aabb(ray_o_transformed, 
                                                         ray_d_transformed, 
                                                         aabb_dynamic.float().to(self.device), 
                                                         self.min_near)
            nears = nears[:,None].to(self.device)
            fars = fars[:,None].to(self.device)
            ray_bundle_list[i].nears = nears
            ray_bundle_list[i].fars = fars
            ray_bundle_list[i].origins = ray_o_transformed
            ray_bundle_list[i].directions = ray_d_transformed
        

        # compare nears, 
        all_nears_intersect_dynamic_objects = torch.cat([ray_bundle_list[i].nears for i in range(1, len(ray_bundle_list))], dim=1)
        all_fars_intersect_dynamic_objects = torch.cat([ray_bundle_list[i].fars for i in range(1, len(ray_bundle_list))], dim=1)
        negative_direction = all_fars_intersect_dynamic_objects<0
        all_nears_intersect_dynamic_objects[negative_direction] = 3.1028e38
        # min_all_nears, argmin_all_nears = torch.min(all_nears_intersect_dynamic_objects, dim=1)
        # dynamic_mask = min_all_nears <=2.0
        # _, sort_indices = torch.sort(all_nears_intersect_dynamic_objects, dim=1)


        vehicle_mask_list = []
        for i in range(1, len(ray_bundle_list)):
            vehicle_mask = torch.where(all_nears_intersect_dynamic_objects[:,i-1]<=2.0, True, False) # select all intersected ray with this bbox
            vehicle_mask_list.append(vehicle_mask)
            ray_bundle_list[i].nears = ray_bundle_list[i].nears[vehicle_mask]
            ray_bundle_list[i].fars = ray_bundle_list[i].fars[vehicle_mask]
            ray_bundle_list[i].origins = ray_bundle_list[i].origins[vehicle_mask]
            ray_bundle_list[i].directions = ray_bundle_list[i].directions[vehicle_mask]
            for key in ray_batch_list[0].keys():
                ray_batch_list[i].update({key:ray_batch_list[0][key][vehicle_mask]})

    
        return self.get_outputs_eval(ray_bundle_list, ray_batch_list, vehicle_mask_list)
    def forward(self, ray_bundle_list: list[RayBundle], ray_batch_list=None, train=True) -> Dict[str, torch.Tensor]:
        """Run forward starting with a ray bundle. This outputs different things depending on the configuration
        of the model and whether or not the batch is provided (whether or not we are training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included
        """
        if self.training:
        # if True:
            return self._forwards_train(ray_bundle_list, ray_batch_list)
        else:
            return self._forwards_eval(ray_bundle_list,ray_batch_list)


    
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["fields"] = list(self.nfl_field.parameters())
        for i in range(len(self.vehicle_fields)):
            param_groups["fields"] += list(self.vehicle_fields[i].parameters())
        for name, param in self.nfl_field.named_parameters():
            print(f'{name}: {param.size()}')
        # if self.field_coarse is None or self.field_fine is None:
        #     raise ValueError("populate_fields() must be called before get_param_groups")
        # param_groups["fields"] = list(self.field_coarse.parameters()) + list(self.field_fine.parameters())
        # if self.temporal_distortion is not None:
        #     param_groups["temporal_distortion"] = list(self.temporal_distortion.parameters())
        return param_groups
    
    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        # anneal for cos in NeuS
        if self.anneal_end > 0:

            def set_anneal(step):
                anneal = min([1.0, step / self.anneal_end])
                self.nfl_field.set_cos_anneal_ratio(anneal)
                if step > self.anneal_end:
                    self.config.finetune = True
                for i in range(len(self.vehicle_fields)):
                    self.vehicle_fields[i].set_cos_anneal_ratio(anneal)

            callbacks+=[
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                ),
            ]

        return callbacks
    


