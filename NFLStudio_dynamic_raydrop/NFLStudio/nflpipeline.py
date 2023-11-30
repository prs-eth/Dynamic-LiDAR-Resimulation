from nerfstudio.pipelines.base_pipeline import *
from NFLStudio.nflmodel import (
    NFLModelConfig,
    NFLDataManagerConfig,
    NFLDataParserConfig
)
from NFLStudio.nflmodel import *

from NFLStudio.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist

import os
import numpy as np
from tqdm import tqdm
@dataclass
class NFLPipelineConfig(cfg.InstantiateConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: NFLPipeline)
    """target class to instantiate"""
    datamanager: NFLDataManagerConfig = NFLDataManagerConfig()
    """specifies the datamanager config"""
    model: NFLModelConfig = NFLModelConfig()
    """specifies the model config"""

class NFLPipeline(Pipeline):
    """The pipeline class for the vanilla nerf setup of multiple cameras for one or a few scenes.

    Args:
        config: configuration to instantiate pipeline
        device: location to place model and data
        test_mode:
            'val': loads train/val datasets into memory
            'test': loads train/test dataset into memory
            'inference': does not load any dataset into memory
        world_size: total number of machines available
        local_rank: rank of current machine

    Attributes:
        datamanager: The data manager that will be used
        model: The model that will be used
    """

    def __init__(
        self,
        config: NFLPipelineConfig,
        device: str = 'cpu',
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 0,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):

        super().__init__()
        self.config = config
        self.test_mode = test_mode
        # self.datamanager = NFLDataManager(NFLDataManagerConfig(), config)
        self.datamanager: NFLDataManager = config.datamanager.setup(
            device=device
        )
        self.datamanager.to(device)
        # TODO(ethan): get rid of scene_bounds from the model
        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        scenebox = SceneBox(aabb=torch.tensor([[-1,-1,-1/2], [1,1,1/2]]).to(device))

        config.model.device = device
        config.model.center = self.datamanager.train_dataset.center
        config.model.extent = self.datamanager.train_dataset.extent
        config.model.num_vehicles = self.datamanager.train_dataset.dynamic_object_counter
        config.model.aabb_vehicle = self.datamanager.train_dataset.aabb_vehicle
        config.model.tsfm_vehicle_train = self.datamanager.train_dataset.tsfm_vehicle
        config.model.tsfm_vehicle_eval = self.datamanager.eval_dataset.tsfm_vehicle

        self._model = config.model.setup(
            scene_box=scenebox,
            num_train_data=len(self.datamanager.train_dataset,),
            
        )
        self.model.to(device)

        # self.world_size = world_size
        # if world_size > 1:
        #     self._model = typing.cast(Model, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True))
        #     dist.barrier(device_ids=[local_rank])

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.model.device

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        self.train()
        phase = 'train'
        ray_bundle_list, ray_batch_list = self.datamanager.next_train(step)


        model_outputs = self.model(ray_bundle_list, ray_batch_list,train=True)
        metrics_dict = self.model.get_metrics_dict(model_outputs, ray_batch_list)


        # decay_rate = self.datamanager.train_count / self.model.config.train_config.max_iters
        loss_dict, metrics_dict = self.model.get_loss_dict(model_outputs, ray_batch_list[0], phase, self.datamanager.train_count, self.datamanager.train_epoch)
        # loss_dict = {}
        # loss_dict['loss'] = result['loss']
        # self.train_count +=1

        return model_outputs, loss_dict, metrics_dict

    def forward(self):
        """Blank forward method

        This is an nn.Module, and so requires a forward() method normally, although in our case
        we do not need a forward() method"""
        raise NotImplementedError

    @profiler.time_function
    def get_eval_loss_dict(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        phase = 'val'
        self.eval()
        ray_bundle_list, ray_batch_list = self.datamanager.next_eval(step)


        model_outputs = self.model(ray_bundle_list, ray_batch_list,train=False)
        metrics_dict = self.model.get_metrics_dict(model_outputs, ray_batch_list)
        # decay_rate = self.datamanager.train_count / self.model.config.train_config.max_iters
        # model_outputs['coarse']['decay_rate'] = decay_rate
        loss_dict, metrics_dict = self.model.get_loss_dict(model_outputs, ray_batch_list[0], phase, self.datamanager.eval_count, self.datamanager.train_epoch)
        # self.eval_count += 1
        self.train()
        return model_outputs, loss_dict, metrics_dict
    
    @profiler.time_function
    def get_test_loss_dict(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        with torch.no_grad():
            phase = 'test'
            self.eval()
            batch_size = self.datamanager.eval_loader.batch_size
            results=None
            batch_full = None
            for i in range(2650*64 // batch_size):
                ray_bundle, batch = self.datamanager.next_eval(step)
                model_outputs = self.model(ray_bundle, batch)
                if results is None:
                    results = model_outputs
                if batch_full is None:
                    batch_full = batch    
                else:
                    for key, value in model_outputs.items():
                        if key=='ratio': continue
                        results[key] = torch.cat((results[key], value), dim=0)
                    for key, value in batch.items():
                        batch_full[key] = torch.cat((batch_full[key], value), dim=0)      
                            
                
            metrics_dict = self.model.get_metrics_dict(results, batch)
            decay_rate = self.datamanager.train_count / self.model.config.train_config.max_iters
            results['coarse']['decay_rate'] = decay_rate
            loss_dict, metrics_dict = self.model.get_loss_dict(results, batch_full, phase,step, 3)
            # self.eval_count += 1
            return model_outputs, loss_dict, metrics_dict
    
    def get_numbers(self, step: int):
        """get metrics for each the scene on the test set"""
        with torch.no_grad():
            cd_metric = chamfer_3DDist()
            phase = 'val'
            self.eval()
            batch_size = self.datamanager.test_loader.batch_size
            mean =0
            median =0
            cd = 0
            length_per_frame = 2650*64
            pcd_gt = []
            pcd_est = []
            dist_gt = []
            dist_est = []
            frame_num = 10
            for f in range(frame_num):
                pcd_gt_frame = []
                pcd_est_frame = []
                dist_gt_frame = []
                dist_est_frame = []
                for i in range( length_per_frame// batch_size):
                    ray_bundle_list, batch_list = self.datamanager.next_test(step)
                    results = self.model(ray_bundle_list, batch_list)
                    batch = batch_list[0]     
                    mask = torch.logical_or(batch['static_mask'].bool(), batch['vehicle_mask'].bool())
                    pcd_gt_frame.append((batch['rays_o'][mask] + batch['rays_d'][mask] * batch['first_dist'][:,None][mask])*self.datamanager.train_dataset.extent)             
                    pcd_est_frame.append((batch['rays_o'][mask] + batch['rays_d'][mask] * results['depth_vol_c'][:,None][mask])*self.datamanager.train_dataset.extent)        
                    dist_gt_frame.append(batch['first_dist'][mask]*self.datamanager.train_dataset.extent)     
                    dist_est_frame.append(results['depth_vol_c'][mask]*self.datamanager.train_dataset.extent)     
                
                    # decay_rate = self.datamanager.train_count / self.model.config.train_config.max_iters
                    # results['coarse']['decay_rate'] = decay_rate
                    loss_dict, metrics_dict = self.model.get_loss_dict(results, batch, "val",step, 3)
                    # mean += metrics_dict['depth_vol_mean'].item()
                    # median += metrics_dict['depth_vol_median']
                    # cd += metrics_dict['chamfer_dist_vol']

                pcd_gt_frame = torch.cat(pcd_gt_frame, dim=0)    
                pcd_est_frame = torch.cat(pcd_est_frame, dim=0)    
                dist_gt_frame = torch.cat(dist_gt_frame, dim=0)    
                dist_est_frame = torch.cat(dist_est_frame, dim=0)    

                pcd_gt.append(pcd_gt_frame)
                pcd_est.append(pcd_est_frame)
                dist_gt.append(dist_gt_frame)
                dist_est.append(dist_est_frame)

            # mean /=  (length_per_frame * 10// batch_size) / 100

             

            # pcd_gt = torch.cat(pcd_gt, dim=0)    
            # pcd_est = torch.cat(pcd_est, dim=0)    
            dist_gt_all = torch.cat(dist_gt, dim=0)    
            dist_est_all = torch.cat(dist_est, dim=0)    
            error_abs = torch.abs(dist_est_all-dist_gt_all)*100
            for i in range(frame_num):
                dist1, dist2, idx1, idx2 = cd_metric(pcd_gt[i][None],pcd_est[i][None])
                dist1, dist2 = dist1**0.5, dist2**0.5
                cd += dist1.mean().detach() + dist2.mean().detach()

                median += torch.abs(dist_gt[i] - dist_est[i]).median()


            cd /= (frame_num) / 100
            mean = error_abs.mean()
            median = error_abs.median()
           



            print("mean: ", mean )
            print("median: ", median )
            print("cd: ", cd ) 

    def get_numbers_vehicles(self):
        """get metrics for each the scene on the test set"""
        with torch.no_grad():
            cd_metric = chamfer_3DDist()
            phase = 'val'
            self.eval()
            batch_size = self.datamanager.test_loader.batch_size
            mean_dynamic =0
            median_dynamic =0
            cd_dynamic = 0
            mean_static =0
            median_static =0
            cd_static = 0
            length_per_frame = 2650*64
            pcd_gt_dynamic = []
            pcd_est_dynamic = []
            dist_gt_dynamic = []
            dist_est_dynamic = []
            pcd_gt_static = []
            pcd_est_static = []
            dist_gt_static = []
            dist_est_static = []
            frame_num = 10
            for f in range(frame_num):
                pcd_gt_frame_dynamic = []
                pcd_est_frame_dynamic = []
                dist_gt_frame_dynamic = []
                dist_est_frame_dynamic = []
                pcd_gt_frame_static = []
                pcd_est_frame_static = []
                dist_gt_frame_static = []
                dist_est_frame_static = []
                for i in range( length_per_frame// batch_size):
                    ray_bundle_list, batch_list = self.datamanager.next_test(0)
                    results = self.model(ray_bundle_list, batch_list)
                    batch = batch_list[0]     
                    dynamic_vehicle_mask = batch['vehicle_mask'].bool()
                    static_vehicle_mask = batch['static_vehicle_mask'].bool()
                    pcd_gt_frame_dynamic.append((batch['rays_o'][dynamic_vehicle_mask] + batch['rays_d'][dynamic_vehicle_mask] * batch['first_dist'][:,None][dynamic_vehicle_mask])*self.datamanager.train_dataset.extent)             
                    pcd_est_frame_dynamic.append((batch['rays_o'][dynamic_vehicle_mask] + batch['rays_d'][dynamic_vehicle_mask] * results['depth_vol_c'][:,None][dynamic_vehicle_mask])*self.datamanager.train_dataset.extent)        
                    dist_gt_frame_dynamic.append(batch['first_dist'][dynamic_vehicle_mask]*self.datamanager.train_dataset.extent)     
                    dist_est_frame_dynamic.append(results['depth_vol_c'][dynamic_vehicle_mask]*self.datamanager.train_dataset.extent)     
                    pcd_gt_frame_static.append((batch['rays_o'][static_vehicle_mask] + batch['rays_d'][static_vehicle_mask] * batch['first_dist'][:,None][static_vehicle_mask])*self.datamanager.train_dataset.extent)             
                    pcd_est_frame_static.append((batch['rays_o'][static_vehicle_mask] + batch['rays_d'][static_vehicle_mask] * results['depth_vol_c'][:,None][static_vehicle_mask])*self.datamanager.train_dataset.extent)        
                    dist_gt_frame_static.append(batch['first_dist'][static_vehicle_mask]*self.datamanager.train_dataset.extent)     
                    dist_est_frame_static.append(results['depth_vol_c'][static_vehicle_mask]*self.datamanager.train_dataset.extent)     
                

                pcd_gt_frame_dynamic = torch.cat(pcd_gt_frame_dynamic, dim=0)    
                pcd_est_frame_dynamic = torch.cat(pcd_est_frame_dynamic, dim=0)    
                dist_gt_frame_dynamic = torch.cat(dist_gt_frame_dynamic, dim=0)    
                dist_est_frame_dynamic = torch.cat(dist_est_frame_dynamic, dim=0)    
                pcd_gt_frame_static = torch.cat(pcd_gt_frame_static, dim=0)    
                pcd_est_frame_static = torch.cat(pcd_est_frame_static, dim=0)    
                dist_gt_frame_static = torch.cat(dist_gt_frame_static, dim=0)    
                dist_est_frame_static = torch.cat(dist_est_frame_static, dim=0)    

                pcd_gt_dynamic.append(pcd_gt_frame_dynamic)
                pcd_est_dynamic.append(pcd_est_frame_dynamic)
                dist_gt_dynamic.append(dist_gt_frame_dynamic)
                dist_est_dynamic.append(dist_est_frame_dynamic)
                pcd_gt_static.append(pcd_gt_frame_static)
                pcd_est_static.append(pcd_est_frame_static)
                dist_gt_static.append(dist_gt_frame_static)
                dist_est_static.append(dist_est_frame_static)


            dist_gt_static_all = torch.cat(dist_gt_static, dim=0).cpu().numpy()    
            dist_est_static_all = torch.cat(dist_est_static, dim=0).cpu().numpy()
            dist_gt_dynamic_all = torch.cat(dist_gt_dynamic, dim=0).cpu().numpy()    
            dist_est_dynamic_all = torch.cat(dist_est_dynamic, dim=0).cpu().numpy()
                
            for i in range(frame_num):
                dist1, dist2, idx1, idx2 = cd_metric(pcd_gt_dynamic[i][None],pcd_est_dynamic[i][None])
                dist1, dist2 = dist1**0.5, dist2**0.5
                cd_dynamic += dist1.mean().detach() + dist2.mean().detach()
                dist1, dist2, idx1, idx2 = cd_metric(pcd_gt_static[i][None],pcd_est_static[i][None])
                dist1, dist2 = dist1**0.5, dist2**0.5
                cd_static += dist1.mean().detach() + dist2.mean().detach()

                # median_dynamic += torch.abs(dist_gt_dynamic[i] - dist_est_dynamic[i]).median()
                # median_static += torch.abs(dist_gt_static[i] - dist_est_static[i]).median()
                
                # mean_dynamic += torch.abs(dist_gt_dynamic[i] - dist_est_dynamic[i]).mean()
                # mean_static += torch.abs(dist_gt_static[i] - dist_est_static[i]).mean()


            dynamic_error_list = np.abs(dist_est_dynamic_all - dist_gt_dynamic_all) * 100
            static_error_list =  np.abs(dist_est_static_all - dist_gt_static_all) * 100
            try:
                os.mkdir('ecdf')
            except FileExistsError:
                pass
            np.save('./ecdf/dynamic_error_list.npz', dynamic_error_list)
            np.save('./ecdf/static_error_list.npz', static_error_list)
            # median_dynamic /= (frame_num) / 100
            median_dynamic = np.median(dynamic_error_list)
            cd_dynamic /= (frame_num) / 100
            # mean_dynamic /= (frame_num) / 100
            mean_dynamic = dynamic_error_list.mean()

            # median_static /= (frame_num) / 100
            median_static = np.median(static_error_list)
            cd_static /= (frame_num) / 100
            # mean_static /= (frame_num) / 100
            mean_static = static_error_list.mean()



            print("mean_dynamic: ", mean_dynamic )
            print("median_dynamic: ", median_dynamic )
            print("cd_dynamic: ", cd_dynamic ) 
            print("mean_static: ", mean_static )
            print("median_static: ", median_static )
            print("cd_static: ", cd_static ) 
    def get_pcd(self, context_name):
        with torch.no_grad():
            phase = 'test'
            self.eval()
            batch_size = self.datamanager.test_loader.batch_size
            results=None
            batch_full = {}
            depth = []
            outputs = []
            outputs_full={}
            depth_vol = []
            batches = []
            self.model.tsfm_vehicle_eval = self.datamanager.test_dataset.tsfm_vehicle.cuda()
            self.model.aabb_vehicle = self.datamanager.test_dataset.aabb_vehicle
            for i in tqdm(range(50*2650*64 // batch_size)):
                ray_bundle, batch = self.datamanager.next_test(0)
                model_outputs = self.model(ray_bundle, batch)
                depth.append(model_outputs['depth_vol_c'].cpu())
                outputs.append({key: model_outputs[key].cpu() for key in model_outputs.keys()})
                batches.append({key: tensor.cpu() for key, tensor in batch[0].items()})

            for key, value in batch[0].items():
                batch_full[key] = torch.cat([b[key] for b in batches], dim=0)
            for key in model_outputs.keys():
                if key in ['wrong_intersection']: continue
                outputs_full[key] = torch.cat([b[key] for b in outputs], dim=0)
            depth_sdf_full = torch.cat([d for d in depth], dim=0) 
            batch_full['rays_o'] *= self.datamanager.test_dataset.extent
            batch_full['first_dist'] *= self.datamanager.test_dataset.extent
            outputs_full['depth_vol_c'] *= self.datamanager.test_dataset.extent
            depth_sdf_full *= self.datamanager.test_dataset.extent
            print(depth_sdf_full.shape)
            print(batch_full)
            try:
                os.mkdir(f'./pcd_out/')
            except FileExistsError:
                pass         
            try:
                os.mkdir(f'./pcd_out/{context_name}')
            except FileExistsError:
                pass
            torch.save(batch_full, f'./pcd_out/{context_name}/batch_full.pt') 
            torch.save(outputs_full, f'./pcd_out/{context_name}/outputs_full_active_intensity_raydrop_60000.pt') 


    def get_pcd_gt_full(self, context_name):
        with torch.no_grad():
            self.eval()
            batch_size = self.datamanager.test_loader.batch_size
            batch_full = {}
            batches = []
            self.model.tsfm_vehicle_eval = self.datamanager.test_dataset.tsfm_vehicle.cuda()
            for i in tqdm(range(50*2650*64 // batch_size)):
                ray_bundle, batch = self.datamanager.next_test(0)
                batches.append({key: tensor.cpu() for key, tensor in batch[0].items()})

            for key in ["rays_o", 'rays_d', 'first_dist', 'static_mask', 'vehicle_mask', 'first_intensity']:
                batch_full[key] = torch.cat([b[key] for b in batches], dim=0)
            batch_full['rays_o'] *= self.datamanager.test_dataset.extent
            batch_full['first_dist'] *= self.datamanager.test_dataset.extent
            print(batch_full)
            try:
                os.mkdir(f'./pcd_out/')
            except FileExistsError:
                pass         
            try:
                os.mkdir(f'./pcd_out/{context_name}_gt')
            except FileExistsError:
                pass
            torch.save(batch_full, f'./pcd_out/{context_name}_gt/batch_full.pt') 
    def get_dynamic_recall(self):
        with torch.no_grad():
            phase = 'test'
            self.eval()
            batch_size = self.datamanager.test_loader.batch_size
            results={}
            batch_full = {}
            model_outputs_full = {}
            depth = []
            depth_vol = []
            batches = []
            model_outputs_list = []
            for i in range(2650*64*10 // batch_size):
                ray_bundle, batch = self.datamanager.next_test(0)
                model_outputs = self.model(ray_bundle, batch)
                batches.append({key: tensor.cpu() for key, tensor in batch[0].items()})
                model_outputs_list.append({key: tensor.cpu() for key, tensor in model_outputs.items()})
                # if results is None:
                #     results = model_outputs
                # if batch_full is None:
                #     batch_full = batch    
                # else:
                #     results['coarse']['depth_vol_f'] = torch.cat((results['coarse']['depth_vol_f'].cpu(), value.cpu()), dim=0)
                #     for key, value in batch.items():
                #         batch_full[key] = torch.cat((batch_full[key].cpu(), value.cpu()), dim=0)      

            batch_full['vehicle_mask'] = torch.cat([b['vehicle_mask'] for b in batches], dim=0)  
            model_outputs_full['predicted_vehicle_mask'] = torch.cat([b['predicted_vehicle_mask'] for b in model_outputs_list], dim=0)  

            predicted_vehicle_mask = model_outputs_full['predicted_vehicle_mask']
            gt_vehicle_mask = batch_full['vehicle_mask']
            ones = torch.ones_like(predicted_vehicle_mask, requires_grad=False)
            intersect = torch.logical_and(predicted_vehicle_mask, gt_vehicle_mask)
            results['pos_recall'] = ones[intersect].sum() / ones[gt_vehicle_mask].sum() if ones[gt_vehicle_mask].sum() >0 else torch.tensor([1])
            results['pos_precision'] = ones[intersect].sum() / ones[predicted_vehicle_mask].sum() if ones[predicted_vehicle_mask].sum() >0 else torch.tensor([1])
            intersect = torch.logical_and(~predicted_vehicle_mask, ~gt_vehicle_mask)
            results['neg_recall'] = ones[intersect].sum() / ones[~gt_vehicle_mask].sum() if ones[~gt_vehicle_mask].sum()>0 else torch.tensor([1])
            results['neg_precision'] = ones[intersect].sum() / ones[~predicted_vehicle_mask].sum() if ones[~predicted_vehicle_mask].sum()>0 else torch.tensor([1])
            results['num_vehicle_points'] = ones[gt_vehicle_mask].sum()
            print(results)
    
    def load_pipeline(self, loaded_state: Dict[str, Any], step: int) -> None:
        """Load the checkpoint from the given path

        Args:
            loaded_state: pre-trained model state dict
            step: training step of the loaded checkpoint
        """
        state = {
            (key[len("module.") :] if key.startswith("module.") else key): value for key, value in loaded_state.items()
        }
        self.model.update_to_step(step)
        self.load_state_dict(state, strict=True)

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns the training callbacks from both the Dataloader and the Model."""
        datamanager_callbacks = self.datamanager.get_training_callbacks(training_callback_attributes)
        model_callbacks = self.model.get_training_callbacks(training_callback_attributes)
        callbacks = datamanager_callbacks + model_callbacks
        return callbacks

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the pipeline.

        Returns:
            A list of dictionaries containing the pipeline's param groups.
        """
        datamanager_params = self.datamanager.get_param_groups()
        model_params = self.model.get_param_groups()
        # TODO(ethan): assert that key names don't overlap
        return {**datamanager_params, **model_params}
