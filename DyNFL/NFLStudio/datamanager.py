from nerfstudio.data.datamanagers.base_datamanager import * 
from NFLStudio.dataparser import NFLDataParser
from nerfstudio.data.utils.dataloaders import CacheDataloader
from torch.utils.data import DataLoader, Dataset
from NFLStudio import raymarching
# from nerfstudio
from NFLStudio.dataparser import (
    NFLDataParserConfig
)
@dataclass
class NFLDataManagerConfig(DataManagerConfig):
    """A basic data manager"""

    _target: Type = field(default_factory=lambda: NFLDataManager)
#     """Target class to instantiate."""
    dataparser_config: NFLDataParserConfig = NFLDataParserConfig()
#     """Specifies the dataparser used to unpack the data."""
    train_num_rays_per_batch: int = 2048
#     """Number of rays per batch to use per training iteration."""
    shuffle: bool = True
    num_workers: int = 8

    eval_num_rays_per_batch: int = 2048
    """Number of rays per batch to use per eval iteration."""

class NFLDataManager(DataManager):
    config: NFLDataManagerConfig
    train_dataset: Dataset
    eval_dataset: Dataset
    train_dataparser_outputs: DataparserOutputs
    train_pixel_sampler: Optional[PixelSampler] = None
    eval_pixel_sampler: Optional[PixelSampler] = None

    def __init__(
        self,
        config: NFLDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
    ):
        super().__init__()
        self.dataparser = NFLDataParser(config.dataparser_config)
        self.train_dataparser_outputs = self.dataparser.get_dataparser_outputs(split='train')
        self.eval_dataparser_outputs = self.dataparser.get_dataparser_outputs(split='val')
        
        self.aabb = self.train_dataparser_outputs.scene_box.aabb.flatten()
        self.train_dataset = self.train_dataparser_outputs.custom_dataset
        self.train_loader = DataLoader(self.train_dataset, 
                                      batch_size=config.train_num_rays_per_batch,
                                      shuffle=config.shuffle,
                                      num_workers=config.num_workers)
        self.iter_train_loader = iter(self.train_loader)
        self.eval_dataset = self.eval_dataparser_outputs.custom_dataset
        self.eval_loader = DataLoader(self.eval_dataset, 
                                      batch_size=config.eval_num_rays_per_batch,
                                      shuffle=True,
                                      num_workers=config.num_workers)
        self.iter_eval_loader = iter(self.eval_loader)

        self.config = config
        self.device = device
        self.eval_count=0
        self.train_count=0
        self.train_epoch = 1
        self.eval_epoch = 1

        self.test_dataparser_outputs =self.dataparser.get_dataparser_outputs(split='test') 
        self.test_dataset = self.test_dataparser_outputs.custom_dataset
        self.test_loader = DataLoader(self.test_dataset, 
                                    batch_size=2650*2,
                                    shuffle=False,
                                    num_workers=config.num_workers)
        self.iter_test_loader = iter(self.test_loader) 
    def next_train(self, step: int) -> Tuple[list[RayBundle], list[Dict]]:
        try:
            ray_batch = next(self.iter_train_loader)
        except StopIteration:
            self.iter_train_loader = iter(self.train_loader)
            ray_batch = next(self.iter_train_loader) 
            self.train_epoch+=1
        for key in self.train_dataset.float_args:
            ray_batch[key] = ray_batch[key].squeeze().float().to(self.device)
        for key in ray_batch.keys():
            ray_batch[key] = ray_batch[key].to(self.device)
        self.train_count += 1
        static_ray_batch = ray_batch 
        static_ray_bundles = RayBundle(
            origins=static_ray_batch['rays_o'],
            directions=static_ray_batch['rays_d'],
            nears=None,
            fars=None,
            times=static_ray_batch['time_stamp'][:,None],
            pixel_area=None
        )
        ray_batch_list = [static_ray_batch]
        ray_bundles_list = [static_ray_bundles]
        for i in range(self.train_dataset.dynamic_object_counter):
            mask = torch.logical_and(ray_batch['dynamic_vehicle_idx'] == i, ray_batch['vehicle_mask'].bool()) 
            dynamic_ray_batch = {}
            for key in ray_batch.keys():
                dynamic_ray_batch.update({key: ray_batch[key][mask]})
            ray_batch_list.append(dynamic_ray_batch)    
            ray_bundles_list.append(
                RayBundle(
                    origins=dynamic_ray_batch['rays_o'],
                    directions=dynamic_ray_batch['rays_d'],
                    nears=None,
                    fars=None,
                    times=None,
                    pixel_area=None
                )
            )
        return ray_bundles_list, ray_batch_list
    
    def next_eval(self, step: int) -> Tuple[list[RayBundle], list[Dict]]:
        try:
            ray_batch = next(self.iter_eval_loader)
        except StopIteration:
            self.iter_eval_loader = iter(self.eval_loader)
            ray_batch = next(self.iter_eval_loader) 
            self.eval_epoch+=1
        for key in self.train_dataset.float_args:
            ray_batch[key] = ray_batch[key].squeeze().float().to(self.device)
        for key in ray_batch.keys():
            ray_batch[key] = ray_batch[key].to(self.device)
        self.eval_count += 1
        static_ray_batch = ray_batch
        ray_batch_list = [static_ray_batch] + [{} for i in range(self.train_dataset.dynamic_object_counter)]
        ray_bundles_list = [RayBundle(
            origins=static_ray_batch['rays_o'],
            directions=static_ray_batch['rays_d'],
            nears=None,
            fars=None,
            times=None,
            pixel_area=None
        ) for i in range(1+self.train_dataset.dynamic_object_counter)]
        

        return ray_bundles_list, ray_batch_list

    def next_test(self, step: int) -> Tuple[RayBundle, Dict]:
        try:
            ray_batch = next(self.iter_test_loader)
        except StopIteration:
            self.iter_test_loader = iter(self.test_loader)
            ray_batch = next(self.iter_test_loader) 
        for key in self.train_dataset.float_args:
            ray_batch[key] = ray_batch[key].squeeze().float().to(self.device)
        for key in ray_batch.keys():
            ray_batch[key] = ray_batch[key].to(self.device)
        static_ray_batch = ray_batch 
        ray_batch_list = [static_ray_batch] + [{} for i in range(self.train_dataset.dynamic_object_counter)]
        ray_bundles_list = [RayBundle(
            origins=static_ray_batch['rays_o'],
            directions=static_ray_batch['rays_d'],
            nears=None,
            fars=None,
            times=None,
            pixel_area=None
        ) for i in range(1+self.train_dataset.dynamic_object_counter)]
        

        return ray_bundles_list, ray_batch_list
        
    def get_train_rays_per_batch(self) -> int:
        return self.config.train_num_rays_per_batch

    def get_eval_rays_per_batch(self) -> int:
        return self.config.eval_num_rays_per_batch
        



        
    