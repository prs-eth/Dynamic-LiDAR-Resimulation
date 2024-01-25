"""my_method/my_config.py"""

from nerfstudio.engine.schedulers import SchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from torch.optim import Optimizer, lr_scheduler
from NFLStudio.nflpipeline import NFLPipeline, NFLPipelineConfig
from nerfstudio.configs.experiment_config import *
from nerfstudio.engine.optimizers import AdamOptimizerConfig
# from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.schedulers import *

import sys
import os
max_num_iterations=60000
@dataclass
class MySchedulerConfig(SchedulerConfig):
    _target: Type = field(default_factory=lambda: MyScheduler)

class MyScheduler(Scheduler):
    config:MySchedulerConfig 
    def get_scheduler(self, optimizer: Optimizer, lr_init: float) -> lr_scheduler._LRScheduler:
        return lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / max_num_iterations, 1 ))

import torch
NFLStudio = MethodSpecification(
  config=TrainerConfig(
    method_name="NFLStudio",
    pipeline= NFLPipelineConfig(),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=0.005, eps = 1e-15),
            "scheduler": MySchedulerConfig(),
        },
    },
    max_num_iterations=max_num_iterations,
    steps_per_eval_batch=500,
    steps_per_eval_image=0,
    steps_per_eval_all_images=0,
    save_only_latest_checkpoint=True,
  ),
  description="Custom description",

)