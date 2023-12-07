# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Data parser for pre-prepared datasets for all cameras, with no additional processing needed
Optional fields - semantics, mask_filenames, cameras.distortion_params, cameras.times
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import numpy as np
import torch

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
    Semantics,
)
from nerfstudio.data.scene_box import SceneBox

from NFLStudio.waymo_dynamic import WaymoDynamic
# from NFLStudio.waymo_dynamic_shift import WaymoDynamic


@dataclass
class NFLDataParserConfig(DataParserConfig):
    """Minimal dataset config"""

    _target: Type = field(default_factory=lambda: NFLDataParser)
    """target class to instantiate"""
    context_name: str = '1005081002024129653_5313_150_5333_150'
    root_dir: str = "/scratch/hanfeng/LidarSimStudio/processed_data_dynamic"
    min_near: float = 2.75  # minimum near distance to sample rays in meters
    max_dist: float = 75 #maximum distance 
    scene_size: int = 50
    min_inc: float = -17.6
    max_inc: float = 2.4 # degress
    width: int = 64
    height: int = 2650
    extra_margin: float = 10.0  #10 meters 
    shift: list[float] = field(default_factory=lambda: [1.5, 1.5, 0.5])
    fp16: bool = True 
    normalize: bool = True 

@dataclass
class NFLDataparserOutputs(DataparserOutputs):
    """Minimal dataset config"""
    custom_dataset: torch.utils.data.Dataset = None 
    scene_box: SceneBox= SceneBox(torch.tensor([[-1,-1,-0.5],[1,1,0.5]]))



@dataclass
class NFLDataParser(DataParser):
    """Minimal DatasetParser"""

    config: NFLDataParserConfig    

    def __init__(self, config: NFLDataParserConfig):
        super().__init__(config)
        self.config = config

    def _generate_dataparser_outputs(self, split="train"):

        self.waymo_dataset = WaymoDynamic(split, self.config)


        dataparser_outputs = NFLDataparserOutputs(
            custom_dataset = self.waymo_dataset,
            image_filenames=None,
            cameras=None
        )
        return dataparser_outputs
