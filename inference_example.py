#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import torch

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.common.utils.utils import get_safe_torch_device

import ipdb
import time
import os

# os.environ['LEROBOT_HOME']='~/.cache/huggingface/lerobot'
@parser.wrap()
def train(cfg: TrainPipelineConfig):
    # ipdb.set_trace()
    cfg.validate()
    device=torch.device('cuda')
    # dataset = make_dataset(cfg)
    # ipdb.set_trace()
    meta = LeRobotDatasetMetadata(repo_id=cfg.dataset.repo_id, local_files_only=cfg.dataset.local_files_only)
    # ipdb.set_trace()
    policy = make_policy(
        cfg=cfg.policy,
        device=device,
        # ds_meta=dataset.meta,
        ds_meta=meta,
    )
    batch = {
        'observation.state': torch.randn((1,14)).to(device),
        'observation.images.cam_high': torch.randn((1,3, 480,640)).to(device),
        'observation.images.cam_left_wrist': torch.randn((1, 3, 480,640)).to(device),
        'observation.images.cam_right_wrist': torch.randn((1, 3, 480,640)).to(device),
        'task':['DEBUG']
    }
    
    # ipdb.set_trace()
    for _ in range(51):
        start = time.time()
        action = policy.select_action(batch)
        end = time.time()
        print(action)
        print(f'Inference time:{round(end - start, 5)}s')


if __name__ == "__main__":
    train()
