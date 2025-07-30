"""
deploy.py

Provide a lightweight server/client implementation for deploying OpenVLA models (through the HF AutoClass API) over a
REST API. This script implements *just* the server, with specific dependencies and instructions below.

Note that for the *client*, usage just requires numpy/json-numpy, and requests; example usage below!

Dependencies:
    => Server (runs OpenVLA model on GPU): `pip install uvicorn fastapi json-numpy`
    => Client: `pip install requests json-numpy`

Client (Standalone) Usage (assuming a server running on 0.0.0.0:8000):

```
import requests
import json_numpy
json_numpy.patch()
import numpy as np

action = requests.post(
    "http://0.0.0.0:8000/act",
    json={"image": np.zeros((256, 256, 3), dtype=np.uint8), "instruction": "do something"}
).json()

Note that if your server is not accessible on the open web, you can use ngrok, or forward ports to your client via ssh:
    => `ssh -L 8000:localhost:8000 ssh USER@<SERVER_IP>`
"""

import os.path

# ruff: noqa: E402
import json_numpy

json_numpy.patch()
import os
import json
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import draccus
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.common.utils.utils import get_safe_torch_device

import ipdb
import time


@parser.wrap()
def create_policy(cfg: TrainPipelineConfig):
    # ipdb.set_trace()
    cfg.validate()
    device=torch.device('cuda')
    # dataset = make_dataset(cfg)
    # ipdb.set_trace()
    meta = LeRobotDatasetMetadata(repo_id=cfg.dataset.repo_id, local_files_only=cfg.dataset.local_files_only)
    
    policy = make_policy(
        cfg=cfg.policy,
        device=device,
        # ds_meta=dataset.meta,
        ds_meta=meta,
    )
    return policy


# === Server Interface ===
class Pi0Server:
    def __init__(self, policy, device) -> Path:
        """
        A simple server for Pi0 models; exposes `/act` to predict an action for a given image + instruction.
            => Takes in {"state": np.ndarray, "image_high": np.ndarray, image_front": np.ndarray, image_left": np.ndarray, image_right": np.ndarray, "instruction": str}
            => Returns  {"action": np.ndarray}
        """
        self.device = device
        self.policy = policy

        # [Hacky] Load Dataset Statistics from Disk (if passing a path to a fine-tuned model)
        # if os.path.isdir(self.openvla_path):
        #     with open(Path(self.openvla_path) / "dataset_statistics.json", "r") as f:
        #         self.vla.norm_stats = json.load(f)

    def predict_action(self, payload: Dict[str, Any]) -> str:
        try:
            if double_encode := "encoded" in payload:
                # Support cases where `json_numpy` is hard to install, and numpy arrays are "double-encoded" as strings
                assert len(payload.keys()) == 1, "Only uses encoded payload!"
                payload = json.loads(payload["encoded"])

            # Parse payload components
            state, image_high, image_left, image_right, instruction = payload['state'], payload["image_high"], payload["image_left"], payload["image_right"], payload["instruction"]

            batch = {
                'observation.state': torch.from_numpy(state).to(self.device),
                'observation.images.cam_high': torch.from_numpy(image_high).to(self.device),
                'observation.images.cam_left_wrist': torch.from_numpy(image_left).to(self.device),
                'observation.images.cam_right_wrist': torch.from_numpy(image_right).to(self.device),
                'task':[instruction]
            }
            
            action = self.policy.select_action(batch).cpu().numpy()
            print(action)
            
            if double_encode:
                return JSONResponse(json_numpy.dumps(action))
            else:
                return JSONResponse(action)
        except:  # noqa: E722
            logging.error(traceback.format_exc())
            logging.warning(
                "Your request threw an error; make sure your request complies with the expected format:\n"
                "{'image': np.ndarray, 'instruction': str}\n"
                "You can optionally an `unnorm_key: str` to specific the dataset statistics you want to use for "
                "de-normalizing the output actions."
            )
            return "error"

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.predict_action)
        uvicorn.run(self.app, host=host, port=port)



def deploy(policy, host, port) -> None:
    server = Pi0Server(policy=policy, device=torch.device('cuda'))
    server.run(host=host, port=port)


if __name__ == "__main__":
    policy = create_policy()
    deploy(policy=policy, host="0.0.0.0", port=8000)

