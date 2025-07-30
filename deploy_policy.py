#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
import os
import json
import sys
import jax
import numpy as np
from openpi.models import model as _model
from openpi.policies import aloha_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

import cv2
from PIL import Image

from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

class PI0:
    def __init__(self, model_path, task_name):
        self.train_config_name = task_name
        self.task_name = task_name
        self.model_path = model_path

        config = _config.get_config(self.train_config_name)
        self.policy = _policy_config.create_trained_policy(config, model_path)
        print("loading model success!")
        self.img_size = (224,224)
        self.observation_window = None
        self.random_set_language()

    # set img_size
    def set_img_size(self,img_size):
        self.img_size = img_size
    
    # set language randomly
    def random_set_language(self):
        json_Path =f"policy/custom_policy/instructions/{self.task_name}.json"
        with open(json_Path, 'r') as f_instr:
            instruction_dict = json.load(f_instr)
        instructions = instruction_dict['instructions']
        instruction = np.random.choice(instructions)
        self.instruction = instruction
        print(f"successfully set instruction:{instruction}")
    
    # Update the observation window buffer
    def update_observation_window(self, img_arr, state):
        img_front, img_right, img_left, puppet_arm = img_arr[0], img_arr[1], img_arr[2], state
        img_front = np.transpose(img_front, (2, 0, 1))
        img_right = np.transpose(img_right, (2, 0, 1))
        img_left = np.transpose(img_left, (2, 0, 1))

        self.observation_window = {
            "state": state,
            "images": {
                "cam_high": img_front,
                "cam_left_wrist": img_left,
                "cam_right_wrist": img_right,
            },
            "prompt": self.instruction,
        }
    def get_action(self, img_arr=None, state=None):
        assert (img_arr is None) ^ (state is None) == False, "input error"
        if (img_arr is not None) and (state is not None):
            self.update_observation_window(img_arr, state)

        return self.policy.infer(self.observation_window)["actions"]

    # def get_action(self):
    #     assert (self.observation_window is not None), "update observation_window first!"
    #     return self.policy.infer(self.observation_window)["actions"]

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        print("successfully unset obs and language intruction")

def get_model(ckpt_folder, task_name):
    print('ckpt_folder: ', ckpt_folder)
    # model_path = os.path.join('checkpoints', ckpt_folder)
    # config_file_path = Path(__file__).parent.joinpath(f'3D-Diffusion-Policy/diffusion_policy_3d/config/robot_dp3.yaml')
    # with open(config_file_path, 'r') as config_file:
    #     config = yaml.safe_load(config_file)
    # task_config_path = Path(__file__).parent.joinpath(f'3D-Diffusion-Policy/diffusion_policy_3d/config/task/{task_name}.yaml')
    # with open(task_config_path, 'r') as task_config:
    #     config['task'] = yaml.safe_load(task_config)
    # config = OmegaConf.create(config)
    return PI0(ckpt_folder, task_name)


def reset_model(model):
    model.reset_obsrvationwindows()
    model.random_set_language()

def eval(TASK_ENV, model, observation):
    observation['observation']['head_camera']['rgb'] = observation['observation']['head_camera']['rgb'][:,:,::-1]
    observation['observation']['left_camera']['rgb'] = observation['observation']['left_camera']['rgb'][:,:,::-1]
    observation['observation']['right_camera']['rgb'] = observation['observation']['right_camera']['rgb'][:,:,::-1]
    obs = TASK_ENV.get_cam_obs(observation)
    obs['agent_pos'] = observation['joint_action']
    
    input_rgb_arr, input_state = [observation['observation']['head_camera']['rgb'], observation['observation']['right_camera']['rgb'], observation['observation']['left_camera']['rgb']], obs['agent_pos'] 
    # input_state[6] /= 0.045 
    # input_state[13] /= 0.045
    # model.update_observation_window(input_rgb_arr, input_state)

    actions = model.get_action(input_rgb_arr, input_state)
    take_actions = actions[:10]

    for action in take_actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        observation['observation']['head_camera']['rgb'] = observation['observation']['head_camera']['rgb'][:,:,::-1]
        observation['observation']['left_camera']['rgb'] = observation['observation']['left_camera']['rgb'][:,:,::-1]
        observation['observation']['right_camera']['rgb'] = observation['observation']['right_camera']['rgb'][:,:,::-1]
        obs = TASK_ENV.get_cam_obs(observation)
        obs['agent_pos'] = observation['joint_action']
        
        input_rgb_arr, input_state = [observation['observation']['head_camera']['rgb'], observation['observation']['right_camera']['rgb'], observation['observation']['left_camera']['rgb']], obs['agent_pos'] 
        # input_state[6] /= 0.045 
        # input_state[13] /= 0.045
        model.update_observation_window(input_rgb_arr, input_state)