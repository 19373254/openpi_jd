# 1. 显式启用 GPU 设备
# import torch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

import tensorflow_datasets as tfds
raw_dataset_name = "libero_10_no_noops"
data_dir = "/home/ubuntu/shiyan/data/.cache/huggingface/your_hf_name_bak/libero"
tfds.load(raw_dataset_name, data_dir=data_dir, split="train")