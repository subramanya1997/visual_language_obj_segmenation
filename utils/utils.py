import os
from os import path
import datetime
import shutil
import torch
import logging


def flatten_temporal_batch_dims(outputs, targets):
    for k in outputs.keys():
        if isinstance(outputs[k], torch.Tensor):
            outputs[k] = outputs[k].flatten(0, 1)
        else:  # list
            outputs[k] = [i for step_t in outputs[k] for i in step_t]
    targets = [frame_t_target for step_t in targets for frame_t_target in step_t]
    return outputs, targets

def create_output_dir(config_path):
    output_dir_path = path.join('runs', datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p"))
    os.makedirs(output_dir_path, exist_ok=True)
    shutil.copyfile(src=config_path, dst=path.join(output_dir_path, 'config.yaml'))
    return output_dir_path

def create_checkpoint_dir(output_dir_path):
    checkpoint_dir_path = path.join(output_dir_path, 'checkpoints')
    os.makedirs(checkpoint_dir_path, exist_ok=True)
    return checkpoint_dir_path

def get_logging(name, level='INFO'):
    if level == 'INFO':
        logging.basicConfig(level=logging.INFO)
    elif level == 'DEBUG':
        logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(name)
    return logger

def to_device(sample, device):
    if isinstance(sample, torch.Tensor):
        sample = sample.to(device)
    elif isinstance(sample, tuple) or isinstance(sample, list):
        sample = [to_device(s, device) for s in sample]
    elif isinstance(sample, dict):
        sample = {k: to_device(v, device) for k, v in sample.items()}
    return sample