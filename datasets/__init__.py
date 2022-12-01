# Copyright (c) Subramanya N. Licensed under the Apache License 2.0. All Rights Reserved
import torch.utils.data
import torchvision

from datasets.mixed import CustomCocoDetection
from datasets.coco import build as build_coco_dataset
from datasets.lvis import build as build_lvis_dataset
from datasets.lvis import LvisDetectionBase
from datasets.lvis_modulation import build as build_modulated_lvis


def get_coco_api_from_dataset(dataset):
    """Hacky way to get the COCO api object from a dataset."""
    for _ in range(10):
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, LvisDetectionBase):
        return dataset.lvis
    if isinstance(dataset, (torchvision.datasets.CocoDetection, CustomCocoDetection)):
        return dataset.coco
    
def build_dataset(dataset_file: str, image_set: str, args):
    if dataset_file == 'coco':
        return build_coco_dataset(image_set, args)
    if dataset_file == 'lvis':
        return build_lvis_dataset(image_set, args)
    if dataset_file == 'modulated_lvis':
        return build_modulated_lvis(image_set, args)
