import json
import torch
import numpy as np
from torchvision.io import read_video
import h5py
from torch.utils.data import Dataset
import torch.distributed as dist
import torchvision.transforms.functional as F
import pandas
from os import path
from glob import glob
from tqdm import tqdm
import datasets.transforms as T
from pycocotools.mask import encode, area
from utils.misc import nested_tensor_from_videos_list
from datasets.a2d_sentences.create_a2d_gt_in_coco_format import create_a2d_sentences_ground_truth_test_annotations
from .utils import get_image_id
from utils.utils import get_logging


class A2DSentencesDatasetFeature(Dataset):
    """
    A Torch dataset for A2D-Sentences.
    For more information check out: https://kgavrilyuk.github.io/publication/actor_action/ or the original paper at:
    https://arxiv.org/abs/1803.07485
    """
    def __init__(self, subset_type: str = 'train', dataset_path: str = './a2d_sentences', window_size=8,
                 dataset_coco_gt_format_path=None, logging_level='INFO', **kwargs):
        super(A2DSentencesDatasetFeature, self).__init__()
        # logging
        self.logger = get_logging(name=__name__, level=logging_level)
        assert subset_type in ['train', 'test'], 'error, unsupported dataset subset type. supported: train, test'
        self.logger.info(f'Loading {subset_type} dataset from {dataset_path}...')
        self.subset_type = subset_type
        self.mask_annotations_dir = path.join(dataset_path, 'text_annotations/a2d_annotation_with_instances')
        self.videos_dir = path.join(dataset_path, 'Release/clips320H')
        self.vidos_ids = self.get_videos(root_path=dataset_path, subset=subset_type)
        self.window_size = window_size
        self.transforms = A2dSentencesTransforms(subset_type, **kwargs)
        self.collator = Collator()

    @staticmethod
    def get_videos(root_path, subset):
        saved_annotations_file_path = path.join(root_path, f'a2d_sentences_single_frame_{subset}_annotations.json')
        if path.exists(saved_annotations_file_path):
            with open(saved_annotations_file_path, 'r') as f:
                text_annotations_by_frame = [tuple(a) for a in json.load(f)]
                #get all videos
                vidos_ids = list(set([a[1] for a in text_annotations_by_frame]))
                return vidos_ids

    def __getitem__(self, idx):
        video_id = self.vidos_ids[idx]

        # read the source window frames:
        video_frames, _, _ = read_video(path.join(self.videos_dir, f'{video_id}.mp4'), pts_unit='sec')  # (T, H, W, C)
    
        # extract the window source frames:
        source_frames = []
        for i in range(0, len(video_frames)):
            i = min(max(i, 0), len(video_frames)-1)  # pad out of range indices with edge frames
            source_frames.append(F.to_pil_image(video_frames[i].permute(2, 0, 1)))

        targets = len(source_frames) * [None]
        source_frames, _ = self.transforms(source_frames, targets)
        return source_frames, self.window_size, video_id

    def __len__(self):
        return len(self.vidos_ids)


class A2dSentencesTransforms:
    def __init__(self, subset_type, horizontal_flip_augmentations, resize_and_crop_augmentations,
                 train_short_size, train_max_size, eval_short_size, eval_max_size, **kwargs):
        self.h_flip_augmentation = subset_type == 'train' and horizontal_flip_augmentations
        normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        scales = [train_short_size]  # no more scales for now due to GPU memory constraints. might be changed later
        transforms = []
        if resize_and_crop_augmentations:
            if subset_type == 'train':
                transforms.append(T.RandomResize(scales, max_size=train_max_size))
            elif subset_type == 'test':
                transforms.append(T.RandomResize([eval_short_size], max_size=eval_max_size)),
        transforms.extend([T.ToTensor(), normalize])
        self.size_transforms = T.Compose(transforms)

    def __call__(self, source_frames, targets):
        if self.h_flip_augmentation and torch.rand(1) > 0.5:
            source_frames = [F.hflip(f) for f in source_frames]
        source_frames, targets = list(zip(*[self.size_transforms(f, t) for f, t in zip(source_frames, targets)]))
        source_frames = torch.stack(source_frames)  # [T, 3, H, W]
        return source_frames, targets


class Collator:
    def __call__(self, batch):
        samples, window_size, video_id = list(zip(*batch))
        _samples = []
        for i in range(0, samples[0].shape[0], window_size[0]):
            _samples.append(nested_tensor_from_videos_list((samples[0][i:i+window_size[0]],)))
        batch_dict = {
            'samples': _samples,
            'video_id': video_id[0]
        }
        return batch_dict