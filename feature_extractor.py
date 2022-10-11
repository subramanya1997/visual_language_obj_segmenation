import math
import sys
import os
from os import path
import shutil
import random
import numpy as np
import wandb
import torch
import torch.cuda.amp as amp
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from utils.utils import create_output_dir, create_checkpoint_dir, flatten_temporal_batch_dims, get_logging, to_device

from models import build_model
from datasets import build_dataset
from einops import rearrange

class FeatureExtractor:
    def __init__(self, config):
        # logger
        self.logger = get_logging(name=__name__ ,level=config.logging_level)
        self.logger.info(f'Initializing Feature Extractor...')

        self.config = config
        self.config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # fix the seed for reproducibility
        seed = config.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.logger.info(f'Using seed {seed}')
        self.logger.info(f'Using device {self.device}')

        model = build_model(config)
        self.model = model

        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f'Number of trainable parameters: {n_parameters}')

        dataset_train = build_dataset(subset_type='train', **vars(config))
        dataset_val = build_dataset(subset_type='test', **vars(config))

        self.data_loader_train = DataLoader(dataset=dataset_train, batch_size=config.batch_size, collate_fn=dataset_train.collator, 
                    pin_memory=True, shuffle=False)
        self.data_loader_val = DataLoader(dataset=dataset_val, batch_size=config.batch_size, collate_fn=dataset_val.collator,
                    pin_memory=True, shuffle=False)

        self.feature_save_path = path.join(self.config.dataset_path, 'Release', 'features')
        if not path.exists(self.feature_save_path):
            os.makedirs(self.feature_save_path)
        self.logger.info(f'Saving features to {self.feature_save_path}')

    def extract_features(self):
        self.model.eval()
        self.model.to(self.device)
        self.logger.info(f'Extracting features...')
        with torch.no_grad():
            for batch_dict in tqdm(self.data_loader_train):
                samples = batch_dict['samples']
                video_id = batch_dict['video_id']
                features = defaultdict(list)
                for _sample in samples:
                    _sample = _sample.to(self.device)
                    _features = self.model(_sample)
                    for i, _f in enumerate(_features):
                        features[i].append(_f.tensors)
                
                for i, _f in features.items():
                    _layer_features = torch.cat(features[i], dim=0).cpu()
                    self.save_features(_layer_features, video_id, layer=i)
            
            for batch_dict in tqdm(self.data_loader_val):
                samples = batch_dict['samples']
                video_id = batch_dict['video_id']
                features = defaultdict(list)
                for _sample in samples:
                    _sample = _sample.to(self.device)
                    _features = self.model(_sample)
                    for i, _f in enumerate(_features):
                        features[i].append(_f.tensors)
                
                for i, _f in features.items():
                    _layer_features = torch.cat(features[i], dim=0).cpu()
                    self.save_features(_layer_features, video_id, layer=i)

    def save_features(self, features, video_id, layer=0):
        save_path = path.join(self.feature_save_path ,f'{video_id}_{layer}.pt')
        torch.save(features, save_path)