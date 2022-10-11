"""
This file contains a Trainer class which handles the training and evaluation of MTTR.
"""

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
from utils.utils import create_output_dir, create_checkpoint_dir, flatten_temporal_batch_dims, get_logging, to_device

from models import build_model
from datasets import build_dataset

class Trainer:
    def __init__(self, config):
        # logger
        self.logger = get_logging(name=__name__ ,level=config.logging_level)
        self.logger.info(f'Initializing trainer...')
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

        model, criterion, postprocessor = build_model(config)

        self.model = model
        self.criterion = criterion
        self.postprocessor = postprocessor

        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f'Number of trainable parameters: {n_parameters}')

        self.dataset_name = self.config.dataset_name
        if self.dataset_name == 'a2d_sentences' or self.dataset_name == 'jhmdb_sentences':
            self.evaluate = self.evaluate_a2d_sentences
        else:
            assert False, f'Error: dataset {self.dataset_name} is not supported'

        dataset_train = build_dataset(subset_type='train', **vars(config))
        dataset_val = build_dataset(subset_type='test', **vars(config))
        self.data_loader_train = DataLoader(dataset=dataset_train, batch_size=config.batch_size, collate_fn=dataset_train.collator, 
                    pin_memory=True, shuffle=True)
        self.data_loader_val = DataLoader(dataset=dataset_val, batch_size=config.batch_size, collate_fn=dataset_val.collator,
                    pin_memory=True, shuffle=False)

        # Optimizer, LR-Scheduler, AMP Grad Scaler:
        param_dicts = [
            {"params": [p for n, p in self.model.named_parameters() if "backbone" not in n and "text_encoder" not in n and p.requires_grad]},
            {"params": [p for n, p in self.model.named_parameters() if "backbone" in n and p.requires_grad], "lr": config.lr_backbone},
            {"params": [p for n, p in self.model.named_parameters() if "text_encoder" in n and p.requires_grad], "lr": config.text_encoder_lr},
        ]

        self.optimizer = torch.optim.AdamW(params=param_dicts, lr=config.lr, weight_decay=config.weight_decay)
        self.lr_scheduler = MultiStepLR(optimizer=self.optimizer, milestones=[50], gamma=0.4, verbose=True)
        self.grad_scaler = amp.GradScaler(enabled=config.enable_amp)
        self.max_norm = config.clip_max_norm

        self.output_dir_path = create_output_dir(config_path=self.config.config_path)
        self.checkpoint_dir_path = create_checkpoint_dir(output_dir_path=self.output_dir_path)
        # init wandb
        self.init_wandb(config=self.config)
        
        self.total_epochs = config.epochs
        self.epoch = 0
        self.iteration = 0
        self.best_mAP = 0
        self.best_loss = math.inf

    def init_wandb(self, config):
        pass
    
    def train(self):
        self.logger.info(f'Starting training...')
        for epoch in range(self.epoch, self.total_epochs):
            self.model.train()
            # self.criterion.train()

            total_epoch_loss = 0
            # loss_sums_dict = {k: 0 for k in self.criterion.weight_dict.keys()}
            for batch_dict in tqdm(self.data_loader_train, desc=f'Epoch {epoch+1}/{self.total_epochs}'):
                samples = batch_dict['samples'].to(self.device)
                targets = to_device(batch_dict['targets'], self.device)
                text_queries = batch_dict['text_queries']

                # keep only the valid targets (targets of frames which are annotated). for example, in a2d-sentences
                # only the center frame in each window is annotated.
                valid_indices = torch.tensor([i for i, t in enumerate(targets) if None not in t]).to(self.device)
                targets = [targets[i] for i in valid_indices.tolist()]

                print( f'Sample: {samples.tensors.shape} {samples.mask.shape}' )
                print( f'targets: {valid_indices}' )
                print( f'Sample: {text_queries}' )

                break
            break
        pass
    
    def evaluate(self):
        self.logger.info(f'Starting evaluation...')

        pass

    @torch.no_grad()
    def evaluate_a2d_sentences(self):
        pass
