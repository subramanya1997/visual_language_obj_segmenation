import argparse
import torch
from trainer import Trainer
from feature_extractor import FeatureExtractor
import yaml
import os
import wandb
from utils.utils import get_logging

def run(args):
    # logger 
    logger = get_logging(name=__name__ ,level=args.logging_level) 
    logger.info(f'Runing...')
    with open(args.config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = {k: v['value'] for k, v in config.items()}
    config = {**config, **vars(args)}
    config = argparse.Namespace(**config)

    if args.running_mode == 'feature_extractor':
        feature_extractor = FeatureExtractor(config)
        feature_extractor.extract_features()
        return 

    trainer = Trainer(config)
    if config.running_mode == 'train':
        trainer.train()
    else: #test mode
        model_state_dict = torch.load(config.checkpoint_path)
        if 'model_state_dict' in model_state_dict.keys():
            model_state_dict = model_state_dict['model_state_dict']
        trainer.model.load_state_dict(model_state_dict, strict=True)
        trainer.evaluate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('MMVN training and evaluation')
    parser.add_argument('--config_path', '-c', required=True,
                        help='path to configuration file')
    parser.add_argument('--running_mode', '-rm', choices=['train', 'eval', 'feature_extractor'], required=True,
                        help="mode to run, either 'train' or 'eval'")
    parser.add_argument('--window_size', '-ws', type=int,
                        help='window length to use during training/evaluation.'
                             'note - in Refer-YouTube-VOS this parameter is used only during training, as'
                             ' during evaluation full-length videos (all annotated frames) are used.')
    parser.add_argument('--batch_size', '-bs', type=int, required=True,
                        help='training batch size per device')
    parser.add_argument('--eval_batch_size', '-ebs', type=int,
                        help='evaluation batch size per device. '
                             'if not provided training batch size will be used instead.')
    parser.add_argument('--checkpoint_path', '-ckpt', type=str, default='checkpoints/',
                        help='path of checkpoint file to load for evaluation purposes')
    parser.add_argument('--logging_level', '-ll', type=str, default='DEBUG')
    args = parser.parse_args()

    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size

    run(args=args)