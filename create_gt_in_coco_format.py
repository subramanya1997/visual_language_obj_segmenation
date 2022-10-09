"""
This script converts the ground-truth annotations of the a2d-sentences dataset to COCO format (for mAP calculation).
This results in a ground-truth JSON file which can be loaded using the pycocotools API.
Note that during evaluation model predictions need to be converted to COCO format as well (check out trainer.py).
"""

import argparse
from os import path
from datasets.a2d_sentences.create_a2d_gt_in_coco_format import create_a2d_sentences_ground_truth_test_annotations

if __name__ == '__main__':
    parser = argparse.ArgumentParser('a2d sentence dataset')
    parser.add_argument('--dataset_name', '-dn', default='a2d_sentences', type=str, help='name of the dataset')
    parser.add_argument('--subset_type', '-st', default='test', type=str, help='train/val/test')
    parser.add_argument('--dataset_path', '-dp', default='./data/a2d_sentences', type=str, help='path to a2d dataset')
    parser.add_argument('--output_path', '-op', default='./data/a2d_sentences/', type=str, help='path to dir to save the json in coco format')
    args = parser.parse_args()

    output_json_path =  path.join(args.output_path, f'{args.dataset_name}_{args.subset_type}_annotations_in_coco_format.json')

    if args.dataset_name in {'a2d_sentences'}:
        if args.dataset_name == 'a2d_sentences':
            pass
            create_a2d_sentences_ground_truth_test_annotations(subset_type=args.subset_type, dataset_path=args.dataset_path, output_path=output_json_path)
    else:
        raise ValueError(f'Dataset {args.dataset_name} not supported')
