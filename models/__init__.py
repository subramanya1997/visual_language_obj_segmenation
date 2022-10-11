from .mmvn import build
from .backbone import build_backbone

def build_model(args):
    if args.running_mode == 'feature_extractor':
        return build_backbone(args)
    return build(args)