# Copyright (c) Subramanya N. Licensed under the Apache License 2.0. All Rights Reserved
from .rdetr import build as build_rdetr

def build_model(args):
    return build_rdetr(args)