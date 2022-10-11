"""
This file contains a wrapper for Video-Swin-Transformer so it can be properly used as a temporal encoder for MTTR.
"""
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from einops import rearrange
from models.swin_transformer import SwinTransformer3D
from utils.misc import NestedTensor


class VideoSwinTransformerBackbone(nn.Module):
    """
    A wrapper which allows using Video-Swin Transformer as a temporal encoder for MTTR.
    Check out video-swin's original paper at: https://arxiv.org/abs/2106.13230 for more info about this architecture.
    Only the 'tiny' version of video swin was tested and is currently supported in our project.
    Additionally, we slightly modify video-swin to make it output per-frame embeddings as required by MTTR (check our
    paper's supplementary for more details), and completely discard of its 4th block.
    """
    def __init__(self, backbone_pretrained, backbone_pretrained_path, train_backbone, running_mode, **kwargs):
        super(VideoSwinTransformerBackbone, self).__init__()
        # patch_size is (1, 4, 4) instead of the original (2, 4, 4).
        # this prevents swinT's original temporal downsampling so we can get per-frame features.
        swin_backbone = SwinTransformer3D(patch_size=(1, 4, 4), embed_dim=96, depths=(2, 2, 6, 2),
                                          num_heads=(3, 6, 12, 24), window_size=(8, 7, 7), drop_path_rate=0.1,
                                          patch_norm=True)
        if backbone_pretrained and running_mode == 'feature_extractor':
            print('Loading pretrained weights for Video-Swin-Transformer...')
            state_dict = torch.load(backbone_pretrained_path)['state_dict']
            # extract swinT's kinetics-400 pretrained weights and ignore the rest (prediction head etc.)
            state_dict = {k[9:]: v for k, v in state_dict.items() if 'backbone.' in k}

            # sum over the patch embedding weight temporal dim  [96, 3, 2, 4, 4] --> [96, 3, 1, 4, 4]
            patch_embed_weight = state_dict['patch_embed.proj.weight']
            patch_embed_weight = patch_embed_weight.sum(dim=2, keepdims=True)
            state_dict['patch_embed.proj.weight'] = patch_embed_weight
            swin_backbone.load_state_dict(state_dict)

        self.patch_embed = swin_backbone.patch_embed
        self.pos_drop = swin_backbone.pos_drop
        self.layers = swin_backbone.layers[:-1]
        self.downsamples = nn.ModuleList()
        for layer in self.layers:
            self.downsamples.append(layer.downsample)
            layer.downsample = None
        self.downsamples[-1] = None  # downsampling after the last layer is not necessary

        self.layer_output_channels = [swin_backbone.embed_dim * 2 ** i for i in range(len(self.layers))]
        self.train_backbone = train_backbone
        if not train_backbone:
            for parameter in self.parameters():
                parameter.requires_grad_(False)

    def forward(self, samples: NestedTensor):
        vid_frames = rearrange(samples.tensors, 't b c h w -> b c t h w')

        vid_embeds = self.patch_embed(vid_frames)
        vid_embeds = self.pos_drop(vid_embeds)
        layer_outputs = []  # layer outputs before downsampling
        for layer, downsample in zip(self.layers, self.downsamples):
            vid_embeds = layer(vid_embeds.contiguous())
            layer_outputs.append(vid_embeds)
            if downsample:
                vid_embeds = rearrange(vid_embeds, 'b c t h w -> b t h w c')
                vid_embeds = downsample(vid_embeds)
                vid_embeds = rearrange(vid_embeds, 'b t h w c -> b c t h w')
        layer_outputs = [rearrange(o, 'b c t h w -> t b c h w') for o in layer_outputs]

        outputs = []
        orig_pad_mask = samples.mask
        for l_out in layer_outputs:
            pad_mask = F.interpolate(orig_pad_mask.float(), size=l_out.shape[-2:]).to(torch.bool)
            outputs.append(NestedTensor(l_out, pad_mask))
        return outputs

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class Backbone(nn.Module):
    """ Backbone to extact features """
    def __init__(self, **kwargs):
        super(Backbone, self).__init__()
        self.backbone = VideoSwinTransformerBackbone(**kwargs)
    
    def forward(self, samples: NestedTensor):
        return self.backbone(samples)

def build_backbone(args):
    model = Backbone(**vars(args))
    model.to(args.device)
    return model