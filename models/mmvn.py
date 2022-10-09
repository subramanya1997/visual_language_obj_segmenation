import torch
import torch.nn.functional as F
from torch import nn
from utils.misc import NestedTensor

class MMVN(nn.Module):
    """ The main module of the Multimodal Tracking Transformer """
    def __init__(self, num_queries, mask_kernels_dim=8, aux_loss=False, **kwargs):
        super().__init__()
        d_model = 256
        self.instance_kernels_head = MLP(d_model, d_model, output_dim=mask_kernels_dim, num_layers=2)

    def forward(self, samples: NestedTensor, valid_indices, text_queries):

        return self.instance_kernels_head(samples)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def build(args):
    model = MMVN(**vars(args))
    model.to(args.device)
    return model, None, None