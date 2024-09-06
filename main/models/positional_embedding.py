"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn


@torch.no_grad()
class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        l, dim = x.shape[-2:]
        # assert dim == self.num_pos_feats and l <= 100, \
        #     f"the size of the pos embedding is not correct, {l} vs {100}"
        if mask is None:
            mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)
        not_mask = ~mask
        embed = not_mask.cumsum(1, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            embed = embed / (embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
 
        pos = embed[:, :, None] / dim_t
        pos = torch.stack((pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=3).flatten(2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.embed = nn.Embedding(500, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.embed.weight)

    def forward(self, x):
        l, dim = x.shape[-2:]
        assert dim == self.num_pos_feats and l <= 500, \
            f"the size of the pos embedding is not correct, {l} vs {500}"
        i = torch.arange(l, device=x.device)
        embeded = self.embed(i)
        pos = embeded.unsqueeze(0).repeat(x.shape[0], 1, 1)
        return pos


def build_position_encoding(type, embed_dim):
    N_steps = embed_dim
    if type in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif type in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {type}")

    return position_embedding
