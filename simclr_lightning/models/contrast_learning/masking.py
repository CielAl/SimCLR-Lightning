import torch
from torch import nn
from typing import Optional


def feat_masking_helper(feat: torch.Tensor, feat_mask: torch.Tensor):
    assert isinstance(feat, torch.Tensor)
    img_size = feat_mask.shape[-2:]
    feat_size = feat.shape[-2:]
    factor = img_size[0] // feat_size[0]
    shuffled: torch.Tensor = nn.PixelUnshuffle(factor)(feat_mask).to(feat.device)
    hw_mask = torch.any(shuffled, dim=1, keepdim=True)
    return hw_mask * feat


def feat_masking(feat_map: torch.Tensor, feat_mask: Optional[torch.Tensor] = None):
    if feat_mask is None:
        return feat_map
    return feat_masking_helper(feat=feat_map, feat_mask=feat_mask)
