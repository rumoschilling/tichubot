from __future__ import annotations

import torch
import torch.nn as nn

from training.model_gt import GT_DERIVE_TEMP, TichuGrandTichuPolicy


def init_gt_smartly(policy: TichuGrandTichuPolicy) -> None:
    """Initialize the derived GT policy to a mild ~20% call prior."""
    with torch.no_grad():
        # p = σ(d/TEMP): a ~20% prior means d = TEMP · logit(0.2).
        nn.init.constant_(policy.adv_head.bias, -1.38629436112 * GT_DERIVE_TEMP)
        nn.init.normal_(policy.adv_head.weight, std=0.01 * GT_DERIVE_TEMP)
