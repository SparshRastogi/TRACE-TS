import numpy as np
import torch
import torch.nn as nn

__all__ = ["MixUpLoss", "mixup_data"]


class MixUpLoss(nn.Module):
    def __init__(self, crit, reduction="mean"):
        super().__init__()
        if hasattr(crit, "reduction"):
            self.crit = crit
            self.old_red = crit.reduction
            setattr(self.crit, "reduction", "none")
        self.reduction = reduction

    def forward(self, output, target):
        if len(target.size()) == 2:
            loss1, loss2 = (
                self.crit(output, target[:, 0].long()),
                self.crit(output, target[:, 1].long()),
            )
            d = loss1 * target[:, 2] + loss2 * (1 - target[:, 2])
        else:
            d = self.crit(output, target)
        if self.reduction == "mean":
            return d.mean()
        elif self.reduction == "sum":
            return d.sum()
        return d

    def get_old(self):
        if hasattr(self, "old_crit"):
            return self.old_crit
        elif hasattr(self, "old_red"):
            setattr(self.crit, "reduction", self.old_red)
            return self.crit


def mixup_data(x, y, alpha=0.4):
    batch_size = x.shape[0]
    lam = np.random.beta(alpha, alpha, batch_size)
    lam = np.concatenate([lam[:, None], 1 - lam[:, None]], 1).max(1)
    lam = x.new(lam)
    shuffle = torch.randperm(batch_size).cuda()
    x1, y1 = (x[shuffle], y[shuffle])
    out_shape = [lam.size(0)] + [1 for _ in range(len(x1.shape) - 1)]
    mixed_x = x * lam.view(out_shape) + x1 * (1 - lam).view(out_shape)
    y_a_y_b_lam = torch.cat(
        [y[:, None].float(), y1[:, None].float(), lam[:, None].float()], 1
    )
    return (mixed_x, y_a_y_b_lam)
