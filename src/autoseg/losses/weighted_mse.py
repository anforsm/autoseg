import torch


class WeightedMSELoss(torch.nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def _calc_loss(self, pred, target, weights):
        scale = weights * (pred - target) ** 2

        if len(torch.nonzero(scale)) != 0:
            mask = torch.masked_select(scale, torch.gt(weights, 0))
            loss = torch.mean(mask)

        else:
            loss = torch.mean(scale)

        return loss

    def forward(self, affs_prediction, affs_target, affs_weights):
        loss = self._calc_loss(affs_prediction, affs_target, affs_weights)

        return loss