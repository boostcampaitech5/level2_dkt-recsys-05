import torch
import torch.nn as nn


# def get_criterion(pred, target, weights=None):
#     if weights is not None:
#         assert len(weights) == 2

#         loss = weights[1] * (target * torch.log(pred)) + weights[0] * (
#             (1 - target) * torch.log(1 - pred)
#         )
#     else:
#         loss = target * torch.log(pred) + (1 - target) * torch.log(1 - pred)

#     return torch.neg(loss)


def get_criterion(pred, target):
    loss = nn.BCELoss(reduction="none")
    return loss(pred, target)
