import torch
from utils import get_device


def mask_nll_loss(inp, target, mask):
    """
    This loss function calculates the average negative log likelihood of the elements that
    correspond to a 1 in the mask tensor.
    :param inp:
    :param target:
    :param mask:
    :return:
    """
    n_total = mask.sum()
    cross_entropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = cross_entropy.masked_select(mask).mean()
    loss = loss.to(get_device())
    return loss, n_total.item()
