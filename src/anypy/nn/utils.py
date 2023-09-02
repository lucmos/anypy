import logging

import torch
from torch import nn

pylogger = logging.getLogger(__name__)


def infer_dimension(width: int, height: int, n_channels: int, model: nn.Module, batch_size: int = 8) -> torch.Tensor:
    """Compute the output of a model given a fake batch.

    Args:
        width: the width of the image to generate the fake batch
        height:  the height of the image to generate the fake batch
        n_channels:  the n_channels of the image to generate the fake batch
        model: the model to use to compute the output
        batch_size: batch size to use for the fake output

    Returns:
        the fake output
    """
    with torch.no_grad():
        model.eval()
        fake_batch = torch.zeros([batch_size, n_channels, width, height])
        fake_out = model(fake_batch)
        model.train()
        return fake_out
