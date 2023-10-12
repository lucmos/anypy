import logging

import torch
from torch import Tensor, nn

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


def calculate_adaptive_weight(
    loss_a: Tensor,
    loss_b: Tensor,
    layer: Tensor,
    min: float = 0.0,
    max: float = 1e6,
    eps: float = 1e-4,
    fallback: float = 0.0,
) -> float:
    """Compute the re-scaling factor to apply to loss_b.

    The rescaling factor is computed such taht the gradients of both the loss terms
    wrt the specified layer have the same norm.

    Args:
        loss_a: first loss term
        loss_b: second loss term
        layer: compute the gradients of loss_a and loss_b with respect to this tensor
        min: minimum value of the scaling factor
        max: maximum value of the scaling factor
        eps: epsilon value to avoid division by zero
        fallback: value to return in case of autograd error

    Returns:
        the scaling factor to apply to loss_b,
        in order to have the same gradient norm of loss_a in the layer
    """
    try:
        loss_a_grads = torch.autograd.grad(loss_a, layer, retain_graph=True)[0]
        loss_b_grads = torch.autograd.grad(loss_b, layer, retain_graph=True)[0]

        loss_b_weight = torch.norm(loss_a_grads) / (torch.norm(loss_b_grads) + eps)
        loss_b_weight = torch.clamp(loss_b_weight, min=min, max=max).detach()
    except RuntimeError:
        pylogger.error("Error in calculating adaptive weight")
        loss_b_weight = torch.tensor(fallback)
    return loss_b_weight
