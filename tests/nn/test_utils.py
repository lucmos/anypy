from typing import Tuple

import pytest
import torch

from anypy.nn.utils import calculate_adaptive_weight


@pytest.mark.parametrize("dim", [10, 100, 1000])
def test_calculate_adaptive_weight(dim: Tuple[int, int, int]):
    layer = torch.randn(dim, requires_grad=True, dtype=torch.double)

    linear_a = torch.randn((dim, dim), dtype=torch.double)
    linear_b = torch.randn((dim, dim), dtype=torch.double)

    loss_a = torch.sum(linear_a @ layer)
    loss_b = torch.sum((linear_b @ layer) ** 2)

    loss_a_gradnorm = torch.norm(torch.autograd.grad(loss_a, layer, retain_graph=True)[0])
    loss_b_gradnorm = torch.norm(torch.autograd.grad(loss_b, layer, retain_graph=True)[0])
    assert not torch.allclose(loss_a_gradnorm, loss_b_gradnorm)

    loss_b_weight = calculate_adaptive_weight(loss_a, loss_b, layer)
    loss_b_weighted = loss_b_weight * loss_b
    loss_b_weighted_gradnorm = torch.norm(torch.autograd.grad(loss_b_weighted, layer, retain_graph=True)[0])
    assert torch.allclose(loss_a_gradnorm, loss_b_weighted_gradnorm)
