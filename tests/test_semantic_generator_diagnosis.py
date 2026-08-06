import torch

from scripts.diagnose_semantic_generator_seeds import gradient_norm


def test_component_gradient_norm_is_finite_and_nonzero():
    parameter = torch.nn.Parameter(torch.tensor([2.0, -1.0]))
    loss = parameter.square().sum()
    value = gradient_norm(loss, [parameter])
    assert value > 0 and torch.isfinite(torch.tensor(value))
