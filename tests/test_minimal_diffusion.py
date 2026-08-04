import torch

from models import MinimalConditionalDiffusion1D


def test_minimal_diffusion_shape_and_backward():
    model = MinimalConditionalDiffusion1D(4, hidden=16, time_dimension=16, blocks=2)
    clean = torch.randn(2, 4, 12); degraded = clean.clone(); mask = torch.rand_like(clean) > 0.3
    output = model(clean, degraded, mask, torch.tensor([0, 5]))
    assert output.shape == clean.shape and torch.isfinite(output).all()
    output.square().mean().backward()
