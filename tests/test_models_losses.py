import pytest

torch = pytest.importorskip("torch")

from losses import (joint_ce_supcon, quality_weighted_supervised_contrastive_loss,
                    supervised_contrastive_loss)
from models import CNN1DClassifier, TCNClassifier


@pytest.mark.parametrize("klass", [TCNClassifier, CNN1DClassifier])
def test_model_shapes_and_batch_one(klass):
    kwargs = dict(in_channels=4, hidden_channels=8, projection_dim=6, num_classes=3)
    model = klass(**kwargs)
    result = model(torch.randn(1, 4, 16))
    assert result["feature_map"].shape == (1, 8, 16)
    assert result["embedding"].shape == (1, 8)
    assert result["projection"].shape == (1, 6)
    assert result["logits"].shape == (1, 3)


def test_supcon_no_positive_is_finite_and_backward_safe():
    features = torch.randn(3, 5, requires_grad=True)
    loss = supervised_contrastive_loss(features, torch.tensor([0, 1, 2]))
    assert torch.isfinite(loss); loss.backward(); assert features.grad is not None


def test_joint_loss_backward():
    logits = torch.randn(4, 2, requires_grad=True); projections = torch.randn(4, 6, requires_grad=True); labels = torch.tensor([0, 0, 1, 1])
    loss = joint_ce_supcon(logits, projections, labels, 0.2, 0.1)
    assert torch.isfinite(loss); loss.backward()


def test_quality_weighted_supcon_is_finite():
    features = torch.randn(8, 6, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
    weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.2, 0.8, 0.3, 0.9])
    loss = quality_weighted_supervised_contrastive_loss(features, labels, weights, 0.1)
    assert torch.isfinite(loss); loss.backward()
