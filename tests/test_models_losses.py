import pytest

torch = pytest.importorskip("torch")

from losses import (joint_ce_supcon, quality_weighted_supervised_contrastive_loss,
                    supervised_contrastive_loss)
from losses.supcon import _weighted_positive_mean
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


def test_quality_one_is_numerically_hard_supcon():
    torch.manual_seed(11)
    features = torch.randn(8, 6)
    labels = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
    hard = supervised_contrastive_loss(features, labels, 0.1)
    weighted = quality_weighted_supervised_contrastive_loss(features, labels, torch.ones(8), 0.1)
    assert torch.allclose(hard, weighted, atol=1e-7, rtol=1e-6)


def test_lower_quality_reduces_corresponding_positive_gradient_contribution():
    log_probability = torch.tensor([[0.0, -0.5, -1.0]], requires_grad=True)
    positive = torch.tensor([[False, True, True]])
    mean, valid = _weighted_positive_mean(log_probability, positive, torch.tensor([1.0, 1.0, 0.2]))
    assert valid.tolist() == [True]
    mean.sum().backward()
    assert abs(log_probability.grad[0, 2]) < abs(log_probability.grad[0, 1])


def test_zero_quality_is_finite_and_backward_safe():
    features = torch.randn(4, 5, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1])
    loss = quality_weighted_supervised_contrastive_loss(features, labels, torch.zeros(4), 0.1)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(features.grad).all()


def test_nonfinite_quality_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        quality_weighted_supervised_contrastive_loss(
            torch.randn(4, 5), torch.tensor([0, 0, 1, 1]),
            torch.tensor([1.0, float("nan"), 1.0, 1.0]), 0.1,
        )
