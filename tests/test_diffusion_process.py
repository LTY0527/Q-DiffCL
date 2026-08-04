import torch

from diffusion import DiffusionSchedule, ddpm_restore
from models import MinimalConditionalDiffusion1D
from scripts.train_diffusion_recovery import (diffusion_objective,
                                                small_subset_gate_checks,
                                                tiny_gate_checks)


class ZeroNoiseModel(torch.nn.Module):
    def forward(self, noisy, degraded, observation_mask, timesteps):
        return torch.zeros_like(noisy)


def test_q_sample_and_x0_round_trip():
    schedule = DiffusionSchedule.cosine(20, "cpu")
    clean = torch.randn(3, 2, 8); noise = torch.randn_like(clean); timesteps = torch.tensor([0, 7, 19])
    noisy = schedule.q_sample(clean, timesteps, noise)
    reconstructed = schedule.predict_x0(noisy, timesteps, noise)
    assert torch.allclose(reconstructed, clean, atol=2e-4)
    assert schedule.alpha_bars[-1] < 1e-3


def test_final_posterior_step_matches_predicted_x0():
    schedule = DiffusionSchedule.cosine(10, "cpu"); clean = torch.randn(2, 2, 6); noise = torch.randn_like(clean)
    timesteps = torch.zeros(2, dtype=torch.long); noisy = schedule.q_sample(clean, timesteps, noise)
    predicted_clean = schedule.predict_x0(noisy, timesteps, noise)
    final = schedule.posterior_step_from_x0(noisy, timesteps, predicted_clean, torch.randn_like(clean))
    assert torch.allclose(final, clean, atol=1e-5)


def test_mask_semantics_and_observed_clamping():
    schedule = DiffusionSchedule.cosine(8, "cpu"); degraded = torch.randn(2, 3, 10)
    observation = torch.rand_like(degraded) > 0.3
    restored = ddpm_restore(ZeroNoiseModel(), degraded, observation, schedule, torch.Generator().manual_seed(4))
    assert torch.equal(restored[observation], degraded[observation])


def test_sampling_is_deterministic_for_same_generator_seed():
    schedule = DiffusionSchedule.cosine(8, "cpu"); degraded = torch.zeros(2, 2, 6); observation = torch.rand_like(degraded) > 0.5
    first = ddpm_restore(ZeroNoiseModel(), degraded, observation, schedule, torch.Generator().manual_seed(9))
    second = ddpm_restore(ZeroNoiseModel(), degraded, observation, schedule, torch.Generator().manual_seed(9))
    assert torch.equal(first, second)


def test_one_batch_fixed_noise_overfit_reduces_objective():
    torch.manual_seed(3); schedule = DiffusionSchedule.cosine(12, "cpu")
    model = MinimalConditionalDiffusion1D(2, hidden=16, time_dimension=16, blocks=2)
    clean = torch.randn(4, 2, 12); observation = torch.rand_like(clean) > 0.3
    degraded = torch.where(observation, clean, torch.zeros_like(clean)); timesteps = torch.full((4,), 5, dtype=torch.long); noise = torch.randn_like(clean)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3); values = []
    for _ in range(80):
        optimizer.zero_grad(); total, _, _, _ = diffusion_objective(model, schedule, clean, degraded, observation, timesteps, noise, 0.1)
        total.backward(); optimizer.step(); values.append(float(total.detach()))
    assert values[-1] < values[0] * 0.25


def test_gate_decisions_follow_majority_rules():
    methods = {"degraded": {"masked_mae": 0.6}, "simple": {"masked_mae": 0.3},
               "diffusion": {"masked_mae": 0.4, "observed_mae": 0.0}}
    tiny = tiny_gate_checks(3.0, methods, [0.40, 0.41, 0.40], [{"masked_mae": 2.0}, {"masked_mae": 0.4}])
    assert sum(tiny.values()) >= 4 and not tiny["meets_suggested_simple_mae_target"]
    task = {
        "simple": {"metrics": {"fault_recall": 0.82, "far": 0.13, "auprc": 0.93}, "teacher_prediction_consistency": 0.91},
        "diffusion": {"metrics": {"fault_recall": 0.85, "far": 0.22, "auprc": 0.92}, "teacher_prediction_consistency": 0.87},
    }
    small = small_subset_gate_checks(methods, task, [{"masked_mae": 2.0}, {"masked_mae": 0.4}])
    assert sum(small.values()) >= 4 and small["at_least_one_task_metric_better_than_simple"]
