import numpy as np
import torch

from recipe._psrnsr.config import AlgoConfig
from recipe._psrnsr.core_algos import compute_grpo_outcome_advantage


def test_grpo_neg_overlong_adv_scale_default_noop():
    token_level_rewards = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [3.0, 3.0, 3.0],
            [2.0, 2.0, 0.0],
        ],
        dtype=torch.float32,
    )
    response_mask = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    index = np.array([0, 0, 0], dtype=np.int64)
    score = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    advantages, returns = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        score=score,
        config=AlgoConfig(adv_estimator="grpo", neg_overlong_adv_scale=1.0),
    )

    expected = torch.tensor(
        [
            [-0.7259, -0.7259, -0.7259],
            [1.1406, 1.1406, 1.1406],
            [-0.4148, -0.4148, 0.0000],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(advantages, expected, atol=1e-4)
    assert torch.allclose(returns, expected, atol=1e-4)


def test_grpo_neg_overlong_adv_scale_only_applies_to_negative_clipped_samples():
    token_level_rewards = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [3.0, 3.0, 3.0],
            [2.0, 2.0, 0.0],
        ],
        dtype=torch.float32,
    )
    response_mask = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    index = np.array([0, 0, 0], dtype=np.int64)
    score = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    advantages_x2, _ = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        score=score,
        config=AlgoConfig(adv_estimator="grpo", neg_overlong_adv_scale=2.0),
    )
    advantages_x05, _ = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        score=score,
        config=AlgoConfig(adv_estimator="grpo", neg_overlong_adv_scale=0.5),
    )

    # Sample 0: score=1.0 (positive) -> never scaled
    assert torch.allclose(advantages_x2[0], torch.tensor([-0.7259, -0.7259, -0.7259]), atol=1e-4)
    # Sample 1: score=0.0, mask=[1,1,1] (neg + overlong) -> scaled
    assert torch.allclose(advantages_x2[1], torch.tensor([2.2813, 2.2813, 2.2813]), atol=1e-4)
    # Sample 2: score=0.0, mask=[1,1,0] (neg but not overlong) -> not scaled
    assert torch.allclose(advantages_x2[2], torch.tensor([-0.4148, -0.4148, 0.0]), atol=1e-4)

    assert torch.allclose(advantages_x05[0], torch.tensor([-0.7259, -0.7259, -0.7259]), atol=1e-4)
    assert torch.allclose(advantages_x05[1], torch.tensor([0.5703, 0.5703, 0.5703]), atol=1e-4)
    assert torch.allclose(advantages_x05[2], torch.tensor([-0.4148, -0.4148, 0.0]), atol=1e-4)


def test_grpo_neg_overlong_adv_scale_8192():
    """Test with realistic 8192 response length to verify neg_overlong detection at scale."""
    seq_len = 8192
    n_samples = 4

    # Build token_level_rewards: score placed on last valid token
    token_level_rewards = torch.zeros(n_samples, seq_len, dtype=torch.float32)
    response_mask = torch.zeros(n_samples, seq_len, dtype=torch.float32)

    # Sample 0: positive, full length (overlong but positive -> no scaling)
    response_mask[0, :] = 1.0
    token_level_rewards[0, -1] = 1.0

    # Sample 1: negative, full length (neg + overlong -> should be scaled)
    response_mask[1, :] = 1.0
    token_level_rewards[1, -1] = 0.0

    # Sample 2: negative, shorter (8000 tokens, NOT overlong -> no scaling)
    response_mask[2, :8000] = 1.0
    token_level_rewards[2, 7999] = 0.0

    # Sample 3: positive, shorter (7000 tokens)
    response_mask[3, :7000] = 1.0
    token_level_rewards[3, 6999] = 1.0

    index = np.array([0, 0, 0, 0], dtype=np.int64)
    score = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)

    # Baseline: scale=1.0
    adv_base, _ = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        score=score,
        config=AlgoConfig(adv_estimator="grpo", neg_overlong_adv_scale=1.0),
    )

    # Scaled: scale=2.0
    adv_x2, _ = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        score=score,
        config=AlgoConfig(adv_estimator="grpo", neg_overlong_adv_scale=2.0),
    )

    # Sample 0 (positive, overlong): unchanged
    assert torch.allclose(adv_x2[0], adv_base[0], atol=1e-6)

    # Sample 1 (neg, overlong): advantage scaled by 2.0
    assert torch.allclose(adv_x2[1], adv_base[1] * 2.0, atol=1e-6)

    # Sample 2 (neg, NOT overlong): unchanged
    assert torch.allclose(adv_x2[2], adv_base[2], atol=1e-6)

    # Sample 3 (positive, not overlong): unchanged
    assert torch.allclose(adv_x2[3], adv_base[3], atol=1e-6)

    # Also verify that only sample 1 differs
    diff = (adv_x2 - adv_base).abs().sum(dim=-1)
    assert diff[0] == 0.0
    assert diff[1] > 0.0  # this one was scaled
    assert diff[2] == 0.0
    assert diff[3] == 0.0
