"""Reward definition for SchedulerEnv steps."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class RewardConfig:
    """Weights and terminal bonuses; tune via env_policy_config.yaml."""

    transfer_penalty_per_unit: float = 0.002
    guidance_improvement_weight: float = 0.5
    terminal_bonus_99: float = 1.5
    terminal_bonus_90: float = 0.8
    terminal_bonus_80: float = 0.4
    terminal_bonus_60: float = 0.1


@dataclass(frozen=True)
class RewardStepContext:
    """Per-step inputs required to compute reward."""

    step_production: float
    plan: np.ndarray
    produced: np.ndarray
    num_transfers: int
    prev_guidance_gap: float
    next_guidance_gap: float
    guidance_target_eqp: np.ndarray
    terminated: bool
    period_start_produced: Optional[np.ndarray] = None


def compute_reward(
    ctx: RewardStepContext,
    config: RewardConfig | None = None,
) -> float:
    """Incremental step reward + optional terminal bonuses."""
    config = config or RewardConfig()
    total_plan = float(np.sum(ctx.plan)) + 1e-6
    reward = ctx.step_production / total_plan

    if ctx.num_transfers > 0:
        reward -= config.transfer_penalty_per_unit * ctx.num_transfers

    if np.any(ctx.guidance_target_eqp):
        target_total = float(np.sum(ctx.guidance_target_eqp)) + 1e-6
        reward += config.guidance_improvement_weight * (
            (ctx.prev_guidance_gap - ctx.next_guidance_gap) / target_total
        )

    if ctx.terminated:
        period_start = (
            ctx.period_start_produced
            if ctx.period_start_produced is not None
            else np.zeros_like(ctx.produced)
        )
        within_period = ctx.produced - period_start
        final_achievement = float(np.sum(within_period)) / total_plan
        if final_achievement >= 0.99:
            reward += config.terminal_bonus_99
        elif final_achievement >= 0.90:
            reward += config.terminal_bonus_90
        elif final_achievement >= 0.80:
            reward += config.terminal_bonus_80
        elif final_achievement >= 0.60:
            reward += config.terminal_bonus_60

        if np.any(ctx.guidance_target_eqp):
            target_total = float(np.sum(ctx.guidance_target_eqp)) + 1e-6
            target_match = max(0.0, 1.0 - (ctx.next_guidance_gap / target_total))
            reward += target_match

    return float(reward)
