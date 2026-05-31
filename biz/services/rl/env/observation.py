"""Observation (state) vector definition for SchedulerEnv."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from biz.services.rl.env.scheduler_env import SchedulerEnv


@dataclass(frozen=True)
class ObservationNormConfig:
    """Normalization scales for each observation block."""

    wip_scale: float = 1000.0
    eqp_scale: float = 100.0
    st_scale: float = 60.0
    plan_scale: float = 1000.0
    bias_feature: float = 1.0


def observation_dim(num_prods: int, num_procs: int, num_models: int) -> int:
    """Flat observation size for product x process x model grids."""
    return num_prods * num_procs * 8 + num_prods * num_procs * num_models * 5 + 2


def build_observation_from_env(
    env: "SchedulerEnv",
    norm: ObservationNormConfig | None = None,
) -> np.ndarray:
    """Build observation vector from a SchedulerEnv instance."""
    norm = norm or ObservationNormConfig()
    num_prods, num_procs, num_models = env.num_prods, env.num_procs, env.num_models

    wip_norm = (env.wip / norm.wip_scale).flatten()
    active_norm = (env.active_eqp.sum(axis=2) / norm.eqp_scale).flatten()
    target_norm = (env.target_eqp.sum(axis=2) / norm.eqp_scale).flatten()
    co_norm = np.zeros((num_prods, num_procs)).flatten()

    period_start = getattr(env, "period_start_produced", np.zeros_like(env.produced))
    within_period_produced = env.produced - period_start
    produced_ratio = (within_period_produced / (env.plan + 1e-6)).flatten()

    st_per_pp = np.zeros((num_prods, num_procs))
    for i in range(num_prods):
        for j in range(num_procs):
            sts = env.st_matrix[i, j, :]
            positive_sts = sts[sts > 0]
            st_per_pp[i, j] = positive_sts.min() if positive_sts.size > 0 else 0.0

    st_norm = (st_per_pp / norm.st_scale).flatten()
    plan_norm = (env.plan / norm.plan_scale).flatten()
    wip_plan_ratio = (env.wip / (env.plan + 1e-6)).flatten()
    model_active_norm = (env.active_eqp / norm.eqp_scale).flatten()
    model_target_norm = (env.target_eqp / norm.eqp_scale).flatten()
    model_st_norm = (env.st_matrix / norm.st_scale).flatten()
    model_avail = env.avail_matrix.astype(np.float32).flatten()
    guidance_target_norm = (env.guidance_target_eqp / norm.eqp_scale).flatten()

    obs = np.concatenate(
        [
            wip_norm,
            active_norm,
            target_norm,
            co_norm,
            produced_ratio,
            st_norm,
            plan_norm,
            wip_plan_ratio,
            model_active_norm,
            model_target_norm,
            model_st_norm,
            model_avail,
            guidance_target_norm,
            [norm.bias_feature],
            [float(env.current_step) / env.max_steps],
        ]
    ).astype(np.float32)

    expected = observation_dim(num_prods, num_procs, num_models)
    if obs.shape != (expected,):
        raise ValueError(f"관측 벡터 크기 불일치: got {obs.shape}, expected ({expected},)")
    return obs
