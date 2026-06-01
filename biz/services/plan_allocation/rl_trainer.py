"""정적 배치 PPO 학습·추론."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from biz.services.plan_allocation.models import build_problem
from biz.services.plan_allocation.moves import apply_move
from biz.services.plan_allocation.optimizer import _objective
from biz.services.plan_allocation.rl_env import StaticAllocationEnv
from biz.services.plan_allocation.static_capacity import evaluate_allocation
from biz.services.plan_allocation.static_result import (
    build_static_allocation_rows,
    format_static_allocation_table,
    save_static_allocation_files,
    static_allocation_payload,
)
from biz.services.plan_allocation.optimizer import OptimizationResult, PlanAllocationOptimizer

DEFAULT_MODEL_PATH = "static_allocation_ppo.zip"


def _best_expert_action(env: StaticAllocationEnv) -> int:
    """한 스텝에서 목적함수를 가장 올리는 액션 (모방학습용)."""
    assert env.problem is not None
    best_idx = 0
    best_obj = _objective(evaluate_allocation(env.problem), env.last_oper_weight)
    for idx, move in enumerate(env._moves):
        if move.kind == "noop":
            continue
        trial = env.clone_problem()
        if not apply_move(trial, move):
            continue
        obj = _objective(evaluate_allocation(trial), env.last_oper_weight)
        if obj > best_obj + 1e-6:
            best_obj = obj
            best_idx = idx
    return best_idx


def collect_expert_trajectories(
    data: Dict,
    num_episodes: int = 200,
    max_steps: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    obs_list, act_list = [], []
    for _ in range(num_episodes):
        env = StaticAllocationEnv(data, max_steps=max_steps)
        obs, _ = env.reset()
        done = False
        while not done:
            action = _best_expert_action(env)
            obs_list.append(obs.copy())
            act_list.append(action)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    return np.array(obs_list, dtype=np.float32), np.array(act_list, dtype=np.int64)


def apply_bc_pretrain(model: PPO, obs: np.ndarray, actions: np.ndarray, epochs: int = 15):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = model.device
    policy = model.policy
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()
    ds = DataLoader(
        TensorDataset(
            torch.tensor(obs, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
        ),
        batch_size=64,
        shuffle=True,
    )
    policy.train()
    for _ in range(epochs):
        for batch_obs, batch_act in ds:
            batch_obs = batch_obs.to(device)
            batch_act = batch_act.to(device)
            dist = policy.get_distribution(batch_obs)
            logits = dist.distribution.logits
            loss = loss_fn(logits, batch_act)
            opt.zero_grad()
            loss.backward()
            opt.step()
    policy.eval()
    print(f"[BC] 정적 배치 모방학습 완료 (samples={len(obs)})")


def train_static_allocation_rl(
    data: Dict,
    *,
    total_timesteps: int = 50_000,
    model_path: str = DEFAULT_MODEL_PATH,
    pretrain_bc: bool = True,
    max_steps: int = 64,
) -> PPO:
    env = Monitor(StaticAllocationEnv(data, max_steps=max_steps))
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=512,
        batch_size=64,
        ent_coef=0.02,
        learning_rate=3e-4,
    )

    if pretrain_bc:
        obs, acts = collect_expert_trajectories(data, num_episodes=150, max_steps=max_steps)
        if len(obs) > 0:
            apply_bc_pretrain(model, obs, acts)

    model.learn(total_timesteps=total_timesteps)
    model.save(model_path)
    print(f"[RL] 모델 저장: {model_path}")
    return model


def run_static_allocation_rl(
    data: Dict,
    *,
    model_path: str = DEFAULT_MODEL_PATH,
    rule_timekey: Optional[str] = None,
    max_steps: int = 64,
    output_dir: str = "output",
) -> Dict:
    env = StaticAllocationEnv(data, max_steps=max_steps)
    obs, _ = env.reset()

    model = None
    if os.path.isfile(model_path + ".zip") or os.path.isfile(model_path):
        try:
            model = PPO.load(model_path)
        except Exception as exc:
            print(f"[경고] RL 모델 로드 실패: {exc}")

    done = False
    steps = 0
    while not done:
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
        else:
            action = _best_expert_action(env)
        obs, _, terminated, truncated, info = env.step(action)
        steps += 1
        done = terminated or truncated
        if info.get("move", {}).get("type") == "noop":
            break

    assert env.problem is not None
    opt_result = OptimizationResult(
        problem=env.problem,
        initial_summary=evaluate_allocation(build_problem(data)),
        optimized_summary=evaluate_allocation(env.problem),
        recommended_allocation=env.recommended_allocation(),
        moves=[],
        iterations=steps,
    )
    payload = static_allocation_payload(opt_result, rule_timekey=rule_timekey)
    payload["method"] = "rl_static_allocation" if model else "expert_greedy"
    payload["rl_steps"] = steps
    print(format_static_allocation_table(payload["allocation_table"]))
    paths = save_static_allocation_files(payload, output_dir=output_dir, basename="static_allocation_rl")
    payload["saved_files"] = paths
    print(f"\n저장: {paths['json']}")
    return payload
