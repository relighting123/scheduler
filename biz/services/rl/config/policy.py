"""YAML 파일에서 관측 정규화 스케일과 보상 가중치를 로드한다.

파일이 없을 경우 코드에 내장된 기본값을 사용하므로,
YAML을 편집하는 것만으로 시뮬레이터 코드를 건드리지 않고 튜닝할 수 있다.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import yaml

from biz.services.rl.env.observation import ObservationNormConfig
from biz.services.rl.env.reward import RewardConfig

_DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "policy_config.yaml")


@dataclass(frozen=True)
class EnvPolicy:
    observation: ObservationNormConfig
    reward: RewardConfig


def load_env_policy(config_path: Optional[str] = None) -> EnvPolicy:
    path = config_path or _DEFAULT_CONFIG_PATH
    if not os.path.isfile(path):
        return EnvPolicy(observation=ObservationNormConfig(), reward=RewardConfig())

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    obs_raw = raw.get("observation") or {}
    reward_raw = raw.get("reward") or {}
    terminal = reward_raw.get("terminal_bonus") or {}

    observation = ObservationNormConfig(
        wip_scale=float(obs_raw.get("wip_scale", 1000.0)),
        eqp_scale=float(obs_raw.get("eqp_scale", 100.0)),
        st_scale=float(obs_raw.get("st_scale", 60.0)),
        plan_scale=float(obs_raw.get("plan_scale", 1000.0)),
        bias_feature=float(obs_raw.get("bias_feature", 1.0)),
    )
    reward = RewardConfig(
        transfer_penalty_per_unit=float(reward_raw.get("transfer_penalty_per_unit", 0.002)),
        guidance_improvement_weight=float(reward_raw.get("guidance_improvement_weight", 0.5)),
        terminal_bonus_99=float(terminal.get("achievement_0_99", 1.5)),
        terminal_bonus_90=float(terminal.get("achievement_0_90", 0.8)),
        terminal_bonus_80=float(terminal.get("achievement_0_80", 0.4)),
        terminal_bonus_60=float(terminal.get("achievement_0_60", 0.1)),
    )
    return EnvPolicy(observation=observation, reward=reward)


def load_observation_and_reward_config(
    config_path: Optional[str] = None,
) -> Tuple[ObservationNormConfig, RewardConfig]:
    policy = load_env_policy(config_path)
    return policy.observation, policy.reward
