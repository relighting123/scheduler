"""여러 RULE_TIMEKEY 스냅샷을 에피소드마다 무작위 선택하여 학습하는 Gym 래퍼."""
import random

import gymnasium as gym

from biz.services.rl.env.env_schema import compute_canonical_schema
from biz.services.rl.env.scheduler_env import SchedulerEnv


class SnapshotRotationEnv(gym.Env):
    """from~to 구간 스냅샷 학습 — 전 스냅샷 합집합으로 obs/action 차원 고정."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, snapshots: list, max_prods=None, max_procs=None):
        if not snapshots:
            raise ValueError("학습용 스냅샷이 비어 있습니다.")
        self.snapshots = snapshots
        self.canonical_products, self.canonical_processes, self.canonical_models = (
            compute_canonical_schema(snapshots, max_prods=max_prods, max_procs=max_procs)
        )
        self._inner = self._make_inner_env(snapshots[0])
        self.observation_space = self._inner.observation_space
        self.action_space = self._inner.action_space

    def _make_inner_env(self, data):
        return SchedulerEnv(
            data=data,
            fixed_products=self.canonical_products,
            fixed_processes=self.canonical_processes,
            fixed_models=self.canonical_models,
        )

    def reset(self, seed=None, options=None):
        data = random.choice(self.snapshots)
        self._inner = self._make_inner_env(data)
        self.observation_space = self._inner.observation_space
        self.action_space = self._inner.action_space
        return self._inner.reset(seed=seed, options=options)

    def step(self, action):
        return self._inner.step(action)

    def render(self):
        if hasattr(self._inner, "render"):
            return self._inner.render()
        return None
