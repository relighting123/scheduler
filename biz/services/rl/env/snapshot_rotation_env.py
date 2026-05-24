"""여러 RULE_TIMEKEY 스냅샷을 에피소드마다 무작위 선택하여 학습하는 Gym 래퍼."""
import random

import gymnasium as gym

from biz.services.rl.env.scheduler_env import SchedulerEnv


class SnapshotRotationEnv(gym.Env):
    """from~to 구간 스냅샷 학습 — 10×10 패딩으로 obs/action 차원 고정."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, snapshots: list, max_prods=None, max_procs=None):
        if not snapshots:
            raise ValueError("학습용 스냅샷이 비어 있습니다.")
        self.snapshots = snapshots
        self.max_prods = max_prods if max_prods is not None else SchedulerEnv.DEFAULT_MAX_PRODS
        self.max_procs = max_procs if max_procs is not None else SchedulerEnv.DEFAULT_MAX_PROCS
        self._inner = self._make_inner_env(snapshots[0])
        self.observation_space = self._inner.observation_space
        self.action_space = self._inner.action_space

    def _make_inner_env(self, data):
        return SchedulerEnv(data=data, max_prods=self.max_prods, max_procs=self.max_procs)

    def reset(self, seed=None, options=None):
        data = random.choice(self.snapshots)
        self._inner = self._make_inner_env(data)
        return self._inner.reset(seed=seed, options=options)

    def step(self, action):
        return self._inner.step(action)

    def render(self):
        if hasattr(self._inner, "render"):
            return self._inner.render()
        return None
