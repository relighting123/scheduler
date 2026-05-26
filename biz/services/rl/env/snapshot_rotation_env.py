"""여러 RULE_TIMEKEY 스냅샷을 에피소드마다 무작위 선택하여 학습하는 Gym 래퍼."""
import random

import gymnasium as gym

from biz.services.rl.env.scheduler_env import SchedulerEnv


class SnapshotRotationEnv(gym.Env):
    """from~to 구간 스냅샷 학습 — 10×10 패딩으로 obs/action 차원 고정."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        snapshots: list,
        max_prods=None,
        max_procs=None,
        max_models=None,
        fixed_products=None,
        fixed_processes=None,
        fixed_models=None,
        guidance_targets=None,
    ):
        if not snapshots:
            raise ValueError("학습용 스냅샷이 비어 있습니다.")
        self.snapshots = snapshots
        self.guidance_targets = guidance_targets or [None] * len(snapshots)
        if len(self.guidance_targets) != len(self.snapshots):
            raise ValueError("guidance_targets length must match snapshots length.")
        self.max_prods = max_prods if max_prods is not None else SchedulerEnv.DEFAULT_MAX_PRODS
        self.max_procs = max_procs if max_procs is not None else SchedulerEnv.DEFAULT_MAX_PROCS
        self.max_models = max_models if max_models is not None else SchedulerEnv.DEFAULT_MAX_MODELS
        self.fixed_products = fixed_products
        self.fixed_processes = fixed_processes
        self.fixed_models = fixed_models
        self._inner = self._make_inner_env(snapshots[0], self.guidance_targets[0])
        self.observation_space = self._inner.observation_space
        self.action_space = self._inner.action_space

    def _make_inner_env(self, data, guidance_target=None):
        return SchedulerEnv(
            data=data,
            max_prods=self.max_prods,
            max_procs=self.max_procs,
            max_models=self.max_models,
            fixed_products=self.fixed_products,
            fixed_processes=self.fixed_processes,
            fixed_models=self.fixed_models,
            guidance_target_allocation=guidance_target,
        )

    def reset(self, seed=None, options=None):
        idx = random.randrange(len(self.snapshots))
        self._inner = self._make_inner_env(self.snapshots[idx], self.guidance_targets[idx])
        return self._inner.reset(seed=seed, options=options)

    def step(self, action):
        return self._inner.step(action)

    def render(self):
        if hasattr(self._inner, "render"):
            return self._inner.render()
        return None
