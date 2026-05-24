"""여러 RULE_TIMEKEY 스냅샷을 에피소드마다 무작위 선택하여 학습하는 Gym 래퍼."""
import random

import gymnasium as gym

from biz.services.rl.env.scheduler_env import SchedulerEnv


class SnapshotRotationEnv(gym.Env):
    """from~to 구간의 스냅샷 목록으로 학습 데이터 다양화."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, snapshots: list):
        if not snapshots:
            raise ValueError("학습용 스냅샷이 비어 있습니다.")
        self.snapshots = snapshots
        self._inner = SchedulerEnv(data=snapshots[0])
        self.observation_space = self._inner.observation_space
        self.action_space = self._inner.action_space

    def reset(self, seed=None, options=None):
        data = random.choice(self.snapshots)
        self._inner = SchedulerEnv(data=data)
        self.observation_space = self._inner.observation_space
        self.action_space = self._inner.action_space
        return self._inner.reset(seed=seed, options=options)

    def step(self, action):
        return self._inner.step(action)

    def render(self):
        if hasattr(self._inner, "render"):
            return self._inner.render()
        return None
