"""정적 배치 강화학습 환경 — 시간 slot 없음, 대수 이동·추가만."""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from biz.services.plan_allocation.models import PlanAllocationProblem, build_problem
from biz.services.plan_allocation.moves import AllocationMove, apply_move, enumerate_moves
from biz.services.plan_allocation.optimizer import _objective
from biz.services.plan_allocation.static_capacity import evaluate_allocation


class StaticAllocationEnv(gym.Env):
    """Input 스냅샷 → 장비 대수 재배치 → 정적 계획 달성률 보상."""

    metadata = {"render_modes": []}
    MAX_ACTIONS = 256
    MAX_SLOTS = 16
    MAX_MODELS = 8

    def __init__(
        self,
        data: Dict,
        max_steps: int = 64,
        last_oper_weight: float = 1.5,
        flow_balance_weight: float = 2.0,
    ):
        super().__init__()
        self.data = data
        self.max_steps = max_steps
        self.last_oper_weight = last_oper_weight
        self.flow_balance_weight = flow_balance_weight

        self.action_space = spaces.Discrete(self.MAX_ACTIONS)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=2.0,
            shape=(self._obs_dim(),),
            dtype=np.float32,
        )

        self.problem: Optional[PlanAllocationProblem] = None
        self._moves: List[AllocationMove] = []
        self._model_index: Dict[str, int] = {}
        self._step = 0
        self._prev_obj = 0.0

    @classmethod
    def _obs_dim(cls) -> int:
        return 1 + cls.MAX_SLOTS * (3 + cls.MAX_MODELS) + cls.MAX_MODELS

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.problem = build_problem(self.data)
        self._model_index = {
            m: i for i, m in enumerate(sorted(self.problem.model_pool.keys()))
        }
        self._step = 0
        self._refresh_moves()
        self._prev_obj = self._score(evaluate_allocation(self.problem))
        return self._get_obs(), {}

    def step(self, action: int):
        assert self.problem is not None
        self._step += 1
        invalid = action < 0 or action >= len(self._moves)
        move = self._moves[0] if invalid else self._moves[int(action)]

        if invalid or (move.kind != "noop" and not apply_move(self.problem, move)):
            reward = -0.15
        elif move.kind == "noop":
            reward = 0.0
        else:
            reward = 0.0

        summary = evaluate_allocation(self.problem)
        obj = self._score(summary)
        reward += (obj - self._prev_obj) * 0.05
        self._prev_obj = obj

        if summary.overall_achievement >= 99.0:
            reward += 0.5

        self._refresh_moves()
        terminated = move.kind == "noop" or self._step >= self.max_steps
        truncated = False
        if self._step >= self.max_steps:
            terminated = True

        info = {
            "overall_achievement": summary.overall_achievement,
            "objective": obj,
            "move": move.describe(self.problem),
            "num_valid_moves": len(self._moves),
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    def _refresh_moves(self) -> None:
        assert self.problem is not None
        moves = enumerate_moves(self.problem)
        if len(moves) > self.MAX_ACTIONS:
            moves = moves[: self.MAX_ACTIONS]
        self._moves = moves

    def _get_obs(self) -> np.ndarray:
        assert self.problem is not None
        summary = evaluate_allocation(self.problem)
        vec = np.zeros(self._obs_dim(), dtype=np.float32)
        vec[0] = summary.overall_achievement / 100.0

        slot_ach = {
            (c.plan_prod_key, c.oper_id): c.achievement_rate / 100.0
            for c in summary.slot_contributions
        }

        base = 1
        for si, slot in enumerate(self.problem.slots[: self.MAX_SLOTS]):
            off = base + si * (3 + self.MAX_MODELS)
            key = slot.key.as_tuple()
            vec[off] = min(slot.plan_qty / 10000.0, 2.0)
            vec[off + 1] = slot_ach.get(key, 0.0)
            vec[off + 2] = min(slot.capacity_per_hour() / 500.0, 2.0)
            for model, qty in slot.allocation.items():
                mi = self._model_index.get(model)
                if mi is not None and mi < self.MAX_MODELS:
                    pool = max(self.problem.model_pool.get(model, 1), 1)
                    vec[off + 3 + mi] = qty / pool

        pool_off = base + self.MAX_SLOTS * (3 + self.MAX_MODELS)
        for model, pool in self.problem.model_pool.items():
            mi = self._model_index.get(model)
            if mi is not None and mi < self.MAX_MODELS:
                used = sum(
                    s.allocation.get(model, 0) for s in self.problem.slots
                )
                vec[pool_off + mi] = used / max(pool, 1)

        return vec

    def _score(self, summary):
        assert self.problem is not None
        return _objective(
            summary,
            self.problem,
            self.last_oper_weight,
            self.flow_balance_weight,
        )

    def recommended_allocation(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        assert self.problem is not None
        out: Dict[str, Dict[str, Dict[str, int]]] = {}
        for slot in self.problem.slots:
            prod, oper = slot.key.plan_prod_key, slot.key.oper_id
            alloc = {m: int(q) for m, q in slot.allocation.items() if q > 0}
            if alloc:
                out.setdefault(prod, {})[oper] = alloc
        return out

    def clone_problem(self) -> PlanAllocationProblem:
        return copy.deepcopy(self.problem)
