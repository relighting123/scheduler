"""시간대별 생산 시뮬레이션 로직 — SchedulerEnv.step()에서 분리."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np

if TYPE_CHECKING:
    from biz.services.rl.env.entities import EquipmentUnit
    from biz.services.rl.env.scheduler_env import SchedulerEnv


# ---------------------------------------------------------------------------
# 결과 데이터 클래스
# ---------------------------------------------------------------------------

@dataclass
class SlotProductionLog:
    """단일 (제품, 공정) 슬롯의 시간대별 생산 기록.

    production_logs 리스트에 dict로 변환되어 저장되며,
    InferenceRunner·Excel 리포트 등에서 사용된다.
    """

    time_slot: int
    batch_id: str
    plan_prod_key: str
    oper_id: str
    active_eqp_cnt: float
    transition_eqp_cnt: float
    production_qty: float
    remain_wip: float
    utilization_rate: float
    cum_produced: float
    plan_qty: float
    achievement_rate: float

    def to_dict(self) -> Dict:
        return {
            "TIME_SLOT": self.time_slot,
            "BATCH_ID": self.batch_id,
            "PLAN_PROD_KEY": self.plan_prod_key,
            "OPER_ID": self.oper_id,
            "ACTIVE_EQP_CNT": self.active_eqp_cnt,
            "TRANSITION_EQP_CNT": self.transition_eqp_cnt,
            "PRODUCTION_QTY": self.production_qty,
            "REMAIN_WIP": self.remain_wip,
            "UTILIZATION_RATE(%)": round(self.utilization_rate, 2),
            "CUM_PRODUCED": self.cum_produced,
            "PLAN_QTY": self.plan_qty,
            "ACHIEVEMENT_RATE(%)": round(self.achievement_rate, 2),
        }


@dataclass
class ProductionStepResult:
    """step() 한 번의 전체 생산 시뮬레이션 결과."""

    total_production: float
    logs: List[SlotProductionLog] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 슬롯 단위 생산 능력 계산
# ---------------------------------------------------------------------------

def compute_slot_capacity(
    env: "SchedulerEnv",
    p: int,
    s: int,
) -> List[Tuple["EquipmentUnit", float]]:
    """슬롯 (p, s)에서 생산 가능한 (설비, 단위_생산량) 목록을 반환한다.

    CONV 중이거나 처리불가(avail=False) 모델은 제외한다.
    """
    contributions = []
    for m in range(env.num_models):
        if not env._is_available(p, s, m):
            continue
        st_val = env.st_matrix[p, s, m]
        if st_val <= 0.0:
            continue
        per_unit_cap = 60.0 / st_val
        for unit in env.slot_equipment.get(env._slot_key(p, s, m), []):
            if not unit.is_in_conv():
                contributions.append((unit, per_unit_cap))
    return contributions


# ---------------------------------------------------------------------------
# 순수 지표 계산 함수
# ---------------------------------------------------------------------------

def compute_utilization_rate(actual_produce: float, capacity: float) -> float:
    """실제 생산량 / 최대 생산 가능량 → 가동률 (%)."""
    return (actual_produce / capacity * 100.0) if capacity > 0 else 0.0


def compute_achievement_rate(cum_produced: float, plan_qty: float) -> float:
    """누적 생산량 / 계획량 → 계획달성률 (%)."""
    return (cum_produced / plan_qty * 100.0) if plan_qty > 0 else 0.0


# ---------------------------------------------------------------------------
# step 단위 생산 시뮬레이션
# ---------------------------------------------------------------------------

def compute_production_step(env: "SchedulerEnv") -> ProductionStepResult:
    """현재 step의 모든 슬롯 생산량을 계산하고 WIP·누적 실적을 갱신한다.

    처리 순서 (슬롯별):
    1. 비-CONV 설비 용량 합산 → 실제 생산량 = min(용량, WIP)
    2. 가동률 = 실제 생산량 / 용량
    3. 설비별 생산 기여분 배분 (비율 기준)
    4. WIP 차감 및 다음 공정 투입
    5. 계획달성률 계산 후 SlotProductionLog 생성
    """
    total_production = 0.0
    logs: List[SlotProductionLog] = []

    for p in range(env.num_prods):
        for s in range(env.num_procs):
            contributions = compute_slot_capacity(env, p, s)
            capacity = sum(cap for _, cap in contributions)
            actual_produce = min(capacity, env.wip[p, s])

            active_count = float(np.sum(env.active_eqp[p, s, :]))
            transition_count = float(np.sum(env.target_eqp[p, s, :]))
            utilization = compute_utilization_rate(actual_produce, capacity)

            # 가동 시간 누적
            env.total_eqp_hours[p, s] += active_count
            if capacity > 0:
                env.operating_eqp_hours[p, s] += (actual_produce / capacity) * active_count

            # 설비별 생산 기여 배분
            if contributions and actual_produce > 0:
                total_unit_cap = sum(cap for _, cap in contributions)
                for unit, unit_cap in contributions:
                    share = actual_produce * (unit_cap / total_unit_cap) if total_unit_cap > 0 else 0.0
                    env._add_unit_segment_production(unit, share)

            # WIP 소모 및 다음 공정 투입
            env.produced[p, s] += actual_produce
            env.wip[p, s] -= actual_produce
            if s < env.num_procs - 1:
                env.wip[p, s + 1] += actual_produce

            total_production += actual_produce

            plan_qty = env.plan[p, s]
            cum_produced = env.produced[p, s]
            achievement = compute_achievement_rate(cum_produced, plan_qty)
            batch_id = env.batch_id_map.get((env.products[p], env.processes[s]), "N/A")

            if plan_qty > 0 or actual_produce > 0 or active_count > 0:
                logs.append(
                    SlotProductionLog(
                        time_slot=env.current_step,
                        batch_id=batch_id,
                        plan_prod_key=env.products[p],
                        oper_id=env.processes[s],
                        active_eqp_cnt=active_count,
                        transition_eqp_cnt=transition_count,
                        production_qty=actual_produce,
                        remain_wip=env.wip[p, s],
                        utilization_rate=utilization,
                        cum_produced=cum_produced,
                        plan_qty=plan_qty,
                        achievement_rate=achievement,
                    )
                )

    return ProductionStepResult(total_production=total_production, logs=logs)
