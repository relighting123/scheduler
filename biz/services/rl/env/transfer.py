"""장비 이동(전환) 실행 로직 — SchedulerEnv.step()에서 분리."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from biz.services.rl.env.scheduler_env import SchedulerEnv


@dataclass
class TransferRecord:
    """단일 장비 이동 이력.

    step()의 info["transfers"] 리스트에 담겨 외부로 반환되며,
    InferenceRunner 등 다운스트림 코드가 dict 키로 접근한다.
    """

    from_prod: str
    from_proc: str
    from_batch: Optional[str]
    to_prod: str
    to_proc: str
    to_batch: Optional[str]
    model: str
    eqp_id: str
    needs_conv: bool

    def to_dict(self) -> Dict:
        return {
            "FROM_PROD": self.from_prod,
            "FROM_PROC": self.from_proc,
            "FROM_BATCH": self.from_batch,
            "TO_PROD": self.to_prod,
            "TO_PROC": self.to_proc,
            "TO_BATCH": self.to_batch,
            "MODEL": self.model,
            "EQP_ID": self.eqp_id,
            "NEEDS_CONV": self.needs_conv,
        }


# ---------------------------------------------------------------------------
# 이동 소스 탐색
# ---------------------------------------------------------------------------

def find_best_source_slot(
    env: "SchedulerEnv",
    t_model: int,
    exclude_prod: int,
    exclude_proc: int,
) -> Optional[Tuple[int, int]]:
    """WIP/UPH 우선순위가 가장 낮은 소스 슬롯 (p, s)를 반환한다.

    우선순위 = WIP ÷ UPH.  값이 가장 작은 슬롯이 여유가 많으므로 장비를
    뺏기에 적합하다.  해당 모델 장비가 없거나 이동 불가 시 None을 반환한다.
    """
    best_p, best_s = None, None
    min_priority = float("inf")

    for p in range(env.num_prods):
        for s in range(env.num_procs):
            if env.active_eqp[p, s, t_model] <= 0:
                continue
            if p == exclude_prod and s == exclude_proc:
                continue

            st_val = env.st_matrix[p, s, t_model]
            uph = (60.0 / st_val) if st_val > 0.0 else 0.0
            priority = (env.wip[p, s] / uph) if uph > 0.0 else -1.0

            if priority < min_priority:
                min_priority = priority
                best_p, best_s = p, s

    return (best_p, best_s) if best_p is not None else None


# ---------------------------------------------------------------------------
# 이동 실행
# ---------------------------------------------------------------------------

def _try_move_from_idle(
    env: "SchedulerEnv",
    t_prod: int,
    t_proc: int,
    t_model: int,
    to_batch: Optional[str],
) -> Optional[TransferRecord]:
    """IDLE 풀에서 장비를 꺼내 목적 슬롯으로 이동한다.

    IDLE에서 오는 경우 배치(Tool) 변경이 없으므로 CONV 비가용 없음.
    이동 성공 시 TransferRecord, 실패 시 None 반환.
    """
    unit = env._take_one_from_idle(t_model)
    if unit is None:
        return None

    env.target_eqp[t_prod, t_proc, t_model] += 1

    if env._needs_tool_conv(unit, None, to_batch, from_is_idle=True):
        env._enqueue_tool_conv(unit, t_prod, t_proc, to_batch)
    else:
        env.pending_unit_moves.append((t_prod, t_proc, t_model, unit))

    env._sync_active_eqp_counts()

    return TransferRecord(
        from_prod="IDLE",
        from_proc="IDLE",
        from_batch=None,
        to_prod=env.products[t_prod],
        to_proc=env.processes[t_proc],
        to_batch=to_batch,
        model=env.models[t_model],
        eqp_id=unit.eqp_id,
        needs_conv=False,
    )


def _try_move_from_slot(
    env: "SchedulerEnv",
    t_prod: int,
    t_proc: int,
    t_model: int,
    to_batch: Optional[str],
) -> Optional[TransferRecord]:
    """우선순위 최하 슬롯에서 장비를 꺼내 목적 슬롯으로 이동한다.

    배치(Tool) 변경이 있으면 1 step CONV 비가용이 발생한다.
    이동 성공 시 TransferRecord, 실패 시 None 반환.
    """
    src = find_best_source_slot(env, t_model, exclude_prod=t_prod, exclude_proc=t_proc)
    if src is None:
        return None

    src_p, src_s = src
    src_key = env._slot_key(src_p, src_s, t_model)
    src_units = env.slot_equipment.get(src_key, [])
    from_batch = src_units[-1].batch_id if src_units else None

    unit = env._take_one_from_slot(src_p, src_s, t_model)
    if unit is None:
        return None

    env.target_eqp[t_prod, t_proc, t_model] += 1
    needs_conv = env._needs_tool_conv(unit, from_batch, to_batch, from_is_idle=False)

    if needs_conv:
        env._enqueue_tool_conv(unit, t_prod, t_proc, to_batch)
    else:
        env.pending_unit_moves.append((t_prod, t_proc, t_model, unit))

    env._sync_active_eqp_counts()

    return TransferRecord(
        from_prod=env.products[src_p],
        from_proc=env.processes[src_s],
        from_batch=from_batch,
        to_prod=env.products[t_prod],
        to_proc=env.processes[t_proc],
        to_batch=to_batch,
        model=env.models[t_model],
        eqp_id=unit.eqp_id,
        needs_conv=needs_conv,
    )


def execute_transfer_action(
    env: "SchedulerEnv",
    action: int,
) -> List[TransferRecord]:
    """액션을 장비 이동으로 변환하고 실행한 뒤 TransferRecord 목록을 반환한다.

    처리 순서:
    1. step 0 또는 no-op(action=0) → 이동 없음
    2. 목적 슬롯이 처리 불가이거나 ST 정보 없음 → 이동 없음
    3. IDLE 풀에 해당 모델 장비 있음 → IDLE에서 이동 (CONV 없음)
    4. IDLE 없음 → WIP/UPH 우선순위 최하 슬롯에서 이동 (배치 변경 시 CONV 발생)
    """
    if env.current_step == 0 or action == 0:
        return []

    target_idx = action - 1
    t_prod = target_idx // (env.num_procs * env.num_models)
    rem = target_idx % (env.num_procs * env.num_models)
    t_proc = rem // env.num_models
    t_model = rem % env.num_models

    if not env._is_available(t_prod, t_proc, t_model):
        return []
    if env.st_matrix[t_prod, t_proc, t_model] <= 0:
        return []

    to_batch = env.batch_id_map.get((env.products[t_prod], env.processes[t_proc]))

    record = _try_move_from_idle(env, t_prod, t_proc, t_model, to_batch)
    if record is not None:
        return [record]

    record = _try_move_from_slot(env, t_prod, t_proc, t_model, to_batch)
    if record is not None:
        return [record]

    return []
