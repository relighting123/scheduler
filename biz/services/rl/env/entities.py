"""시뮬레이터 도메인 엔티티: 설비 단위, 할당 구간, 툴교체 작업."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ConvJob:
    """배치(Tool) 변경 시 1 step 비가용 후 목적지 슬롯에 배치되는 교체 작업."""
    unit: "EquipmentUnit"
    release_step: int
    t_prod: int
    t_proc: int
    to_batch: Optional[str]


@dataclass
class AssignmentSegment:
    """장비(EQP_ID)별 제품·공정(또는 IDLE/CONV) 할당 구간 — SEQ_NO는 장비 내 1부터."""
    seq_no: int
    plan_prod_attr_val: str
    batch_id: Optional[str] = None
    start_step: int = 0
    end_step: Optional[int] = None
    produced_qty: float = 0.0


@dataclass
class EquipmentUnit:
    """모델 풀 기반 개별 설비. EQP_ID는 생성 후 변경하지 않는다."""
    eqp_id: str
    eqp_model_cd: str
    batch_id: Optional[str] = None
    plan_prod_key: Optional[str] = None
    oper_id: Optional[str] = None
    produced_qty: float = 0.0
    conv_remaining_steps: int = 0
    assignment_history: List[AssignmentSegment] = field(default_factory=list)
    _logged_attr: Optional[str] = field(default=None, repr=False)

    def is_idle(self) -> bool:
        return self.plan_prod_key is None and self.conv_remaining_steps <= 0

    def is_in_conv(self) -> bool:
        return self.conv_remaining_steps > 0

    @property
    def plan_prod_attr_val(self) -> Optional[str]:
        if self.conv_remaining_steps > 0:
            return "CONV"
        if self.plan_prod_key and self.oper_id:
            return f"{self.plan_prod_key}|{self.oper_id}"
        return "IDLE"
