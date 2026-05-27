import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import os

from biz.services.rl.env.env_schema import discover_entities_from_data


@dataclass
class ConvJob:
    """Tool 교체(배치 변경) 시 1 slot 비가용 후 목적지 배치."""
    unit: "EquipmentUnit"
    release_step: int
    t_prod: int
    t_proc: int
    to_batch: Optional[str]


@dataclass
class AssignmentSegment:
    """장비(EQP_ID)별 제품·공정(또는 IDLE/CONV) 할당 순서 — SEQ_NO는 장비 내 1부터."""
    seq_no: int
    plan_prod_attr_val: str
    batch_id: Optional[str] = None
    start_step: int = 0
    end_step: Optional[int] = None
    produced_qty: float = 0.0
    hourly_produced: Dict[int, float] = field(default_factory=dict)


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


class SchedulerEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    DEFAULT_MAX_PRODS = 10
    DEFAULT_MAX_PROCS = 10
    DEFAULT_MAX_MODELS = 10

    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        max_prods: int = DEFAULT_MAX_PRODS,
        max_procs: int = DEFAULT_MAX_PROCS,
        max_models: int = DEFAULT_MAX_MODELS,
        max_steps: int = 24,
        fixed_products: Optional[List[str]] = None,
        fixed_processes: Optional[List[str]] = None,
        fixed_models: Optional[List[str]] = None,
        guidance_target_allocation: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
    ):
        super(SchedulerEnv, self).__init__()
        self.data = data
        self.max_prods = max_prods
        self.max_procs = max_procs
        self.max_models = max_models
        self.max_steps = max_steps
        self.fixed_products = fixed_products
        self.fixed_processes = fixed_processes
        self.fixed_models = fixed_models
        self.guidance_target_allocation = guidance_target_allocation

        self.products = []
        self.processes = []
        self.models = []
        self.conv_queue: List[ConvJob] = []

        self._discover_entities()
        
        self.num_prods = len(self.products)
        self.num_procs = len(self.processes)
        self.num_models = len(self.models)
        
        # Ensure we have at least 1 dimension
        if self.num_prods == 0: self.num_prods = 1
        if self.num_procs == 0: self.num_procs = 1
        if self.num_models == 0: self.num_models = 1

        self.action_space = spaces.Discrete(self.num_prods * self.num_procs * self.num_models + 1)
        
        self.obs_dim = (
            self.num_prods * self.num_procs * 8
            + self.num_prods * self.num_procs * self.num_models * 5
            + 2
        )
        self.observation_space = spaces.Box(low=0, high=1000, shape=(self.obs_dim,), dtype=np.float32)

        self.reset()

    def _discover_entities(self):
        if self.fixed_products is not None:
            self.products = list(self.fixed_products)
            self.processes = list(self.fixed_processes)
            self.models = list(self.fixed_models)
        else:
            self.products, self.processes, self.models = discover_entities_from_data(
                self.data,
                max_prods=self.max_prods,
                max_procs=self.max_procs,
                max_models=self.max_models,
            )
        while len(self.products) < self.max_prods:
            self.products.append(f"PAD_PROD_{len(self.products)}")
        while len(self.processes) < self.max_procs:
            self.processes.append(f"PAD_PROC_{len(self.processes)}")
        while len(self.models) < self.max_models:
            self.models.append(f"PAD_MODEL_{len(self.models)}")

    def _needs_tool_conv(
        self,
        unit: EquipmentUnit,
        from_batch: Optional[str],
        to_batch: Optional[str],
        from_is_idle: bool,
    ) -> bool:
        """배치(Tool) 변경 시 1 slot 교체 비가용 — 동일 batch는 교체 없음."""
        if from_is_idle:
            return False
        if from_batch is None and to_batch is None:
            return False
        return str(from_batch or "") != str(to_batch or "")

    def _load_avail_matrix(self):
        """AVAIL_INFO: 장비가 오더에 배치되어 있어도 AVAIL_YN='N'이면 해당 공정에서 진행 불가."""
        self.avail_matrix.fill(True)
        avail_df = self.data.get('avail_info', pd.DataFrame())
        if avail_df.empty:
            return
        for _, row in avail_df.iterrows():
            p, s, m = row['PLAN_PROD_KEY'], row['OPER_ID'], row['EQP_MODEL_CD']
            if p in self.prod_idx and s in self.proc_idx and m in self.model_idx:
                yn = str(row['AVAIL_YN']).strip().upper()
                self.avail_matrix[self.prod_idx[p], self.proc_idx[s], self.model_idx[m]] = (yn == 'Y')

    def _apply_avail_to_st_matrix(self):
        """처리 불가 조합은 ST를 0으로 두어 생산·배치 액션에서 제외."""
        for p in range(self.num_prods):
            for s in range(self.num_procs):
                for m in range(self.num_models):
                    if not self.avail_matrix[p, s, m]:
                        self.st_matrix[p, s, m] = 0.0

    def _relocate_eqp_from_unavailable_slots(self):
        """진행 불가 슬롯에 배치된 장비는 해당 모델의 IDLE 풀로 반환."""
        for p in range(self.num_prods):
            for s in range(self.num_procs):
                for m in range(self.num_models):
                    if not self.avail_matrix[p, s, m]:
                        key = self._slot_key(p, s, m)
                        units = self.slot_equipment.pop(key, [])
                        for unit in units:
                            self._return_unit_to_idle(unit)
        self._sync_active_eqp_counts()

    def _is_available(self, p: int, s: int, m: int) -> bool:
        return bool(self.avail_matrix[p, s, m])

    def _slot_key(self, p: int, s: int, m: int) -> Tuple[int, int, int]:
        return (p, s, m)

    def _current_unit_attr(self, unit: EquipmentUnit) -> str:
        return unit.plan_prod_attr_val or "IDLE"

    def _renumber_assignment_history(self, unit: EquipmentUnit):
        for idx, seg in enumerate(unit.assignment_history, start=1):
            seg.seq_no = idx

    def _close_ephemeral_idle_before_conv(self, unit: EquipmentUnit, new_attr: str):
        """슬롯 이탈 직후 같은 step에 생기는 IDLE→CONV 중 IDLE 행 제거."""
        if new_attr != "CONV" or not unit.assignment_history:
            return
        prev = unit.assignment_history[-1]
        if (
            prev.plan_prod_attr_val == "IDLE"
            and prev.start_step == self.current_step
            and (prev.end_step is None or prev.end_step == self.current_step)
        ):
            unit.assignment_history.pop()
            self._renumber_assignment_history(unit)

    def _sync_unit_assignment_log(self, unit: EquipmentUnit):
        """장비별 할당 상태 변경 시 SEQ_NO(1부터) 이력 기록."""
        attr = self._current_unit_attr(unit)
        if unit._logged_attr == attr:
            return
        if unit.assignment_history:
            unit.assignment_history[-1].end_step = self.current_step
        self._close_ephemeral_idle_before_conv(unit, attr)
        unit._logged_attr = attr
        end_step = None
        if attr == "CONV":
            end_step = self.current_step + 1
        unit.assignment_history.append(
            AssignmentSegment(
                seq_no=len(unit.assignment_history) + 1,
                plan_prod_attr_val=attr,
                batch_id=unit.batch_id,
                start_step=self.current_step,
                end_step=end_step,
            )
        )

    def _add_unit_segment_production(self, unit: EquipmentUnit, qty: float):
        if qty <= 0:
            return
        unit.produced_qty += qty
        if unit.assignment_history:
            seg = unit.assignment_history[-1]
            seg.produced_qty += qty
            step_key = int(self.current_step)
            seg.hourly_produced[step_key] = seg.hourly_produced.get(step_key, 0.0) + qty

    def _clear_unit_location(self, unit: EquipmentUnit, log_assignment: bool = False):
        """슬롯에서 빼기만 할 때는 IDLE 로그 생략(CONV·pending 이동 직전)."""
        unit.batch_id = None
        unit.plan_prod_key = None
        unit.oper_id = None
        if log_assignment:
            self._sync_unit_assignment_log(unit)

    def _assign_unit_to_slot(self, unit: EquipmentUnit, p: int, s: int, batch_id: Optional[str]):
        m = self.model_idx[unit.eqp_model_cd]
        key = self._slot_key(p, s, m)
        self.slot_equipment.setdefault(key, []).append(unit)
        unit.batch_id = batch_id
        unit.plan_prod_key = self.products[p]
        unit.oper_id = self.processes[s]
        self._sync_unit_assignment_log(unit)

    def _take_one_from_idle(self, m: int) -> Optional[EquipmentUnit]:
        model = self.models[m]
        if not self.idle_equipment.get(model):
            return None
        return self.idle_equipment[model].pop()

    def _take_one_from_slot(self, p: int, s: int, m: int) -> Optional[EquipmentUnit]:
        key = self._slot_key(p, s, m)
        units = self.slot_equipment.get(key, [])
        if not units:
            return None
        unit = units.pop()
        self._clear_unit_location(unit, log_assignment=False)
        return unit

    def _return_unit_to_idle(self, unit: EquipmentUnit):
        if unit.is_in_conv():
            return
        self._clear_unit_location(unit, log_assignment=True)
        self.idle_equipment.setdefault(unit.eqp_model_cd, []).append(unit)

    def _release_conv_units(self, force: bool = False):
        """교체 비가용 기간이 끝난 장비를 목적 슬롯에 배치."""
        remaining: List[ConvJob] = []
        for job in self.conv_queue:
            if force or self.current_step >= job.release_step:
                job.unit.conv_remaining_steps = 0
                self._assign_unit_to_slot(job.unit, job.t_prod, job.t_proc, job.to_batch)
            else:
                remaining.append(job)
        self.conv_queue = remaining
        self._sync_active_eqp_counts()

    def _flush_conv_queue(self):
        """에피소드 종료 시 남은 CONV 장비를 목적지에 배치."""
        self._release_conv_units(force=True)

    def _enqueue_tool_conv(
        self,
        unit: EquipmentUnit,
        t_prod: int,
        t_proc: int,
        to_batch: Optional[str],
    ):
        """이동 직후 1 slot 비가용(CONV) — 해당 step 1시간만 기록, 다음 step 배치."""
        unit.conv_remaining_steps = 1
        unit._logged_attr = None
        self._sync_unit_assignment_log(unit)
        self.conv_queue.append(
            ConvJob(
                unit=unit,
                release_step=self.current_step + 1,
                t_prod=t_prod,
                t_proc=t_proc,
                to_batch=to_batch,
            )
        )

    def _sync_active_eqp_counts(self):
        self.active_eqp.fill(0)
        for (p, s, m), units in self.slot_equipment.items():
            self.active_eqp[p, s, m] = float(len(units))
        for m_idx, model in enumerate(self.models):
            self.idle_eqp[m_idx] = float(len(self.idle_equipment.get(model, [])))

    def _build_equipment_pool(self):
        """EQP_QTY_INFO 합계 기준 모델별 설비 풀 생성 (EQP_ID = {MODEL}-{00001})."""
        eqp_qty_df = self.data.get('eqp_qty_info', pd.DataFrame())
        model_totals = {model: 0 for model in self.models}

        if not eqp_qty_df.empty and 'EQP_MODEL_CD' in eqp_qty_df.columns:
            grouped = eqp_qty_df.groupby('EQP_MODEL_CD')['EQP_QTY'].sum()
            for model in self.models:
                if model in grouped.index:
                    model_totals[model] = int(grouped[model])

        for m_idx, model in enumerate(self.models):
            assigned = int(round(float(np.sum(self.active_eqp[:, :, m_idx]))))
            model_totals[model] = max(model_totals.get(model, 0), assigned)

        self.equipment_units: List[EquipmentUnit] = []
        self.idle_equipment: Dict[str, List[EquipmentUnit]] = {model: [] for model in self.models}
        self.slot_equipment: Dict[Tuple[int, int, int], List[EquipmentUnit]] = {}
        self.pending_unit_moves: List[Tuple[int, int, int, EquipmentUnit]] = []

        model_counters = {model: 0 for model in self.models}
        for model in self.models:
            total = model_totals.get(model, 0)
            for _ in range(total):
                model_counters[model] += 1
                unit = EquipmentUnit(
                    eqp_id=f"{model}-{model_counters[model]:05d}",
                    eqp_model_cd=model,
                )
                self.equipment_units.append(unit)
                self.idle_equipment[model].append(unit)

    def _deploy_initial_equipment(self):
        """초기 active_eqp 대수에 맞춰 IDLE 풀에서 슬롯으로 배치."""
        for p in range(self.num_prods):
            for s in range(self.num_procs):
                prod = self.products[p]
                oper = self.processes[s]
                batch_id = self.batch_id_map.get((prod, oper))
                for m_idx, model in enumerate(self.models):
                    count = int(round(float(self.active_eqp[p, s, m_idx])))
                    for _ in range(count):
                        unit = self._take_one_from_idle(m_idx)
                        if unit is None:
                            break
                        self._assign_unit_to_slot(unit, p, s, batch_id)
        self._sync_active_eqp_counts()

    def _resolve_pending_unit_moves(self):
        for t_prod, t_proc, t_model, unit in self.pending_unit_moves:
            batch_id = self.batch_id_map.get(
                (self.products[t_prod], self.processes[t_proc])
            )
            self._assign_unit_to_slot(unit, t_prod, t_proc, batch_id)
        self.pending_unit_moves = []
        self._sync_active_eqp_counts()

    def get_deployed_equipment_units(self) -> List[EquipmentUnit]:
        """현재 슬롯에 배치된(비-IDLE) 설비 목록."""
        deployed = []
        for units in self.slot_equipment.values():
            deployed.extend(units)
        return deployed

    def get_all_equipment_units(self) -> List[EquipmentUnit]:
        """설비 풀 전체를 EQP_ID 순으로 반환."""
        if not hasattr(self, "equipment_units"):
            return []
        return sorted(self.equipment_units, key=lambda u: u.eqp_id)

    def finalize_assignment_history(self):
        """에피소드 종료 시 마지막 할당 구간의 end_step 확정."""
        end_step = self.current_step
        for unit in self.get_all_equipment_units():
            if unit.assignment_history and unit.assignment_history[-1].end_step is None:
                unit.assignment_history[-1].end_step = end_step

    def _segment_covering_step(
        self, unit: EquipmentUnit, step: int, max_steps: int
    ) -> Optional[AssignmentSegment]:
        for seg in unit.assignment_history:
            end_step = seg.end_step if seg.end_step is not None else max_steps
            if int(seg.start_step) <= step < int(end_step):
                return seg
        return None

    def iter_rts_assignment_records(self) -> List[Tuple[EquipmentUnit, AssignmentSegment]]:
        """RTS_RSLT_MAS용 (장비, 장비별 SEQ 할당 구간) 목록."""
        self.finalize_assignment_history()
        records: List[Tuple[EquipmentUnit, AssignmentSegment]] = []
        for unit in self.get_all_equipment_units():
            if not unit.assignment_history:
                self._sync_unit_assignment_log(unit)
            for seg in unit.assignment_history:
                records.append((unit, seg))
        return records

    def iter_rts_hourly_records(
        self, max_steps: Optional[int] = None
    ) -> List[Tuple[EquipmentUnit, AssignmentSegment, int]]:
        """RTS_RSLT_MAS용 (장비, 해당 시각 할당 구간, slot step) — 장비×시간(1hr) 단위."""
        slot_count = int(max_steps if max_steps is not None else self.max_steps)
        self.finalize_assignment_history()
        records: List[Tuple[EquipmentUnit, AssignmentSegment, int]] = []
        for unit in self.get_all_equipment_units():
            if not unit.assignment_history:
                self._sync_unit_assignment_log(unit)
                self.finalize_assignment_history()
            for step in range(slot_count):
                seg = self._segment_covering_step(unit, step, slot_count)
                if seg is None:
                    continue
                records.append((unit, seg, step))
        return records


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        self.wip = np.zeros((self.num_prods, self.num_procs))
        self.plan = np.zeros((self.num_prods, self.num_procs))
        self.produced = np.zeros((self.num_prods, self.num_procs))
        self.total_eqp_hours = np.zeros((self.num_prods, self.num_procs))
        self.operating_eqp_hours = np.zeros((self.num_prods, self.num_procs))

        self.active_eqp = np.zeros((self.num_prods, self.num_procs, self.num_models))
        self.target_eqp = np.zeros((self.num_prods, self.num_procs, self.num_models))
        self.idle_eqp = np.zeros(self.num_models)
        self.st_matrix = np.zeros((self.num_prods, self.num_procs, self.num_models))
        self.avail_matrix = np.ones((self.num_prods, self.num_procs, self.num_models), dtype=bool)
        self.guidance_target_eqp = np.zeros((self.num_prods, self.num_procs, self.num_models))
        
        self.prod_idx = {p: i for i, p in enumerate(self.products)}
        self.proc_idx = {p: i for i, p in enumerate(self.processes)}
        self.model_idx = {m: i for i, m in enumerate(self.models)}
        
        # Load from df
        wip_df = self.data.get('wip_info', pd.DataFrame())
        if not wip_df.empty:
            for _, row in wip_df.iterrows():
                p, s = row['PLAN_PROD_KEY'], row['OPER_ID']
                if p in self.prod_idx and s in self.proc_idx:
                    self.wip[self.prod_idx[p], self.proc_idx[s]] += float(row['WIP_QTY'])
                    
        plan_df = self.data.get('plan_info', pd.DataFrame())
        if not plan_df.empty:
            for _, row in plan_df.iterrows():
                p, s = row['PLAN_PROD_KEY'], row['OPER_ID']
                if p in self.prod_idx and s in self.proc_idx:
                    self.plan[self.prod_idx[p], self.proc_idx[s]] += float(row['PLAN_QTY'])
                    
        uph_df = self.data.get('uph_info', pd.DataFrame())
        if not uph_df.empty:
            for _, row in uph_df.iterrows():
                p, s, m = row['PLAN_PROD_KEY'], row['OPER_ID'], row['EQP_MODEL_CD']
                if p in self.prod_idx and s in self.proc_idx and m in self.model_idx:
                    uph = float(row['UPH'])
                    if uph > 0:
                        self.st_matrix[self.prod_idx[p], self.proc_idx[s], self.model_idx[m]] = 60.0 / uph

        self._load_avail_matrix()
        self._apply_avail_to_st_matrix()
        self._load_guidance_target_allocation()
                        
        # Initialize active equipment from DB
        batch_df = self.data.get('batch_tool_info', pd.DataFrame())
        eqp_qty_df = self.data.get('eqp_qty_info', pd.DataFrame())
        
        # Precompute batch_id mapping to speed up step()
        self.batch_id_map = {}
        if not batch_df.empty:
            for _, row in batch_df.iterrows():
                self.batch_id_map[(row['PLAN_PROD_KEY'], row['OPER_ID'])] = row['BATCH_ID']
        
        has_eqp_data = False
        if not batch_df.empty and not eqp_qty_df.empty:
            merged = pd.merge(batch_df, eqp_qty_df, on='BATCH_ID')
            if not merged.empty:
                has_eqp_data = True
                for _, row in merged.iterrows():
                    p, s, m = row['PLAN_PROD_KEY'], row['OPER_ID'], row['EQP_MODEL_CD']
                    if p in self.prod_idx and s in self.proc_idx and m in self.model_idx:
                        self.active_eqp[self.prod_idx[p], self.proc_idx[s], self.model_idx[m]] += float(row['EQP_QTY'])
                        
        if not has_eqp_data:
            # Dummy init for active equipment if there's no data
            for p in range(self.num_prods):
                for s in range(self.num_procs):
                    for m in range(self.num_models):
                        if self.st_matrix[p, s, m] > 0:
                            self.active_eqp[p, s, m] = 1.0

        self.conv_queue = []
        self._build_equipment_pool()
        self._deploy_initial_equipment()
        self._relocate_eqp_from_unavailable_slots()
        self._record_initial_assignments()
        self.production_logs = []
        return self._get_obs(), {}

    def _load_guidance_target_allocation(self):
        if not self.guidance_target_allocation:
            return
        for prod, opers in self.guidance_target_allocation.items():
            if prod not in self.prod_idx:
                continue
            for oper, models in opers.items():
                if oper not in self.proc_idx:
                    continue
                for model, qty in models.items():
                    if model in self.model_idx:
                        self.guidance_target_eqp[
                            self.prod_idx[prod],
                            self.proc_idx[oper],
                            self.model_idx[model],
                        ] = float(qty)

    def _guidance_allocation_gap(self) -> float:
        if not np.any(self.guidance_target_eqp):
            return 0.0
        return float(np.sum(np.abs(self.active_eqp - self.guidance_target_eqp)))

    def _record_initial_assignments(self):
        """초기 배치·유휴 상태를 장비별 SEQ_NO=1부터 이력에 기록."""
        for unit in self.equipment_units:
            unit.assignment_history = []
            unit._logged_attr = None
            self._sync_unit_assignment_log(unit)

    def _get_obs(self):
        # Simplified normalization for observation
        wip_norm = (self.wip / 1000.0).flatten()
        active_norm = (self.active_eqp.sum(axis=2) / 100.0).flatten()
        target_norm = (self.target_eqp.sum(axis=2) / 100.0).flatten()
        co_norm = np.zeros((self.num_prods, self.num_procs)).flatten() # simplified
        produced_ratio = (self.produced / (self.plan + 1e-6)).flatten()
        
        st_per_pp = np.zeros((self.num_prods, self.num_procs))
        for i in range(self.num_prods):
            for j in range(self.num_procs):
                sts = self.st_matrix[i, j, :]
                positive_sts = sts[sts > 0]
                st_per_pp[i, j] = positive_sts.min() if positive_sts.size > 0 else 0.0
                
        st_norm = (st_per_pp / 60.0).flatten()
        plan_norm = (self.plan / 1000.0).flatten()
        wip_plan_ratio = (self.wip / (self.plan + 1e-6)).flatten()
        model_active_norm = (self.active_eqp / 100.0).flatten()
        model_target_norm = (self.target_eqp / 100.0).flatten()
        model_st_norm = (self.st_matrix / 60.0).flatten()
        model_avail = self.avail_matrix.astype(np.float32).flatten()
        guidance_target_norm = (self.guidance_target_eqp / 100.0).flatten()
        
        obs = np.concatenate([
            wip_norm, active_norm, target_norm, co_norm,
            produced_ratio, st_norm, plan_norm, wip_plan_ratio,
            model_active_norm, model_target_norm, model_st_norm,
            model_avail, guidance_target_norm,
            [1.0], [float(self.current_step) / self.max_steps]
        ]).astype(np.float32)
        if obs.shape != (self.obs_dim,):
            raise ValueError(
                f"관측 벡터 크기 불일치: got {obs.shape}, expected ({self.obs_dim},)"
            )
        return obs

    def step(self, action):
        transfers = []
        prev_guidance_gap = self._guidance_allocation_gap()

        self._release_conv_units()

        # 초반(current_step == 0)에는 장비 전환 생성이 있을 수 없으므로 action을 0(전환 없음)으로 처리
        if self.current_step == 0:
            action = 0
            
        if action > 0:
            target_idx = action - 1
            t_prod = target_idx // (self.num_procs * self.num_models)
            rem = target_idx % (self.num_procs * self.num_models)
            t_proc = rem // self.num_models
            t_model = rem % self.num_models
            
            # UPH>0 이고 AVAIL_YN='Y'인 경우만 배치 가능
            if self._is_available(t_prod, t_proc, t_model) and self.st_matrix[t_prod, t_proc, t_model] > 0:
                moved = False
                to_batch = self.batch_id_map.get(
                    (self.products[t_prod], self.processes[t_proc])
                )
                # Try pulling from IDLE first
                unit = self._take_one_from_idle(t_model)
                if unit is not None:
                    self.target_eqp[t_prod, t_proc, t_model] += 1
                    if self._needs_tool_conv(unit, None, to_batch, from_is_idle=True):
                        self._enqueue_tool_conv(unit, t_prod, t_proc, to_batch)
                    else:
                        self.pending_unit_moves.append((t_prod, t_proc, t_model, unit))
                    moved = True
                    self._sync_active_eqp_counts()
                    transfers.append({
                        'FROM_PROD': 'IDLE', 'FROM_PROC': 'IDLE', 'FROM_BATCH': None,
                        'TO_PROD': self.products[t_prod], 'TO_PROC': self.processes[t_proc],
                        'TO_BATCH': to_batch,
                        'MODEL': self.models[t_model], 'EQP_ID': unit.eqp_id,
                        'NEEDS_CONV': False,
                    })

                # If no IDLE, pull from another active node (Priority-based: pull from least needed node first)
                if not moved:
                    best_src_p, best_src_s = None, None
                    min_priority = float('inf')

                    for p in range(self.num_prods):
                        for s in range(self.num_procs):
                            if self.active_eqp[p, s, t_model] > 0 and (p != t_prod or s != t_proc):
                                uph = 0.0
                                st_val = self.st_matrix[p, s, t_model]
                                if st_val > 0.0:
                                    uph = 60.0 / st_val

                                if uph == 0.0:
                                    priority = -1.0
                                else:
                                    priority = self.wip[p, s] / uph

                                if priority < min_priority:
                                    min_priority = priority
                                    best_src_p, best_src_s = p, s

                    if best_src_p is not None:
                        src_key = self._slot_key(best_src_p, best_src_s, t_model)
                        src_units = self.slot_equipment.get(src_key, [])
                        from_batch = src_units[-1].batch_id if src_units else None
                        unit = self._take_one_from_slot(best_src_p, best_src_s, t_model)
                        if unit is not None:
                            self.target_eqp[t_prod, t_proc, t_model] += 1
                            needs_conv = self._needs_tool_conv(
                                unit, from_batch, to_batch, from_is_idle=False
                            )
                            if needs_conv:
                                self._enqueue_tool_conv(unit, t_prod, t_proc, to_batch)
                            else:
                                self.pending_unit_moves.append((t_prod, t_proc, t_model, unit))
                            moved = True
                            self._sync_active_eqp_counts()
                            transfers.append({
                                'FROM_PROD': self.products[best_src_p],
                                'FROM_PROC': self.processes[best_src_s],
                                'FROM_BATCH': from_batch,
                                'TO_PROD': self.products[t_prod],
                                'TO_PROC': self.processes[t_proc],
                                'TO_BATCH': to_batch,
                                'MODEL': self.models[t_model],
                                'EQP_ID': unit.eqp_id,
                                'NEEDS_CONV': needs_conv,
                            })

        # Simulate production
        step_production = 0.0
        for p in range(self.num_prods):
            for s in range(self.num_procs):
                capacity = 0.0
                unit_contributions = []

                for m in range(self.num_models):
                    if not self._is_available(p, s, m):
                        continue
                    st_val = self.st_matrix[p, s, m]
                    if st_val > 0.0:
                        per_unit_cap = 60.0 / st_val
                        slot_units = self.slot_equipment.get(self._slot_key(p, s, m), [])
                        for unit in slot_units:
                            if unit.is_in_conv():
                                continue
                            unit_contributions.append((unit, per_unit_cap))
                            capacity += per_unit_cap

                active_count = np.sum(self.active_eqp[p, s, :])
                transition_count = np.sum(self.target_eqp[p, s, :])
                actual_produce = min(capacity, self.wip[p, s])

                utilization = (actual_produce / capacity * 100.0) if capacity > 0 else 0.0
                self.total_eqp_hours[p, s] += active_count * 1.0
                self.operating_eqp_hours[p, s] += (
                    actual_produce / capacity * active_count * 1.0
                ) if capacity > 0 else 0.0

                if unit_contributions:
                    total_unit_cap = sum(cap for _, cap in unit_contributions)
                    for unit, unit_cap in unit_contributions:
                        share = actual_produce * (unit_cap / total_unit_cap) if total_unit_cap > 0 else 0.0
                        self._add_unit_segment_production(unit, share)

                self.produced[p, s] += actual_produce
                step_production += actual_produce
                self.wip[p, s] -= actual_produce
                if s < self.num_procs - 1:
                    self.wip[p, s+1] += actual_produce

                plan_qty = self.plan[p, s]
                cum_produced = self.produced[p, s]
                achievement_rate = (cum_produced / plan_qty * 100.0) if plan_qty > 0 else 0.0
                batch_id = self.batch_id_map.get((self.products[p], self.processes[s]), 'N/A')

                if plan_qty > 0 or actual_produce > 0 or active_count > 0:
                    self.production_logs.append({
                        'TIME_SLOT': self.current_step,
                        'BATCH_ID': batch_id,
                        'PLAN_PROD_KEY': self.products[p],
                        'OPER_ID': self.processes[s],
                        'ACTIVE_EQP_CNT': active_count,
                        'TRANSITION_EQP_CNT': transition_count,
                        'PRODUCTION_QTY': actual_produce,
                        'REMAIN_WIP': self.wip[p, s],
                        'UTILIZATION_RATE(%)': round(utilization, 2),
                        'CUM_PRODUCED': cum_produced,
                        'PLAN_QTY': plan_qty,
                        'ACHIEVEMENT_RATE(%)': round(achievement_rate, 2)
                    })

        self._resolve_pending_unit_moves()
        self.target_eqp.fill(0)
        next_guidance_gap = self._guidance_allocation_gap()
        
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        if terminated:
            self._flush_conv_queue()
            self.finalize_assignment_history()

        # Reward calculation (Incremental)
        # 1. 이번 시간(Step)에 생산한 양에 비례한 보상
        total_plan = np.sum(self.plan) + 1e-6
        reward = step_production / total_plan
        
        # 2. 장비 이동이 발생한 경우 페널티 부여 (패널티를 0.01에서 0.002로 완화하여 장비 재배치를 유도)
        if len(transfers) > 0:
            reward -= 0.002 * len(transfers)

        if np.any(self.guidance_target_eqp):
            target_total = float(np.sum(self.guidance_target_eqp)) + 1e-6
            reward += 0.5 * ((prev_guidance_gap - next_guidance_gap) / target_total)
            
        # 3. 에피소드 종료 시 최종 계획 달성률에 따른 보너스 부여 (계단식 보상 추가로 Sparse Reward 개선)
        if terminated:
            final_achievement = np.sum(self.produced) / total_plan
            if final_achievement >= 0.99:
                reward += 1.5
            elif final_achievement >= 0.90:
                reward += 0.8
            elif final_achievement >= 0.80:
                reward += 0.4
            elif final_achievement >= 0.60:
                reward += 0.1
            if np.any(self.guidance_target_eqp):
                target_total = float(np.sum(self.guidance_target_eqp)) + 1e-6
                target_match = max(0.0, 1.0 - (next_guidance_gap / target_total))
                reward += target_match
        
        return self._get_obs(), float(reward), terminated, False, {'transfers': transfers}

    def print_final_summary(self, method_name: str = "simulation", save_excel: bool = True):
        """최종 기준에서 제품별 공정별 장비 할당대수와 계획 달성률/가동률 정보를 출력하고 엑셀 파일로 저장합니다."""
        print("\n" + "="*100)
        print(f" [최종 장비 할당 및 실적/가동률 요약 리포트 - {method_name.upper()}]")
        print("="*100)
        
        summary_data = []
        
        # Header formatting: 방법||장비||모델1||모델2||.. ||계획||실적||달성률||전체시간||가동시간||유휴시간||가동률
        headers = ["방법", "장비"] + self.models + ["계획", "실적", "달성률(%)", "전체시간(Hr)", "가동시간(Hr)", "유휴시간(Hr)", "가동률(%)"]
        header_line = "||".join(headers)
        print(header_line)
        print("-" * len(header_line))
        
        for p in range(self.num_prods):
            p_name = self.products[p]
            # Skip padding products
            if "PAD_PROD" in p_name or p_name.startswith("_EMPTY"):
                continue
            for s in range(self.num_procs):
                s_name = self.processes[s]
                if "PAD_PROC" in s_name or s_name.startswith("_EMPTY"):
                    continue
                
                # Check if there is any plan, production, or active equipment
                plan_qty = self.plan[p, s]
                produced_qty = self.produced[p, s]
                active_eqp_list = [self.active_eqp[p, s, m] for m in range(self.num_models)]
                total_active = sum(active_eqp_list)
                
                if plan_qty == 0 and produced_qty == 0 and total_active == 0:
                    continue
                
                row_key = f"{p_name}_{s_name}"
                
                # Equipment model counts
                eqp_counts = [int(count) for count in active_eqp_list] # Store as int for Excel
                
                # Calculations
                achievement = (produced_qty / plan_qty * 100.0) if plan_qty > 0 else 0.0
                total_hr = self.total_eqp_hours[p, s]
                op_hr = self.operating_eqp_hours[p, s]
                idle_hr = max(0.0, total_hr - op_hr)
                util_rate = (op_hr / total_hr * 100.0) if total_hr > 0 else 0.0
                
                row_values_for_print = (
                    [method_name, row_key] + 
                    [f"{count}" for count in eqp_counts] + # Convert back to string for printing
                    [
                        f"{plan_qty:,.1f}", 
                        f"{produced_qty:,.1f}", 
                        f"{achievement:.2f}%", 
                        f"{total_hr:.1f}", 
                        f"{op_hr:.1f}", 
                        f"{idle_hr:.1f}", 
                        f"{util_rate:.2f}%"
                    ]
                )
                print("||".join(row_values_for_print))

                # Collect data for Excel
                summary_row = {
                    "방법": method_name,
                    "장비": row_key,
                }
                for i, model_name in enumerate(self.models):
                    summary_row[model_name] = eqp_counts[i]
                summary_row.update({
                    "계획": plan_qty,
                    "실적": produced_qty,
                    "달성률(%)": round(achievement, 2),
                    "전체시간(Hr)": round(total_hr, 1),
                    "가동시간(Hr)": round(op_hr, 1),
                    "유휴시간(Hr)": round(idle_hr, 1),
                    "가동률(%)": round(util_rate, 2)
                })
                summary_data.append(summary_row)

        print("="*100 + "\n")

        # Save to Excel
        if summary_data and save_excel:
            df_summary = pd.DataFrame(summary_data)
            df_logs = pd.DataFrame()
            if hasattr(self, 'production_logs') and self.production_logs:
                df_logs = pd.DataFrame(self.production_logs)
                if not df_logs.empty:
                    df_logs.rename(columns={
                        'TIME_SLOT': '시간대',
                        'BATCH_ID': '배치ID',
                        'PLAN_PROD_KEY': '제품',
                        'OPER_ID': '공정',
                        'ACTIVE_EQP_CNT': '가동장비대수',
                        'TRANSITION_EQP_CNT': '전이장비대수',
                        'PRODUCTION_QTY': '시간당생산량',
                        'REMAIN_WIP': '남은WIP',
                        'UTILIZATION_RATE(%)': '가동률(%)',
                        'CUM_PRODUCED': '누적생산량',
                        'PLAN_QTY': '계획량',
                        'ACHIEVEMENT_RATE(%)': '달성률(%)'
                    }, inplace=True)
                    df_logs.insert(0, "방법", method_name)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = "logs/simulation_logs"
            import os
            os.makedirs(output_dir, exist_ok=True)
            excel_filename = os.path.join(output_dir, f"final_summary_{method_name}_{timestamp}.xlsx")
            
            # Write sheets
            with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
                df_summary.to_excel(writer, sheet_name="최종 요약", index=False)
                if not df_logs.empty:
                    df_logs.to_excel(writer, sheet_name="시간대별 상세", index=False)
                    
            # Apply styling
            try:
                import openpyxl
                from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
                from openpyxl.utils import get_column_letter
                
                wb = openpyxl.load_workbook(excel_filename)
                
                # Colors and styles
                header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
                header_font = Font(name="Malgun Gothic", size=11, bold=True, color="FFFFFF")
                regular_font = Font(name="Malgun Gothic", size=10)
                
                thin_side = Side(border_style="thin", color="D9D9D9")
                thin_border = Border(left=thin_side, right=thin_side, top=thin_side, bottom=thin_side)
                
                for sheet_name in wb.sheetnames:
                    ws = wb[sheet_name]
                    
                    # Columns styling and sizing
                    for col_idx, col in enumerate(ws.columns, 1):
                        max_len = 0
                        for cell in col:
                            cell.font = regular_font
                            cell.border = thin_border
                            cell.alignment = Alignment(horizontal="center", vertical="center")
                            
                            if isinstance(cell.value, (int, float)):
                                header_val = ws.cell(row=1, column=col_idx).value
                                if header_val and ("달성률" in str(header_val) or "가동률" in str(header_val)):
                                    cell.number_format = '#,##0.0"%"'
                                else:
                                    cell.number_format = '#,##0'
                                    
                            val_str = str(cell.value or '')
                            if len(val_str) > max_len:
                                max_len = len(val_str)
                                
                        col_letter = get_column_letter(col_idx)
                        ws.column_dimensions[col_letter].width = max(max_len + 4, 12)
                        
                    # Style headers
                    for cell in ws[1]:
                        cell.fill = header_fill
                        cell.font = header_font
                        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
                    ws.row_dimensions[1].height = 28
                    
                wb.save(excel_filename)
            except Exception as e:
                print(f"엑셀 스타일 적용 중 오류 발생: {e}")
                
            print(f"요약 및 시간대별 리포트가 '{excel_filename}' 파일로 저장되었습니다.")
