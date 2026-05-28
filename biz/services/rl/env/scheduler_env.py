"""시간대별 설비 배치 강화학습 시뮬레이터 (Gymnasium Env)."""
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from biz.services.rl.config.policy import load_observation_and_reward_config
from biz.services.rl.config.schema import discover_entities_from_data
from biz.services.rl.env.entities import AssignmentSegment, ConvJob, EquipmentUnit
from biz.services.rl.env.observation import (
    ObservationNormConfig,
    build_observation_from_env,
    observation_dim,
)
from biz.services.rl.env.production import compute_production_step
from biz.services.rl.env.reward import RewardConfig, RewardStepContext, compute_reward
from biz.services.rl.env.transfer import execute_transfer_action


class SchedulerEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    DEFAULT_MAX_PRODS = 10
    DEFAULT_MAX_PROCS = 10
    DEFAULT_MAX_MODELS = 10

    # =========================================================================
    # 초기화
    # =========================================================================

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
        env_policy_config_path: Optional[str] = None,
        observation_norm: Optional[ObservationNormConfig] = None,
        reward_config: Optional[RewardConfig] = None,
    ):
        super(SchedulerEnv, self).__init__()
        self.data = data
        loaded_obs, loaded_reward = load_observation_and_reward_config(env_policy_config_path)
        self.observation_norm = observation_norm or loaded_obs
        self.reward_config = reward_config or loaded_reward
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

        if self.num_prods == 0: self.num_prods = 1
        if self.num_procs == 0: self.num_procs = 1
        if self.num_models == 0: self.num_models = 1

        self.action_space = spaces.Discrete(self.num_prods * self.num_procs * self.num_models + 1)

        self.obs_dim = observation_dim(self.num_prods, self.num_procs, self.num_models)
        self.observation_space = spaces.Box(low=0, high=1000, shape=(self.obs_dim,), dtype=np.float32)

        self.reset()

    # =========================================================================
    # Gymnasium 인터페이스: reset / step / print_final_summary
    # =========================================================================

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

        batch_df = self.data.get('batch_tool_info', pd.DataFrame())
        eqp_qty_df = self.data.get('eqp_qty_info', pd.DataFrame())

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

    def step(self, action):
        prev_guidance_gap = self._guidance_allocation_gap()

        self._release_conv_units()

        # 장비 이동 실행 (step 0은 초기 배치 확정 단계이므로 이동 없음)
        transfers = execute_transfer_action(self, action)

        # 생산 시뮬레이션 (용량 계산 → WIP 소모 → 가동률/달성률 기록)
        production_result = compute_production_step(self)
        self.production_logs.extend(log.to_dict() for log in production_result.logs)

        self._resolve_pending_unit_moves()
        self.target_eqp.fill(0)
        next_guidance_gap = self._guidance_allocation_gap()

        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        if terminated:
            self._flush_conv_queue()
            self.finalize_assignment_history()

        # 보상 계산
        reward = compute_reward(
            RewardStepContext(
                step_production=production_result.total_production,
                plan=self.plan,
                produced=self.produced,
                num_transfers=len(transfers),
                prev_guidance_gap=prev_guidance_gap,
                next_guidance_gap=next_guidance_gap,
                guidance_target_eqp=self.guidance_target_eqp,
                terminated=terminated,
            ),
            self.reward_config,
        )

        return self._get_obs(), reward, terminated, False, {"transfers": [t.to_dict() for t in transfers]}

    def print_final_summary(self, method_name: str = "simulation", save_excel: bool = True):
        """최종 장비 할당 및 계획 달성률/가동률 요약을 콘솔에 출력하고 엑셀로 저장한다."""
        print("\n" + "=" * 100)
        print(f" [최종 장비 할당 및 실적/가동률 요약 리포트 - {method_name.upper()}]")
        print("=" * 100)

        summary_data = []
        headers = ["방법", "장비"] + self.models + ["계획", "실적", "달성률(%)", "전체시간(Hr)", "가동시간(Hr)", "유휴시간(Hr)", "가동률(%)"]
        header_line = "||".join(headers)
        print(header_line)
        print("-" * len(header_line))

        for p in range(self.num_prods):
            p_name = self.products[p]
            if "PAD_PROD" in p_name or p_name.startswith("_EMPTY"):
                continue
            for s in range(self.num_procs):
                s_name = self.processes[s]
                if "PAD_PROC" in s_name or s_name.startswith("_EMPTY"):
                    continue

                plan_qty = self.plan[p, s]
                produced_qty = self.produced[p, s]
                active_eqp_list = [self.active_eqp[p, s, m] for m in range(self.num_models)]
                total_active = sum(active_eqp_list)

                if plan_qty == 0 and produced_qty == 0 and total_active == 0:
                    continue

                row_key = f"{p_name}_{s_name}"
                eqp_counts = [int(count) for count in active_eqp_list]

                achievement = (produced_qty / plan_qty * 100.0) if plan_qty > 0 else 0.0
                total_hr = self.total_eqp_hours[p, s]
                op_hr = self.operating_eqp_hours[p, s]
                idle_hr = max(0.0, total_hr - op_hr)
                util_rate = (op_hr / total_hr * 100.0) if total_hr > 0 else 0.0

                row_values_for_print = (
                    [method_name, row_key]
                    + [f"{count}" for count in eqp_counts]
                    + [
                        f"{plan_qty:,.1f}",
                        f"{produced_qty:,.1f}",
                        f"{achievement:.2f}%",
                        f"{total_hr:.1f}",
                        f"{op_hr:.1f}",
                        f"{idle_hr:.1f}",
                        f"{util_rate:.2f}%",
                    ]
                )
                print("||".join(row_values_for_print))

                summary_row = {"방법": method_name, "장비": row_key}
                for i, model_name in enumerate(self.models):
                    summary_row[model_name] = eqp_counts[i]
                summary_row.update({
                    "계획": plan_qty,
                    "실적": produced_qty,
                    "달성률(%)": round(achievement, 2),
                    "전체시간(Hr)": round(total_hr, 1),
                    "가동시간(Hr)": round(op_hr, 1),
                    "유휴시간(Hr)": round(idle_hr, 1),
                    "가동률(%)": round(util_rate, 2),
                })
                summary_data.append(summary_row)

        print("=" * 100 + "\n")

        if summary_data and save_excel:
            self._save_summary_excel(summary_data, method_name)

    # =========================================================================
    # 공개 조회 메서드
    # =========================================================================

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
        """에피소드 종료 시 마지막 할당 구간의 end_step을 확정한다."""
        end_step = self.current_step
        for unit in self.get_all_equipment_units():
            if unit.assignment_history and unit.assignment_history[-1].end_step is None:
                unit.assignment_history[-1].end_step = end_step

    def iter_rts_assignment_records(self) -> List[Tuple[EquipmentUnit, AssignmentSegment]]:
        """RTS_RSLT_MAS 저장을 위한 (장비, 할당 구간) 레코드 목록."""
        self.finalize_assignment_history()
        records: List[Tuple[EquipmentUnit, AssignmentSegment]] = []
        for unit in self.get_all_equipment_units():
            if not unit.assignment_history:
                self._sync_unit_assignment_log(unit)
            for seg in unit.assignment_history:
                records.append((unit, seg))
        return records

    # =========================================================================
    # 내부 헬퍼 — 초기화
    # =========================================================================

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

    def _load_avail_matrix(self):
        """AVAIL_INFO: AVAIL_YN='N'이면 해당 공정에서 진행 불가."""
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
        """처리 불가 조합의 ST를 0으로 만들어 생산·배치 액션에서 제외한다."""
        for p in range(self.num_prods):
            for s in range(self.num_procs):
                for m in range(self.num_models):
                    if not self.avail_matrix[p, s, m]:
                        self.st_matrix[p, s, m] = 0.0

    def _build_equipment_pool(self):
        """EQP_QTY_INFO 기준 모델별 설비 풀을 생성한다."""
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
        """초기 active_eqp 대수에 맞춰 IDLE 풀에서 슬롯으로 배치한다."""
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

    def _relocate_eqp_from_unavailable_slots(self):
        """진행 불가 슬롯에 배치된 설비를 IDLE 풀로 반환한다."""
        for p in range(self.num_prods):
            for s in range(self.num_procs):
                for m in range(self.num_models):
                    if not self.avail_matrix[p, s, m]:
                        key = self._slot_key(p, s, m)
                        units = self.slot_equipment.pop(key, [])
                        for unit in units:
                            self._return_unit_to_idle(unit)
        self._sync_active_eqp_counts()

    def _record_initial_assignments(self):
        """초기 배치·유휴 상태를 장비별 SEQ_NO=1부터 이력에 기록한다."""
        for unit in self.equipment_units:
            unit.assignment_history = []
            unit._logged_attr = None
            self._sync_unit_assignment_log(unit)

    # =========================================================================
    # 내부 헬퍼 — step 처리
    # =========================================================================

    def _get_obs(self):
        return build_observation_from_env(self, self.observation_norm)

    def _guidance_allocation_gap(self) -> float:
        if not np.any(self.guidance_target_eqp):
            return 0.0
        return float(np.sum(np.abs(self.active_eqp - self.guidance_target_eqp)))

    def _needs_tool_conv(
        self,
        unit: EquipmentUnit,
        from_batch: Optional[str],
        to_batch: Optional[str],
        from_is_idle: bool,
    ) -> bool:
        """배치(Tool) 변경 시 1 step 교체 비가용 필요 여부."""
        if from_is_idle:
            return False
        if from_batch is None and to_batch is None:
            return False
        return str(from_batch or "") != str(to_batch or "")

    def _release_conv_units(self, force: bool = False):
        """교체 비가용 기간이 끝난 설비를 목적 슬롯에 배치한다."""
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
        """에피소드 종료 시 남은 CONV 설비를 모두 목적지에 배치한다."""
        self._release_conv_units(force=True)

    def _resolve_pending_unit_moves(self):
        for t_prod, t_proc, t_model, unit in self.pending_unit_moves:
            batch_id = self.batch_id_map.get(
                (self.products[t_prod], self.processes[t_proc])
            )
            self._assign_unit_to_slot(unit, t_prod, t_proc, batch_id)
        self.pending_unit_moves = []
        self._sync_active_eqp_counts()

    # =========================================================================
    # 내부 헬퍼 — 설비 풀 조작
    # =========================================================================

    def _is_available(self, p: int, s: int, m: int) -> bool:
        return bool(self.avail_matrix[p, s, m])

    def _slot_key(self, p: int, s: int, m: int) -> Tuple[int, int, int]:
        return (p, s, m)

    def _sync_active_eqp_counts(self):
        self.active_eqp.fill(0)
        for (p, s, m), units in self.slot_equipment.items():
            self.active_eqp[p, s, m] = float(len(units))
        for m_idx, model in enumerate(self.models):
            self.idle_eqp[m_idx] = float(len(self.idle_equipment.get(model, [])))

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

    def _clear_unit_location(self, unit: EquipmentUnit, log_assignment: bool = False):
        """슬롯에서 설비를 제거한다. CONV·pending 이동 직전에는 IDLE 로그를 생략한다."""
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

    def _enqueue_tool_conv(
        self,
        unit: EquipmentUnit,
        t_prod: int,
        t_proc: int,
        to_batch: Optional[str],
    ):
        """이동 직후 1 step 비가용(CONV)으로 등록하고 다음 step에 목적 슬롯에 배치한다."""
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

    # =========================================================================
    # 내부 헬퍼 — 할당 이력 기록
    # =========================================================================

    def _sync_unit_assignment_log(self, unit: EquipmentUnit):
        """설비 상태가 변경될 때마다 SEQ_NO 기반 이력을 갱신한다."""
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

    def _current_unit_attr(self, unit: EquipmentUnit) -> str:
        return unit.plan_prod_attr_val or "IDLE"

    def _renumber_assignment_history(self, unit: EquipmentUnit):
        for idx, seg in enumerate(unit.assignment_history, start=1):
            seg.seq_no = idx

    def _close_ephemeral_idle_before_conv(self, unit: EquipmentUnit, new_attr: str):
        """슬롯 이탈 직후 같은 step에 생기는 IDLE→CONV 중 IDLE 행을 제거한다."""
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

    def _add_unit_segment_production(self, unit: EquipmentUnit, qty: float):
        if qty <= 0:
            return
        unit.produced_qty += qty
        if unit.assignment_history:
            unit.assignment_history[-1].produced_qty += qty

    # =========================================================================
    # 내부 헬퍼 — 리포트 저장
    # =========================================================================

    def _save_summary_excel(self, summary_data: list, method_name: str):
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
        os.makedirs(output_dir, exist_ok=True)
        excel_filename = os.path.join(output_dir, f"final_summary_{method_name}_{timestamp}.xlsx")

        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            df_summary.to_excel(writer, sheet_name="최종 요약", index=False)
            if not df_logs.empty:
                df_logs.to_excel(writer, sheet_name="시간대별 상세", index=False)

        try:
            import openpyxl
            from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
            from openpyxl.utils import get_column_letter

            wb = openpyxl.load_workbook(excel_filename)

            header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
            header_font = Font(name="Malgun Gothic", size=11, bold=True, color="FFFFFF")
            regular_font = Font(name="Malgun Gothic", size=10)
            thin_side = Side(border_style="thin", color="D9D9D9")
            thin_border = Border(left=thin_side, right=thin_side, top=thin_side, bottom=thin_side)

            for sheet_name in wb.sheetnames:
                ws = wb[sheet_name]
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
                for cell in ws[1]:
                    cell.fill = header_fill
                    cell.font = header_font
                    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
                ws.row_dimensions[1].height = 28

            wb.save(excel_filename)
        except Exception as e:
            print(f"엑셀 스타일 적용 중 오류 발생: {e}")

        print(f"요약 및 시간대별 리포트가 '{excel_filename}' 파일로 저장되었습니다.")
