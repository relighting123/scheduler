"""RTS_RSLT_MAS row building and persistence helpers."""

import os
from datetime import datetime, timedelta

import pandas as pd


def format_timekey_14(time_value):
    """Normalize a time value to YYYYMMDDHHMMSS."""
    return str(time_value).strip().ljust(14, "0")[:14]


def resolve_simulation_time_range(data):
    """Resolve whole simulation START_TM and END_TM from PLAN_INFO."""
    plan_df = data.get("plan_info", pd.DataFrame())
    if plan_df.empty or not {"START_TIME", "END_TIME"}.issubset(plan_df.columns):
        now = datetime.now().strftime("%Y%m%d%H%M%S")
        return now, now
    start_tm = format_timekey_14(plan_df["START_TIME"].astype(str).min())
    end_tm = format_timekey_14(plan_df["END_TIME"].astype(str).max())
    return start_tm, end_tm


def resolve_simulation_base_datetime(data, rule_timekey=None):
    """Resolve the base datetime for step-to-slot conversion."""
    plan_df = data.get("plan_info", pd.DataFrame())
    if not plan_df.empty and "START_TIME" in plan_df.columns:
        base_str = format_timekey_14(plan_df["START_TIME"].astype(str).min())
    elif rule_timekey and str(rule_timekey) not in ("", "N/A", "benchmark"):
        base_str = format_timekey_14(rule_timekey)
    else:
        base_str = datetime.now().strftime("%Y%m%d%H%M%S")

    if len(base_str) >= 14:
        return datetime.strptime(base_str[:14], "%Y%m%d%H%M%S")
    if len(base_str) >= 10:
        return datetime.strptime(base_str[:10], "%Y%m%d%H")
    return datetime.strptime(base_str[:8] + "00", "%Y%m%d%H")


def step_to_slot_tm(base_dt, step_index: int) -> str:
    """Convert an env step index to a slot start timestamp."""
    return (base_dt + timedelta(hours=int(step_index))).strftime("%Y%m%d%H%M%S")


def segment_slot_time_range(data, seg, max_steps: int, rule_timekey=None):
    """Resolve START_TM and END_TM for one assignment segment."""
    base_dt = resolve_simulation_base_datetime(data, rule_timekey=rule_timekey)
    start_step = int(seg.start_step)
    end_step = seg.end_step
    if end_step is None:
        end_step = min(start_step + 1, max_steps)
    else:
        end_step = int(end_step)
    if end_step <= start_step:
        end_step = start_step + 1
    return step_to_slot_tm(base_dt, start_step), step_to_slot_tm(base_dt, end_step)


def _append_hourly_rts_row(
    rows,
    *,
    rule_timekey,
    unit,
    seg,
    step_index,
    seq_no,
    max_steps,
    data,
    crt_user_id,
    crt_tm,
    cum_produced,
):
    """Append one RTS_RSLT_MAS row for a single simulation hour slot."""
    base_dt = resolve_simulation_base_datetime(data, rule_timekey=rule_timekey)
    slot_start = step_to_slot_tm(base_dt, step_index)
    slot_end = step_to_slot_tm(base_dt, min(step_index + 1, max_steps))
    hour_qty = 0.0
    if hasattr(seg, "hourly_produced") and seg.hourly_produced:
        hour_qty = float(seg.hourly_produced.get(int(step_index), 0.0))
    elif seg.end_step is not None:
        span = max(int(seg.end_step) - int(seg.start_step), 1)
        hour_qty = float(seg.produced_qty) / span if int(seg.start_step) == step_index else 0.0
    rows.append({
        "RULE_TIMEKEY": str(rule_timekey),
        "SEQ_NO": int(seq_no),
        "EQP_ID": unit.eqp_id,
        "EQP_MODEL_CD": unit.eqp_model_cd,
        "BATCH_ID": seg.batch_id,
        "START_TM": slot_start,
        "END_TM": slot_end,
        "PLAN_PROD_ATTR_VAL": seg.plan_prod_attr_val,
        "PROD_QTY": str(round(hour_qty, 4)),
        "CUM_PROD_QTY": str(round(float(cum_produced), 4)),
        "CRT_USET_ID": crt_user_id,
        "CRT_TM": crt_tm,
    })


def build_rts_rslt_mas_rows(env, data, rule_timekey, crt_user_id="SYSTEM"):
    """Convert final env state to RTS_RSLT_MAS rows (장비×1시간 slot 단위, 24h 생산흐름)."""
    fallback_start, fallback_end = resolve_simulation_time_range(data)
    max_steps = int(getattr(env, "max_steps", 24))
    crt_tm = datetime.now().strftime("%Y%m%d%H%M%S")
    rows = []

    all_units = []
    if hasattr(env, "get_all_equipment_units"):
        all_units = env.get_all_equipment_units()
    elif hasattr(env, "get_deployed_equipment_units"):
        all_units = env.get_deployed_equipment_units()

    if hasattr(env, "iter_rts_hourly_records"):
        unit_cum = {unit.eqp_id: 0.0 for unit in env.get_all_equipment_units()}
        for unit, seg, step_index in env.iter_rts_hourly_records(max_steps=max_steps):
            hour_qty = 0.0
            if hasattr(seg, "hourly_produced") and seg.hourly_produced:
                hour_qty = float(seg.hourly_produced.get(int(step_index), 0.0))
            unit_cum[unit.eqp_id] = unit_cum.get(unit.eqp_id, 0.0) + hour_qty
            _append_hourly_rts_row(
                rows,
                rule_timekey=rule_timekey,
                unit=unit,
                seg=seg,
                step_index=step_index,
                seq_no=step_index + 1,
                max_steps=max_steps,
                data=data,
                crt_user_id=crt_user_id,
                crt_tm=crt_tm,
                cum_produced=unit_cum[unit.eqp_id],
            )
        return rows

    if hasattr(env, "iter_rts_assignment_records"):
        for unit in env.get_all_equipment_units() if hasattr(env, "get_all_equipment_units") else []:
            unit_cum = 0.0
            records = [
                (u, seg)
                for u, seg in env.iter_rts_assignment_records()
                if u.eqp_id == unit.eqp_id
            ]
            for _, seg in records:
                end_step = seg.end_step if seg.end_step is not None else max_steps
                for step_index in range(int(seg.start_step), int(end_step)):
                    hour_qty = 0.0
                    if hasattr(seg, "hourly_produced") and seg.hourly_produced:
                        hour_qty = float(seg.hourly_produced.get(step_index, 0.0))
                    unit_cum += hour_qty
                    _append_hourly_rts_row(
                        rows,
                        rule_timekey=rule_timekey,
                        unit=unit,
                        seg=seg,
                        step_index=step_index,
                        seq_no=step_index + 1,
                        max_steps=max_steps,
                        data=data,
                        crt_user_id=crt_user_id,
                        crt_tm=crt_tm,
                        cum_produced=unit_cum,
                    )
        if rows:
            return rows

    if all_units:
        base_dt = resolve_simulation_base_datetime(data, rule_timekey)
        per_hour = float(sum(u.produced_qty for u in all_units) or 0.0) / max(
            len(all_units) * max_steps, 1
        )
        for unit in all_units:
            unit_cum = 0.0
            for step_index in range(max_steps):
                unit_cum += per_hour
                rows.append({
                    "RULE_TIMEKEY": str(rule_timekey),
                    "SEQ_NO": step_index + 1,
                    "EQP_ID": unit.eqp_id,
                    "EQP_MODEL_CD": unit.eqp_model_cd,
                    "BATCH_ID": unit.batch_id,
                    "START_TM": step_to_slot_tm(base_dt, step_index),
                    "END_TM": step_to_slot_tm(base_dt, min(step_index + 1, max_steps)),
                    "PLAN_PROD_ATTR_VAL": unit.plan_prod_attr_val or "",
                    "PROD_QTY": str(round(per_hour, 4)),
                    "CUM_PROD_QTY": str(round(unit_cum, 4)),
                    "CRT_USET_ID": crt_user_id,
                    "CRT_TM": crt_tm,
                })
        return rows

    for p_idx, prod in enumerate(env.products):
        if prod.startswith("PAD_PROD_") or prod.startswith("_EMPTY"):
            continue
        for s_idx, oper in enumerate(env.processes):
            if oper.startswith("PAD_PROC_") or oper.startswith("_EMPTY"):
                continue
            batch_id = env.batch_id_map.get((prod, oper), "EQP")
            produced_qty = float(env.produced[p_idx, s_idx])
            for m_idx, model in enumerate(env.models):
                eqp_count = int(round(float(env.active_eqp[p_idx, s_idx, m_idx])))
                if eqp_count <= 0:
                    continue
                prod_qty_str = str(round(produced_qty / eqp_count, 4))
                for eqp_seq in range(1, eqp_count + 1):
                    rows.append({
                        "RULE_TIMEKEY": str(rule_timekey),
                        "SEQ_NO": 1,
                        "EQP_ID": f"{model}-{eqp_seq:05d}",
                        "EQP_MODEL_CD": model,
                        "BATCH_ID": batch_id,
                        "START_TM": fallback_start,
                        "END_TM": fallback_end,
                        "PLAN_PROD_ATTR_VAL": f"{prod}|{oper}",
                        "PROD_QTY": prod_qty_str,
                        "CUM_PROD_QTY": prod_qty_str,
                        "CRT_USET_ID": crt_user_id,
                        "CRT_TM": crt_tm,
                    })
    return rows


def save_rts_rslt_mas(db, env, data, rule_timekey, crt_user_id="SYSTEM"):
    """Print, save, and optionally persist RTS_RSLT_MAS output rows."""
    rows = build_rts_rslt_mas_rows(env, data, rule_timekey, crt_user_id=crt_user_id)
    if not rows:
        print("\n[RTS_RSLT_MAS] 저장할 결과가 없습니다.")
        return rows

    df = pd.DataFrame(rows)
    print(f"\n[RTS_RSLT_MAS] 총 {len(df)}건 Output")
    print("-" * 80)
    print(df.to_string(index=False))
    print("-" * 80)

    log_dir = os.path.join(os.getcwd(), "logs", "simulation_logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_timekey = str(rule_timekey).replace(" ", "_")
    file_path = os.path.join(log_dir, f"rts_rslt_mas_{safe_timekey}_{timestamp}.xlsx")
    df.to_excel(file_path, index=False)
    print(f"[성공] RTS_RSLT_MAS Output이 엑셀로 저장되었습니다: {file_path}")

    if db is None:
        return rows

    try:
        db.execute(
            "DELETE FROM RTS_RSLT_MAS WHERE RULE_TIMEKEY = :tk",
            {"tk": str(rule_timekey)},
        )
        db.bulk_execute(
            """
            INSERT INTO RTS_RSLT_MAS (
                RULE_TIMEKEY, SEQ_NO, EQP_ID, EQP_MODEL_CD, BATCH_ID, START_TM, END_TM,
                PLAN_PROD_ATTR_VAL, PROD_QTY, CUM_PROD_QTY, CRT_USET_ID, CRT_TM
            ) VALUES (
                :RULE_TIMEKEY, :SEQ_NO, :EQP_ID, :EQP_MODEL_CD, :BATCH_ID, :START_TM, :END_TM,
                :PLAN_PROD_ATTR_VAL, :PROD_QTY, :CUM_PROD_QTY, :CRT_USET_ID, :CRT_TM
            )
            """,
            rows,
        )
        print(f"[성공] RTS_RSLT_MAS {len(rows)}건 DB 저장 완료 (RULE_TIMEKEY={rule_timekey})")
    except Exception as exc:
        if "ORA-00942" in str(exc) or "table or view does not exist" in str(exc).lower():
            print(
                "[안내] RTS_RSLT_MAS 테이블이 없어 DB 저장을 건너뜁니다. "
                "schema/scheduler_tables.sql을 적용하세요."
            )
        else:
            print(f"[경고] RTS_RSLT_MAS DB 저장 실패: {exc}")
    return rows
