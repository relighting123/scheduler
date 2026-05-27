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


def product_key_from_attr(plan_prod_attr_val: str) -> str:
    """Merge key: same PLAN_PROD_KEY → one SEQ; IDLE/CONV are separate keys."""
    attr = str(plan_prod_attr_val or "").strip()
    if not attr or attr in ("IDLE", "CONV"):
        return attr or "IDLE"
    if "|" in attr:
        return attr.split("|", 1)[0].strip()
    return attr


def merge_assignment_segments_by_product(segments, max_steps: int):
    """Merge consecutive assignment segments with the same product into one row group."""
    merged = []
    for seg in segments:
        key = product_key_from_attr(seg.plan_prod_attr_val)
        end_step = int(seg.end_step) if seg.end_step is not None else int(max_steps)
        start_step = int(seg.start_step)
        if end_step <= start_step:
            end_step = start_step + 1

        if merged and merged[-1]["product_key"] == key:
            group = merged[-1]
            group["end_step"] = max(group["end_step"], end_step)
            group["produced_qty"] += float(getattr(seg, "produced_qty", 0.0) or 0.0)
            group["plan_prod_attr_val"] = seg.plan_prod_attr_val
            if getattr(seg, "batch_id", None):
                group["batch_id"] = seg.batch_id
        else:
            merged.append(
                {
                    "product_key": key,
                    "start_step": start_step,
                    "end_step": end_step,
                    "produced_qty": float(getattr(seg, "produced_qty", 0.0) or 0.0),
                    "plan_prod_attr_val": seg.plan_prod_attr_val,
                    "batch_id": getattr(seg, "batch_id", None),
                }
            )
    return merged


def build_rts_rslt_mas_rows(env, data, rule_timekey, crt_user_id="SYSTEM"):
    """Convert final env state to RTS_RSLT_MAS rows (장비별·제품 단위 병합 구간)."""
    fallback_start, fallback_end = resolve_simulation_time_range(data)
    max_steps = int(getattr(env, "max_steps", 24))
    base_dt = resolve_simulation_base_datetime(data, rule_timekey=rule_timekey)
    crt_tm = datetime.now().strftime("%Y%m%d%H%M%S")
    rows = []

    if hasattr(env, "finalize_assignment_history"):
        env.finalize_assignment_history()

    all_units = []
    if hasattr(env, "get_all_equipment_units"):
        all_units = env.get_all_equipment_units()

    if all_units and hasattr(all_units[0], "assignment_history"):
        for unit in all_units:
            segments = list(getattr(unit, "assignment_history", []) or [])
            if not segments and hasattr(env, "_sync_unit_assignment_log"):
                env._sync_unit_assignment_log(unit)
                env.finalize_assignment_history()
                segments = list(unit.assignment_history or [])

            merged_groups = merge_assignment_segments_by_product(segments, max_steps)
            unit_cum = 0.0
            for seq_no, group in enumerate(merged_groups, start=1):
                unit_cum += group["produced_qty"]
                rows.append({
                    "RULE_TIMEKEY": str(rule_timekey),
                    "SEQ_NO": seq_no,
                    "EQP_ID": unit.eqp_id,
                    "EQP_MODEL_CD": unit.eqp_model_cd,
                    "BATCH_ID": group["batch_id"],
                    "START_TM": step_to_slot_tm(base_dt, group["start_step"]),
                    "END_TM": step_to_slot_tm(base_dt, group["end_step"]),
                    "PLAN_PROD_ATTR_VAL": group["plan_prod_attr_val"],
                    "PROD_QTY": str(round(group["produced_qty"], 4)),
                    "CUM_PROD_QTY": str(round(unit_cum, 4)),
                    "CRT_USET_ID": crt_user_id,
                    "CRT_TM": crt_tm,
                })
        if rows:
            return rows

    if hasattr(env, "get_deployed_equipment_units"):
        all_units = env.get_deployed_equipment_units()

    if all_units:
        per_hour = float(sum(getattr(u, "produced_qty", 0.0) for u in all_units) or 0.0) / max(
            len(all_units) * max_steps, 1
        )
        for unit in all_units:
            attr = getattr(unit, "plan_prod_attr_val", None) or ""
            rows.append({
                "RULE_TIMEKEY": str(rule_timekey),
                "SEQ_NO": 1,
                "EQP_ID": unit.eqp_id,
                "EQP_MODEL_CD": unit.eqp_model_cd,
                "BATCH_ID": unit.batch_id,
                "START_TM": step_to_slot_tm(base_dt, 0),
                "END_TM": step_to_slot_tm(base_dt, max_steps),
                "PLAN_PROD_ATTR_VAL": attr,
                "PROD_QTY": str(round(float(getattr(unit, "produced_qty", 0.0) or per_hour * max_steps), 4)),
                "CUM_PROD_QTY": str(round(float(getattr(unit, "produced_qty", 0.0) or per_hour * max_steps), 4)),
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
