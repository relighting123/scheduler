"""RTS_RSLT_MAS: 장비별·동일 제품 구간 병합 및 SEQ 증가 검증."""

import pandas as pd

from biz.services.rl.env.scheduler_env import AssignmentSegment, SchedulerEnv
from biz.services.rl.infer.rts_output import (
    build_rts_rslt_mas_rows,
    merge_assignment_segments_by_product,
    product_key_from_attr,
)


def _mini_data(rule_timekey: str):
    return {
        "wip_info": pd.DataFrame(
            [
                {
                    "RULE_TIMEKEY": rule_timekey,
                    "PLAN_PROD_KEY": "P1",
                    "OPER_ID": "OP10",
                    "OPER_SEQ": 10,
                    "WIP_QTY": 5000,
                }
            ]
        ),
        "plan_info": pd.DataFrame(
            [
                {
                    "RULE_TIMEKEY": rule_timekey,
                    "PLAN_PROD_KEY": "P1",
                    "OPER_ID": "OP10",
                    "START_TIME": "20251020070000",
                    "END_TIME": "20251020230000",
                    "PLAN_QTY": 2400,
                }
            ]
        ),
        "uph_info": pd.DataFrame(
            [
                {
                    "RULE_TIMEKEY": rule_timekey,
                    "PLAN_PROD_KEY": "P1",
                    "OPER_ID": "OP10",
                    "EQP_MODEL_CD": "MODEL_A",
                    "UPH": 100,
                }
            ]
        ),
        "eqp_qty_info": pd.DataFrame(
            [
                {
                    "RULE_TIMEKEY": rule_timekey,
                    "BATCH_ID": "B1",
                    "EQP_MODEL_CD": "MODEL_A",
                    "EQP_QTY": 2,
                }
            ]
        ),
        "batch_tool_info": pd.DataFrame(
            [
                {
                    "RULE_TIMEKEY": rule_timekey,
                    "BATCH_ID": "B1",
                    "PLAN_PROD_KEY": "P1",
                    "OPER_ID": "OP10",
                }
            ]
        ),
        "avail_info": pd.DataFrame(),
    }


def test_merge_same_product_different_oper():
    segments = [
        AssignmentSegment(1, "P1|OP10", "B1", 0, 8, 400.0),
        AssignmentSegment(2, "P1|OP20", "B1", 8, 16, 300.0),
        AssignmentSegment(3, "P2|OP10", "B2", 16, 24, 200.0),
    ]
    merged = merge_assignment_segments_by_product(segments, 24)
    assert len(merged) == 2
    assert merged[0]["product_key"] == "P1"
    assert merged[0]["start_step"] == 0
    assert merged[0]["end_step"] == 16
    assert merged[0]["produced_qty"] == 700.0
    assert merged[0]["plan_prod_attr_val"] == "P1|OP20"
    assert merged[1]["product_key"] == "P2"
    assert product_key_from_attr("P1|OP10") == "P1"
    assert product_key_from_attr("IDLE") == "IDLE"


def test_rts_single_product_merged_row():
    tk = "20251020070000"
    env = SchedulerEnv(_mini_data(tk), max_steps=24)
    env.reset()
    done = False
    while not done:
        _, _, terminated, truncated, _ = env.step(0)
        done = terminated or truncated

    rows = build_rts_rslt_mas_rows(env, _mini_data(tk), tk)
    eqp_ids = sorted({r["EQP_ID"] for r in rows})
    assert len(eqp_ids) == 2

    for eqp_id in eqp_ids:
        eqp_rows = [r for r in rows if r["EQP_ID"] == eqp_id]
        assert len(eqp_rows) == 1, f"{eqp_id}: same product should merge to 1 row"
        row = eqp_rows[0]
        assert row["SEQ_NO"] == 1
        assert row["START_TM"] == "20251020070000"
        assert row["END_TM"] == "20251021070000"
        assert row["PLAN_PROD_ATTR_VAL"] == "P1|OP10"
        assert float(row["PROD_QTY"]) > 0

    print("OK: RTS merged by product", len(rows), "rows")


if __name__ == "__main__":
    test_merge_same_product_different_oper()
    test_rts_single_product_merged_row()
