"""RTS_RSLT_MAS hourly output: 장비×24시간 생산흐름 검증."""

import pandas as pd

from biz.services.rl.env.scheduler_env import SchedulerEnv
from biz.services.rl.infer.rts_output import build_rts_rslt_mas_rows


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


def test_rts_hourly_rows_per_equipment():
    tk = "20251020070000"
    env = SchedulerEnv(_mini_data(tk), max_steps=24)
    env.reset()
    done = False
    while not done:
        _, _, terminated, truncated, _ = env.step(0)
        done = terminated or truncated

    rows = build_rts_rslt_mas_rows(env, _mini_data(tk), tk)
    eqp_ids = sorted({r["EQP_ID"] for r in rows})
    assert len(eqp_ids) == 2, f"expected 2 equipment, got {eqp_ids}"

    for eqp_id in eqp_ids:
        eqp_rows = [r for r in rows if r["EQP_ID"] == eqp_id]
        assert len(eqp_rows) == 24, f"{eqp_id}: expected 24 hourly rows, got {len(eqp_rows)}"
        starts = [r["START_TM"] for r in eqp_rows]
        assert len(set(starts)) == 24, f"{eqp_id}: duplicate hour slots"
        assert eqp_rows[0]["START_TM"] == "20251020070000"
        assert eqp_rows[-1]["START_TM"] == "20251021060000"
        assert eqp_rows[0]["END_TM"] == "20251020080000"
        seq_nos = [r["SEQ_NO"] for r in eqp_rows]
        assert seq_nos == list(range(1, 25)), f"{eqp_id}: SEQ_NO should be 1..24, got {seq_nos}"
        total_prod = sum(float(r["PROD_QTY"]) for r in eqp_rows)
        assert total_prod > 0
        assert float(eqp_rows[-1]["CUM_PROD_QTY"]) == total_prod

    print("OK: RTS hourly flow", len(rows), "rows for", len(eqp_ids), "equipment")


if __name__ == "__main__":
    test_rts_hourly_rows_per_equipment()
