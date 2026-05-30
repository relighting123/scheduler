"""SchedulerEnv 인스턴스 생성 및 추론 헬퍼."""

import numpy as np

from biz.services.rl.config.env_schema import load_env_schema
from biz.services.rl.env.scheduler_env import SchedulerEnv


def predict_action(model, obs):
    """PPO 모델로 결정적 액션을 예측한다. 관측 차원이 맞지 않으면 예외를 발생시킨다."""
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    expected = model.observation_space.shape
    if obs.shape != expected:
        raise ValueError(
            f"관측 차원 불일치: env={obs.shape}, model={expected}. "
            "동일 패딩(10×10) env로 재학습하세요."
        )
    action, _ = model.predict(obs, deterministic=True)
    return int(action)


def collect_op20_metrics(env, products=None):
    """OP20 공정의 제품별 계획 달성률을 집계한다."""
    if products is None:
        products = ["P1", "P2", "P3"]
    results = {}
    for p_idx, p_name in enumerate(env.products):
        if p_name not in products:
            continue
        s_idx = env.proc_idx.get("OP20")
        if s_idx is None:
            continue
        prod_qty = env.produced[p_idx, s_idx]
        plan_qty = env.plan[p_idx, s_idx]
        rate = (prod_qty / plan_qty * 100.0) if plan_qty > 0 else 0.0
        results[f"{p_name}_OP20_ACHIEVEMENT"] = round(rate, 2)
    avg = round(sum(results.values()) / len(results), 2) if results else 0.0
    results["AVG_ACHIEVEMENT"] = avg
    return results


class SchedulerEnvFactory:
    """학습 시 저장한 스키마를 적용해 추론·벤치마크용 SchedulerEnv를 생성한다."""

    def __init__(self):
        self.active_env_schema = None

    def make_scheduler_env(self, data, max_steps=24, guidance_target_allocation=None):
        schema = self.active_env_schema or load_env_schema()
        schema_kwargs = {}
        if schema:
            schema_kwargs = {
                "fixed_products": schema.get("products"),
                "fixed_processes": schema.get("processes"),
                "fixed_models": schema.get("models"),
                "max_prods": schema.get("max_prods", len(schema.get("products", []))),
                "max_procs": schema.get("max_procs", len(schema.get("processes", []))),
                "max_models": schema.get("max_models", len(schema.get("models", []))),
            }
            schema_kwargs = {k: v for k, v in schema_kwargs.items() if v}
        else:
            schema_kwargs = {
                "max_prods": SchedulerEnv.DEFAULT_MAX_PRODS,
                "max_procs": SchedulerEnv.DEFAULT_MAX_PROCS,
                "max_models": SchedulerEnv.DEFAULT_MAX_MODELS,
            }
        return SchedulerEnv(
            data=data,
            max_steps=max_steps,
            guidance_target_allocation=guidance_target_allocation,
            **schema_kwargs,
        )
