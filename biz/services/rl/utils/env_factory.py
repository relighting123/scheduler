"""Scheduler environment construction and policy helpers."""

import numpy as np

from biz.services.rl.env.env_schema import load_env_schema
from biz.services.rl.env.scheduler_env import SchedulerEnv


def predict_action(model, obs):
    """Run deterministic PPO predict with observation shape validation."""
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
    """Aggregate plan achievement at OP20 and transfer counts."""
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
    """Build SchedulerEnv instances with optional fixed schema from training."""

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
