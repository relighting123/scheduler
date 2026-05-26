"""Run PPO inference on a DB snapshot and persist outputs."""

from datetime import datetime

from stable_baselines3 import PPO

from biz.services.rl.infer.outputs import (
    build_final_allocation_df,
    build_last_process_achievement_df,
    save_action_results,
    save_inference_summary,
    save_production_logs,
)
from biz.services.rl.infer.rts_output import save_rts_rslt_mas
from biz.services.rl.utils.env_factory import predict_action


class InferenceRunner:
    """Execute inference loop and write RTD/RTS artifacts."""

    def __init__(self, service):
        self._svc = service

    def run(self, rule_timekey=None):
        resolved_timekey = self._svc.data.resolve_rule_timekey(rule_timekey)
        print(f"[추론] RULE_TIMEKEY={resolved_timekey} (입력 스냅샷·결과 출력 공통)")
        data = self._svc.data.fetch_data(rule_timekey=resolved_timekey)

        model_path = "scheduler_ppo_model"
        model = None
        try:
            model = PPO.load(model_path)
        except Exception:
            print("저장된 모델이 없습니다. 임의의 액션으로 시뮬레이션 합니다.")

        env = self._svc.env_factory.make_scheduler_env(data)
        if model is not None:
            print(
                f"[추론 env] obs={env.observation_space.shape}, "
                f"model obs={model.observation_space.shape}"
            )

        obs, _ = env.reset()
        done = False
        results = []

        while not done:
            if model:
                try:
                    action = predict_action(model, obs)
                except ValueError as exc:
                    print(f"[경고] 모델/env 차원 불일치 - 무작위 액션으로 전환합니다. {exc}")
                    model = None
                    action = env.action_space.sample()
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated

            if "transfers" in info:
                for transfer in info["transfers"]:
                    results.append(
                        {
                            "RULE_TIMEKEY": resolved_timekey,
                            "FROM_PLAN_PROD_KEY": transfer["FROM_PROD"],
                            "FROM_OPER_ID": transfer["FROM_PROC"],
                            "EQP_MODEL_CD": transfer["MODEL"],
                            "TO_PLAN_PROD_KEY": transfer["TO_PROD"],
                            "TO_OPER_ID": transfer["TO_PROC"],
                            "START_CONV_TIME": datetime.now().strftime("%Y%m%d%H%M%S"),
                            "EQP_QTY": 1,
                        }
                    )

        env.print_final_summary(method_name="inference")
        save_action_results(results)
        if hasattr(env, "production_logs") and env.production_logs:
            save_production_logs(env.production_logs)

        allocation_df = build_final_allocation_df(env)
        achievement_df = build_last_process_achievement_df(env, data)
        save_inference_summary(resolved_timekey, allocation_df, achievement_df)
        save_rts_rslt_mas(self._svc.db, env, data, resolved_timekey)

        return results
