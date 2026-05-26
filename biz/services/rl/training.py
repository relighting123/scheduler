"""Training orchestration for the scheduler PPO policy."""

import contextlib
import copy
import io

import numpy as np
import torch
import torch.optim as optim
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from torch.utils.data import DataLoader, TensorDataset

from biz.services.rl.callbacks import PlottingCallback
from biz.services.rl.env.env_schema import compute_canonical_schema, save_env_schema
from biz.services.rl.env.scheduler_env import SchedulerEnv
from biz.services.rl.env.snapshot_rotation_env import SnapshotRotationEnv


class BenchmarkTrainer:
    """DB snapshot training flow and benchmark validation scoring."""

    def __init__(self, service):
        self.service = service

    def generate_expert_data(
        self,
        env,
        num_samples=5000,
        expert_cls=None,
        target_allocation=None,
    ):
        """Generate observation/action pairs from an expert policy."""
        from biz.services.rl.expert import HeuristicExpert, OptimalExpert

        expert_cls = expert_cls or HeuristicExpert
        if target_allocation is not None and expert_cls is OptimalExpert:
            expert = expert_cls(env, target_allocation=target_allocation)
        else:
            expert = expert_cls(env)

        obs_list = []
        action_list = []

        obs, _ = env.reset()
        for _ in range(num_samples):
            action = expert.select_action()
            obs_list.append(obs.copy())
            action_list.append(action)

            obs, _, done, truncated, _ = env.step(action)
            if done or truncated:
                obs, _ = env.reset()

        return np.array(obs_list), np.array(action_list)

    def score_benchmark_model(self, model, scenario_payloads, max_steps=24):
        """Score a PPO candidate against benchmark optimal references without report files."""
        from biz.services.rl.benchmark_evaluator import BenchmarkEvaluator
        from biz.services.rl.expert import OptimalExpert

        evaluator = BenchmarkEvaluator(self.service)
        rows = []
        score_parts = []
        for scenario, data, target_alloc in scenario_payloads:
            with contextlib.redirect_stdout(io.StringIO()):
                ref_metrics, _, _ = evaluator.run_simulation(
                    data,
                    max_steps=max_steps,
                    expert_cls=OptimalExpert,
                    method_name="score_optimal",
                    target_allocation=target_alloc,
                )
                rl_metrics, env, _ = evaluator.run_simulation(
                    data,
                    max_steps=max_steps,
                    model=model,
                    method_name="score_rl",
                    target_allocation=target_alloc,
                )

            keys = [
                key for key in ref_metrics
                if key.endswith("_OP20_ACHIEVEMENT") or key == "AVG_ACHIEVEMENT"
            ]
            ach_mae = float(np.mean([
                abs(float(rl_metrics.get(key, 0.0)) - float(ref_metrics.get(key, 0.0)))
                for key in keys
            ])) if keys else 100.0
            transfer_diff = abs(
                float(rl_metrics.get("TRANSFERS", 0.0))
                - float(ref_metrics.get("TRANSFERS", 0.0))
            )
            target_total = float(np.sum(getattr(env, "guidance_target_eqp", 0.0))) + 1e-6
            alloc_gap = (
                float(env._guidance_allocation_gap()) / target_total
                if hasattr(env, "_guidance_allocation_gap") else 0.0
            )
            scenario_score = -ach_mae - (0.25 * transfer_diff) - (10.0 * alloc_gap)
            score_parts.append(scenario_score)
            rows.append({
                "scenario": scenario,
                "rl_avg": rl_metrics.get("AVG_ACHIEVEMENT", 0.0),
                "opt_avg": ref_metrics.get("AVG_ACHIEVEMENT", 0.0),
                "ach_mae": round(ach_mae, 2),
                "transfer_diff": int(transfer_diff),
                "alloc_gap": round(alloc_gap, 3),
            })

        return float(np.mean(score_parts)) if score_parts else float("-inf"), rows

    def train_model(
        self,
        total_timesteps=10000,
        pretrain_bc=True,
        n_envs=4,
        batch_size=64,
        n_steps=1024,
        bc_samples=5000,
        bc_epochs=20,
        ent_coef=0.05,
        rule_timekey=None,
        from_rule_timekey=None,
        to_rule_timekey=None,
        run_test_eval=True,
        benchmark_dataset="benchmark_dataset",
        benchmark_datasets=None,
        evaluate_all_benchmarks=True,
    ):
        """Train on DB snapshots selected by RULE_TIMEKEY, then optionally benchmark."""
        snapshots = self.service.fetch_training_snapshots(
            from_rule_timekey=from_rule_timekey,
            to_rule_timekey=to_rule_timekey,
            rule_timekey=rule_timekey,
        )
        bc_data = snapshots[0]
        pad_p = SchedulerEnv.DEFAULT_MAX_PRODS
        pad_s = SchedulerEnv.DEFAULT_MAX_PROCS
        pad_m = SchedulerEnv.DEFAULT_MAX_MODELS
        fixed_products, fixed_processes, fixed_models = compute_canonical_schema(
            snapshots,
            max_prods=pad_p,
            max_procs=pad_s,
            max_models=pad_m,
        )
        sample_env = SchedulerEnv(
            data=bc_data,
            max_prods=pad_p,
            max_procs=pad_s,
            max_models=pad_m,
            fixed_products=fixed_products,
            fixed_processes=fixed_processes,
            fixed_models=fixed_models,
        )
        self.service._active_env_schema = {
            "products": list(sample_env.products),
            "processes": list(sample_env.processes),
            "models": list(sample_env.models),
            "max_prods": sample_env.num_prods,
            "max_procs": sample_env.num_procs,
            "max_models": sample_env.num_models,
        }
        print(
            f"[학습 env] max_prods={sample_env.num_prods}, "
            f"max_procs={sample_env.num_procs}, max_models={sample_env.num_models}, "
            f"obs_dim={sample_env.observation_space.shape[0]}, "
            f"action_dim={sample_env.action_space.n}"
        )

        def make_env():
            if len(snapshots) == 1:
                return Monitor(
                    SchedulerEnv(
                        data=snapshots[0],
                        max_prods=pad_p,
                        max_procs=pad_s,
                        max_models=pad_m,
                        fixed_products=fixed_products,
                        fixed_processes=fixed_processes,
                        fixed_models=fixed_models,
                    )
                )
            return Monitor(
                SnapshotRotationEnv(
                    snapshots,
                    max_prods=pad_p,
                    max_procs=pad_s,
                    max_models=pad_m,
                    fixed_products=fixed_products,
                    fixed_processes=fixed_processes,
                    fixed_models=fixed_models,
                )
            )

        if n_envs > 1:
            env = SubprocVecEnv([make_env for _ in range(n_envs)])
        else:
            env = DummyVecEnv([make_env])

        model = PPO(
            "MlpPolicy",
            env,
            batch_size=batch_size,
            n_steps=n_steps,
            ent_coef=ent_coef,
            verbose=1,
        )

        if pretrain_bc:
            single_env = SchedulerEnv(
                data=bc_data,
                max_prods=pad_p,
                max_procs=pad_s,
                max_models=pad_m,
                fixed_products=fixed_products,
                fixed_processes=fixed_processes,
                fixed_models=fixed_models,
            )
            expert_obs, expert_actions = self.generate_expert_data(
                single_env,
                num_samples=bc_samples,
            )
            self.pretrain_behavior_cloning(
                model,
                expert_obs,
                expert_actions,
                epochs=bc_epochs,
            )

        print("강화학습(PPO) 시작...")
        callback = PlottingCallback(save_path="learning_curve.png")
        model.learn(total_timesteps=total_timesteps, callback=callback)
        model.save("scheduler_ppo_model")
        schema_path = save_env_schema(
            sample_env.products,
            sample_env.processes,
            sample_env.models,
            model_path="scheduler_ppo_model",
        )
        print("학습 완료 및 모델 저장됨")
        print(f"학습 env 스키마 저장 완료: {schema_path}")

        benchmark_results = None
        if run_test_eval:
            print("\n[학습 후] 벤치마크 데이터셋 기반 성능 비교를 수행합니다...")
            if evaluate_all_benchmarks and benchmark_datasets is None:
                benchmark_results = self.service.evaluate_all_benchmark_datasets()
            else:
                datasets = benchmark_datasets or [benchmark_dataset]
                if len(datasets) == 1:
                    benchmark_results = self.service.evaluate_on_benchmark_dataset(
                        benchmark_dataset=datasets[0],
                    )
                else:
                    benchmark_results = self.service.evaluate_all_benchmark_datasets(
                        datasets=datasets,
                    )
        return benchmark_results

    def pretrain_behavior_cloning(
        self,
        model,
        expert_obs,
        expert_actions,
        epochs=20,
        batch_size=64,
        learning_rate=3e-4,
    ):
        """Supervised behavior cloning for the SB3 policy network."""
        print("모방학습(Behavior Cloning) 사전 학습 시작...")
        policy = model.policy
        optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

        dataset = TensorDataset(
            torch.tensor(expert_obs, dtype=torch.float32).to(policy.device),
            torch.tensor(expert_actions).to(policy.device),
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        policy.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_obs, batch_acts in dataloader:
                _, log_prob, _ = policy.evaluate_actions(batch_obs, batch_acts)
                loss = -log_prob.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")
        print("모방학습 완료.")
