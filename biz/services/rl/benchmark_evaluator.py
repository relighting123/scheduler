"""벤치마크 시뮬레이션·리포트 (학습 후 성능 검증)."""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from stable_baselines3 import PPO

from biz.services.rl.benchmark_report import (
    build_comparison_row,
    build_scenario_detail_df,
    capture_initial_allocation,
    print_scenario_detail_report,
)


class BenchmarkEvaluator:
    """RLSchedulerService에 위임되는 벤치마크 평가 로직."""

    def __init__(self, service):
        self._svc = service

    def run_simulation(
        self,
        data,
        max_steps=24,
        model=None,
        expert_cls=None,
        method_name='simulation',
        target_allocation=None,
    ):
        svc = self._svc
        env = svc._make_scheduler_env(data, max_steps=max_steps)
        expert = None
        if expert_cls is not None:
            if target_allocation is not None and expert_cls.__name__ == 'OptimalExpert':
                expert = expert_cls(env, target_allocation=target_allocation)
            else:
                expert = expert_cls(env)

        obs, _ = env.reset()
        initial_eqp = capture_initial_allocation(env)
        transfers = 0
        done = False
        while not done:
            if expert is not None:
                action = expert.select_action()
            elif model is not None:
                action = svc._predict_action(model, obs)
            else:
                action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if info.get('transfers'):
                transfers += len(info['transfers'])
            done = terminated or truncated

        env.print_final_summary(method_name=method_name, save_excel=False)
        metrics = svc._collect_op20_metrics(env)
        metrics['TRANSFERS'] = transfers
        detail_df, avg_util = build_scenario_detail_df(env, initial_eqp, method_name)
        metrics['AVG_UTILIZATION'] = avg_util
        return metrics, env, detail_df

    def write_excel_sheets(self, writer, scenario_id, comparison_df, detail_by_method, optimal_ref):
        prefix = scenario_id[:20]
        comparison_df.to_excel(writer, sheet_name=f'{prefix}_COMP', index=False)
        pd.DataFrame([optimal_ref]).to_excel(writer, sheet_name=f'{prefix}_REF', index=False)
        sheet_names = {
            '1. 정답지 (Optimal GT)': f'{prefix}_OPT',
            '2. 휴리스틱 (Expert)': f'{prefix}_HEU',
            '3. 강화학습 (RL PPO)': f'{prefix}_RL',
        }
        for method_label, detail_df in detail_by_method.items():
            sheet = sheet_names.get(method_label, f'{prefix}_DET')[:31]
            detail_df.to_excel(writer, sheet_name=sheet, index=False)

    def evaluate_dataset(
        self,
        benchmark_dataset='benchmark_dataset',
        model_path='scheduler_ppo_model',
        max_steps=24,
        save_excel=True,
        excel_writer=None,
    ) -> Dict[str, Any]:
        from biz.services.rl.test_data_loader import TestDataLoader
        from biz.services.rl.expert import HeuristicExpert, OptimalExpert

        svc = self._svc
        print("\n" + "=" * 80)
        print(f" [벤치마크 데이터셋 성능 비교 — dataset: {benchmark_dataset}]")
        print("=" * 80)

        loader = TestDataLoader()
        data = loader.load_for_env(benchmark_dataset)
        ground_truth = loader.load_ground_truth(benchmark_dataset)
        target_alloc = ground_truth.get('target_allocation')

        print("\n[1] 정답지(Optimal Ground Truth) 시뮬레이션...")
        gt_metrics, _, gt_detail = self.run_simulation(
            data, max_steps=max_steps, expert_cls=OptimalExpert,
            method_name='optimal_test', target_allocation=target_alloc,
        )
        print("\n[2] 휴리스틱(Heuristic Expert) 시뮬레이션...")
        heu_metrics, _, heu_detail = self.run_simulation(
            data, max_steps=max_steps, expert_cls=HeuristicExpert, method_name='heuristic_test',
        )
        print("\n[3] 강화학습(RL PPO) 시뮬레이션...")
        try:
            model = PPO.load(model_path)
        except Exception:
            print(f"[경고] 모델 '{model_path}' 없음 — 무작위 액션으로 평가합니다.")
            model = None
        rl_metrics, env_rl, rl_detail = self.run_simulation(
            data, max_steps=max_steps, model=model, method_name='rl_test',
        )

        comparison_df = pd.DataFrame([
            build_comparison_row('1. 정답지 (Optimal GT)', gt_metrics, gt_metrics.get('AVG_UTILIZATION', 0)),
            build_comparison_row('2. 휴리스틱 (Expert)', heu_metrics, heu_metrics.get('AVG_UTILIZATION', 0)),
            build_comparison_row('3. 강화학습 (RL PPO)', rl_metrics, rl_metrics.get('AVG_UTILIZATION', 0)),
        ])
        detail_by_method = {
            '1. 정답지 (Optimal GT)': gt_detail,
            '2. 휴리스틱 (Expert)': heu_detail,
            '3. 강화학습 (RL PPO)': rl_detail,
        }
        for label, detail_df in detail_by_method.items():
            avg_util = float(comparison_df.loc[comparison_df['Method'] == label, '평균 장비가동률(%)'].iloc[0])
            print_scenario_detail_report(benchmark_dataset, label, detail_df, avg_util)

        print("\n" + "=" * 80)
        print(f" [벤치마크 비교 요약 — {benchmark_dataset}]")
        print("=" * 80)
        print(comparison_df.to_string(index=False))
        print("=" * 80)

        ref = ground_truth.setdefault('optimal_reference', {})
        for k, v in gt_metrics.items():
            ref[k] = v

        if save_excel:
            log_dir = os.path.join(os.getcwd(), 'logs', 'simulation_logs')
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(
                log_dir, f'benchmark_dataset_eval_{benchmark_dataset}_{timestamp}.xlsx'
            )
            with pd.ExcelWriter(report_path) as writer:
                self.write_excel_sheets(writer, benchmark_dataset, comparison_df, detail_by_method, ref)
            print(f"[성공] 벤치마크 평가 리포트 저장: {report_path}")
        elif excel_writer is not None:
            self.write_excel_sheets(excel_writer, benchmark_dataset, comparison_df, detail_by_method, ref)

        rl_allocation_df = svc._build_final_allocation_df(env_rl)
        rl_achievement_df = svc._build_last_process_achievement_df(env_rl, data)
        svc.save_inference_summary(
            rule_timekey=ground_truth.get('rule_timekey', 'test'),
            allocation_df=rl_allocation_df,
            achievement_df=rl_achievement_df,
            file_prefix=f'benchmark_inference_{benchmark_dataset}',
        )
        return {
            'scenario_id': benchmark_dataset,
            'comparison_df': comparison_df,
            'detail_by_method': detail_by_method,
            'ground_truth': ground_truth,
        }

    def evaluate_all(
        self,
        datasets: Optional[List[str]] = None,
        model_path='scheduler_ppo_model',
        max_steps=24,
    ) -> Dict[str, Any]:
        from biz.services.rl.test_data_loader import TestDataLoader

        loader = TestDataLoader()
        datasets = datasets or loader.list_scenarios() or ['benchmark_dataset']
        print("\n" + "#" * 80)
        print(f" [전체 벤치마크 평가 — {len(datasets)}개: {', '.join(datasets)}]")
        print("#" * 80)

        results = {}
        log_dir = os.path.join(os.getcwd(), 'logs', 'simulation_logs')
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unified_path = os.path.join(log_dir, f'benchmark_unified_report_{timestamp}.xlsx')

        with pd.ExcelWriter(unified_path) as writer:
            summary_rows = []
            for scenario in datasets:
                report = self.evaluate_dataset(
                    benchmark_dataset=scenario,
                    model_path=model_path,
                    max_steps=max_steps,
                    save_excel=False,
                    excel_writer=writer,
                )
                results[scenario] = report
                comp = report['comparison_df'].copy()
                comp.insert(0, '데이터셋', scenario)
                summary_rows.append(comp)
            if summary_rows:
                pd.concat(summary_rows, ignore_index=True).to_excel(
                    writer, sheet_name='ALL_SUMMARY', index=False
                )

        print(f"\n[성공] 통합 벤치마크 리포트: {unified_path}")
        return results
