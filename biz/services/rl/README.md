# Scheduler RL Structure

This package keeps policy logic, environment dynamics, training, evaluation, and output persistence separate.

## Where to change behavior

- `env/scheduler_env.py`
  - Observation/state vector construction
  - Action decoding and transition logic
  - Reward calculation
  - Equipment movement and production simulation

- `expert.py`
  - Heuristic and optimal expert action selection
  - Behavior cloning labels

- `training.py`
  - DB snapshot training flow (`train_model`)
  - Behavior cloning pretraining
  - PPO fine-tuning
  - Benchmark scoring used only for validation/model comparison

- `benchmark_evaluator.py`
  - Benchmark simulation and comparison reports

- `inference_outputs.py`
  - Inference summary tables
  - Action and production log Excel output

- `rts_output.py`
  - RTS_RSLT_MAS row conversion
  - RTS_RSLT_MAS Excel and DB persistence

- `../rl_scheduler_service.py`
  - Public orchestration API used by the app
  - DB fetch/init helpers
  - Thin compatibility wrappers around the modules above

## Training vs inference data

- `train_model(...)` trains from DB snapshots selected by `RULE_TIMEKEY`, `from_rule_timekey`, or `to_rule_timekey`.
- Benchmark CSV scenarios under `test/data` are validation/evaluation data only.
- `run_inference(...)` fetches DB data by `RULE_TIMEKEY` and runs the saved `scheduler_ppo_model`.
- `evaluate_on_benchmark_dataset(...)` and `evaluate_all_benchmark_datasets(...)` run benchmark CSV evaluation.
- `run_benchmark_evaluation(...)` never trains; it only validates the saved model.

The saved model path is shared by default (`scheduler_ppo_model`), so train with `train_model(...)` before inference or benchmark validation.

## Design rule

Keep reward, action, and observation changes inside `env/scheduler_env.py` unless the data contract itself changes. Keep file output, Excel, and DB persistence outside the environment.
