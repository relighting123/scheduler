# Scheduler RL Structure

```
biz/services/rl/
  env/          # Gymnasium environments (observation, action, reward)
  train/        # PPO training, behavior cloning, experts
  validation/   # Benchmark CSV evaluation (no training)
  infer/        # DB inference outputs (RTD, RTS, Excel)
  utils/        # DB access, DDL, env factory
```

`biz/services/rl_scheduler_service.py` is the thin endpoint facade used by the app.

## Where to change behavior

- `env/scheduler_env.py` — observation, action, reward, simulation
- `train/expert.py` — heuristic/optimal experts and BC labels
- `train/trainer.py` — `train_model`, behavior cloning, PPO
- `validation/benchmark_evaluator.py` — benchmark simulation and reports
- `infer/outputs.py` — inference summary and action/production logs
- `infer/rts_output.py` — RTS_RSLT_MAS rows and DB persistence
- `utils/data_access.py` — DB snapshot fetch and RULE_TIMEKEY filtering
- `utils/db_schema.py` — input/output table DDL

## Training vs validation vs inference

- `train_model(...)` trains from DB snapshots (`RULE_TIMEKEY` range).
- `run_benchmark_evaluation(...)` / `evaluate_on_benchmark_dataset(...)` use CSV data under `test/data` only (validation).
- `run_inference(...)` runs the saved model on a DB snapshot (infer).

The default model path is `scheduler_ppo_model`.
