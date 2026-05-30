# Scheduler RL (`biz/services/rl`)

스케줄러 강화학습 패키지. **레이어별로 역할을 분리**했으며, 진입점은 상위 `biz/services/rl_scheduler_service.py` 퍼사드 하나다.

## 아키텍처 (레이어)

```
                    rl_scheduler_service.py  (퍼사드: train / infer / validate)
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
     train/                      infer/                    validation/
   (학습 유스케이스)            (추론 유스케이스)            (벤치마크 유스케이스)
        │                           │                           │
        └───────────────────────────┼───────────────────────────┘
                                    ▼
              env/ (시뮬레이션)  +  reporting/ (결과 리포트·DB 산출물)
                                    │
                                    ▼
              db/ (입력 데이터 조회)    config/ (정책 YAML·env 스키마)
```

| 레이어 | 폴더 | 책임 |
|--------|------|------|
| **설정** | `config/` | 관측·보상 튜닝 YAML, 제품/공정/모델 고정 스키마 |
| **데이터** | `db/` | Oracle DDL, EAV 입력 → env 스냅샷 조회 ([상세](db/README.md)) |
| **시뮬레이션** | `env/` | Gymnasium `SchedulerEnv`, 관측·보상·생산·전환 |
| **산출물** | `reporting/` | 할당/달성률 DataFrame, Excel, RTS_RSLT_MAS, 벤치마크 표 |
| **학습** | `train/` | BC·PPO, 전문가 정책, 콜백 |
| **추론** | `infer/` | DB 스냅샷 + PPO 루프 **실행만** (저장 로직은 `reporting/`) |
| **검증** | `validation/` | CSV 벤치마크 시나리오 비교 **실행만** |

## 디렉터리 구조

```
biz/services/rl/
├── README.md
├── config/
│   ├── policy_config.yaml      # 보상·관측 스케일 (튜닝)
│   ├── policy.py               # YAML → RewardConfig / ObservationNormConfig
│   └── env_schema.py           # 제품·공정·모델 합집합 · .schema.json
├── db/                         # 입력 데이터·DDL (README 참고)
├── env/
│   ├── factory.py              # SchedulerEnvFactory, predict_action, collect_op20_metrics
│   ├── scheduler_env.py        # 메인 시뮬레이터
│   ├── snapshot_rotation_env.py
│   ├── entities.py             # ConvJob, EquipmentUnit, …
│   ├── observation.py
│   ├── reward.py
│   ├── production.py
│   └── transfer.py
├── reporting/                  # ★ 추론·벤치마크 공통 결과 처리
│   ├── allocation.py           # 할당·달성률 DataFrame
│   ├── excel.py                # action/production/summary Excel
│   ├── benchmark.py            # 벤치마크 상세·비교 표
│   └── rts_rslt_mas.py         # RTS_RSLT_MAS 행 생성·DB 저장
├── train/
│   ├── trainer.py
│   ├── expert.py
│   └── callbacks.py
├── infer/
│   └── runner.py               # InferenceRunner (오케스트레이션)
└── validation/
    ├── evaluator.py            # BenchmarkEvaluator
    └── fixtures.py             # TestDataLoader (test/data CSV)
```

### `reporting/` — 왜 `infer/`와 분리했는가

이전에는 `infer/outputs.py`와 `infer/rts_output.py`가 공존해 역할이 불명확했고, 벤치마크(`validation/`)도 `infer.outputs`에 의존했다.

| 모듈 | 역할 |
|------|------|
| `allocation.py` | 시뮬 종료 후 **DataFrame** (할당, 마지막 공정 달성률) |
| `excel.py` | **파일·콘솔** 출력 (요약 Excel, 액션/생산 로그) |
| `benchmark.py` | 벤치마크 **상세·비교** 표 |
| `rts_rslt_mas.py` | **RTS_RSLT_MAS** 행 빌드 + Excel + DB INSERT |

`infer/runner.py`와 `validation/evaluator.py`는 시뮬 루프만 돌리고, 결과 가공은 모두 `reporting/`을 호출한다.

## 동작 모드

| 모드 | 퍼사드 메서드 | 데이터 | 핵심 모듈 |
|------|---------------|--------|-----------|
| 학습 | `train_model()` | DB `RULE_TIMEKEY` | `train/`, `db/` |
| 추론 | `run_inference()` | DB 단일 스냅샷 | `infer/runner.py`, `reporting/` |
| 검증 | `run_benchmark_evaluation()` | `test/data/*.csv` | `validation/evaluator.py`, `reporting/` |

## 튜닝·확장 시 보면 될 파일

| 목적 | 위치 |
|------|------|
| 보상·관측 스케일 | `config/policy_config.yaml` |
| 관측 벡터 | `env/observation.py` |
| 보상 식 | `env/reward.py` |
| 시뮬 동역학 | `env/scheduler_env.py` |
| DB 입력·스냅샷 SQL | `db/input_data_snapshot_sql.py` |
| 추론 Excel·RTS DB | `reporting/excel.py`, `reporting/rts_rslt_mas.py` |
| 벤치마크 리포트 | `reporting/benchmark.py` |
| 전문가·BC 레이블 | `train/expert.py` |

기본 모델: `scheduler_ppo_model` (+ `scheduler_ppo_model.schema.json`)

## Import 예시

```python
# 퍼사드 (앱 진입)
from biz.services.rl_scheduler_service import RLSchedulerService

# 벤치마크 CSV
from biz.services.rl.validation.fixtures import TestDataLoader

# 시뮬 결과만 재사용
from biz.services.rl.reporting import build_final_allocation_df, save_rts_rslt_mas
```
