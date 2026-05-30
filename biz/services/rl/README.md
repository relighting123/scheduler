# Scheduler RL Structure

```
biz/services/rl/
  config/       # 설정 인프라: YAML 정책 튜닝, obs/action 스키마 저장·복원
  db/           # DB 인프라: 테이블 DDL, LINEDB EAV 조회·변환, 스냅샷 액세스
  env/          # 핵심 시뮬레이션 도메인: 환경, 관측, 보상, env 팩토리
  train/        # PPO 학습, 행동 복제(BC), 전문가 정책
  validation/   # 벤치마크 CSV 검증 (학습 없음)
  infer/        # DB 추론 결과 출력 (RTD, RTS, Excel)
```

`biz/services/rl_scheduler_service.py`는 앱에서 사용하는 얇은 퍼사드 엔드포인트다.

## 폴더별 역할

### `config/` — 설정 로딩 (비즈니스 로직과 무관한 인프라)
- `policy_config.yaml` — 관측 정규화 스케일과 보상 가중치 (코드 수정 없이 튜닝)
- `policy.py` — YAML을 읽어 `ObservationNormConfig`, `RewardConfig` 반환
- `schema.py` — 여러 스냅샷의 제품·공정·모델 합집합 계산 및 `.schema.json` 저장·복원

### `db/` — DB 인프라 (비즈니스 로직과 무관한 인프라)
- `ddl.py` — `RTS_LINEDSDB_INF` 입력·출력 테이블 `CREATE TABLE` DDL
- `linedb_constants.py` — EAV 테이블 메타, `GBN_CD`, D0/D1 계획 구간
- `linedb_snapshot_sql.py` — Oracle SQL로 필터·집계 후 env용 7종 DataFrame 조회
- `linedb_snapshot_pandas.py` — EAV pandas 폴백 변환
- `training_data_access.py` — `RTS_LINEDSDB_INF` RULE_TIMEKEY 기반 스냅샷 조회

### `env/` — 핵심 시뮬레이션 도메인
- `entities.py` — `ConvJob`, `AssignmentSegment`, `EquipmentUnit` 도메인 엔티티
- `observation.py` — 관측 벡터 레이아웃과 정규화
- `reward.py` — 보상 항목과 터미널 보너스
- `scheduler_env.py` — 메인 시뮬레이터 (`reset` → `step` 순서로 흐름 파악 가능)
- `snapshot_rotation_env.py` — 에피소드마다 무작위 스냅샷 선택 래퍼
- `env_factory.py` — `SchedulerEnvFactory`, `predict_action`, `collect_op20_metrics`

### `train/` — 학습
- `trainer.py` — BC 사전학습 → PPO 학습 → 모델 저장 → 벤치마크 평가
- `expert.py` — `HeuristicExpert`, `OptimalExpert` (BC 레이블 및 벤치마크 기준)
- `callbacks.py` — 학습 곡선 플롯 콜백

### `validation/` — 검증 (학습 없음, CSV 데이터 사용)
- `benchmark_evaluator.py` — Optimal vs Heuristic vs RL 시뮬레이션 비교
- `benchmark_report.py` — 상세 리포트 집계
- `test_data_loader.py` — `test/data/<scenario>/` CSV + `ground_truth.json` 로드

### `infer/` — 추론 결과 출력
- `inference_runner.py` — PPO 모델 로드 → env 루프 → 결과 저장
- `outputs.py` — 할당 피벗, 달성률, Excel 요약
- `rts_output.py` — `RTS_RSLT_MAS` 행 생성 및 DB 저장

## 동작 모드

| 모드 | 진입점 | 데이터 소스 |
|------|--------|-------------|
| **학습** | `train_model(...)` | DB 스냅샷 (`RULE_TIMEKEY` 범위) |
| **검증** | `run_benchmark_evaluation(...)` / `evaluate_on_benchmark_dataset(...)` | `test/data/` CSV |
| **추론** | `run_inference(...)` | DB 단일 스냅샷 |

## 튜닝 위치

| 항목 | 파일 |
|------|------|
| 보상 가중치 / 관측 스케일 | `config/policy_config.yaml` |
| 관측 벡터 레이아웃 | `env/observation.py` |
| 보상 항목 | `env/reward.py` |
| 시뮬레이션 동역학 | `env/scheduler_env.py` |
| DB 쿼리 | `db/linedb_snapshot_sql.py`, `db/training_data_access.py` |
| 테이블 DDL | `db/ddl.py` |
| env 생성·추론 헬퍼 | `env/env_factory.py` |
| 전문가 정책 | `train/expert.py` |

기본 모델 경로: `scheduler_ppo_model` (+ `scheduler_ppo_model.schema.json`)
