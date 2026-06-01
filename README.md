# Scheduler — 장비 배치 · 계획 달성

계획제품(`PLAN_PROD_KEY`) × 공정(`OPER_ID`) × 장비모델(`EQP_MODEL_CD`) 별 **몇 대 배치할지** 산출하는 프로젝트입니다.

---

## 사전 준비

```bash
pip install -r requirements.txt
```

---

## 실행 파일 · 명령어

### `run.py` — 메인 CLI (권장)

프로젝트 루트에서 실행합니다.

```bash
python3 run.py -h
```

#### 정적 배치 (학습 없음, 조합 탐색)

```bash
# CSV (로컬 샘플)
python3 run.py allocate --benchmark-dataset benchmark_dataset

# 실제 DB (RTS_LINEDSDB_INF)
python3 run.py allocate --use-db --timekey 20251020070000
python3 run.py allocate --use-db   # MAX(RULE_TIMEKEY)
```

| 옵션 | 설명 |
|------|------|
| `--use-db` | Oracle `RTS_LINEDSDB_INF` 조회 (**실데이터 추론·배치 시 필수**) |
| `--benchmark-dataset` | CSV 시나리오 (`--use-db` 없을 때, 기본 `benchmark_dataset`) |
| `--timekey` | RULE_TIMEKEY (`YYYYMMDDHHMMSS`, DB/CSV 필터) |
| `--no-optimize` | 재배치 탐색 없이 현재 Input만 평가 |
| `--verbose` | 상세 리포트 |
| `--output-dir` | JSON/CSV 저장 폴더 (기본 `output/`) |

**산출물**: `output/static_allocation_<RULE_TIMEKEY>.json`, `.csv`

---

#### 정적 배치 RL (몇 대 — 시간 slot 없음)

```bash
# CSV로 학습/추론
python3 run.py train-allocate --benchmark-dataset benchmark_dataset --steps 50000
python3 run.py infer-allocate --benchmark-dataset benchmark_dataset

# 실제 DB로 추론
python3 run.py infer-allocate --use-db --timekey 20251020070000
python3 run.py infer-allocate --use-db --model-path static_allocation_ppo

# 옵션 예
python3 run.py train-allocate --steps 50000 --model-path static_allocation_ppo --no-bc
python3 run.py infer-allocate --model-path static_allocation_ppo --output-dir output
```

| 옵션 | 설명 |
|------|------|
| `--steps` | PPO 타임스텝 (기본 `100000`) |
| `--model-path` | 모델 파일명, 확장자 제외 (기본 `static_allocation_ppo`) |
| `--no-bc` | 모방학습 사전학습 생략 |

**산출물**: `static_allocation_ppo.zip`, `output/static_allocation_rl_*.json`, `.csv`

---

#### 시간대별 RL (1시간 slot 시뮬, 선택)

```bash
python3 run.py train --timekey 20251020070000 --steps 100000
python3 run.py train --from-timekey 20251020070000 --to-timekey 20251020120000 --steps 50000
python3 run.py infer --timekey 20251020070000
python3 run.py benchmark --single-benchmark --benchmark-dataset benchmark_dataset
```

| 옵션 | 설명 |
|------|------|
| `--no-init-db` | DB 시나리오 초기화 생략 |
| `--no-test-eval` | 학습 후 벤치마크 평가 생략 |
| `--single-benchmark` | 벤치마크 1개만 평가 |

**산출물**: `scheduler_ppo_model.zip`, 추론 시 `RTD_CONV` 등 (DB 연동 시)

> **실 DB**: `config.yaml` 의 Oracle 접속 정보를 맞춘 뒤 `--use-db` 를 붙입니다.  
> Input EAV 테이블 **`RTS_LINEDSDB_INF`** 에 해당 `RULE_TIMEKEY` 스냅샷이 있어야 합니다.  
> `train` / `infer` (시간대 RL)는 기본이 DB이며, `infer` 는 결과를 `RTD_CONV` 등에 씁니다.

---

### `main.py` — API 서버 (FastAPI)

```bash
python3 main.py
# 또는
uvicorn main:app --host 0.0.0.0 --port 8000
```

작업 등록 예:

```bash
curl -X POST http://127.0.0.1:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{"action":"static_allocate","parameters":{"scenario":"benchmark_dataset","optimize":true}}'

curl -X POST http://127.0.0.1:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{"action":"rl_train_allocate","parameters":{"scenario":"benchmark_dataset","total_timesteps":50000}}'

curl -X POST http://127.0.0.1:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{"action":"rl_infer_allocate","parameters":{"scenario":"benchmark_dataset"}}'
```

| action | 설명 |
|--------|------|
| `static_allocate` | 정적 배치 (조합 탐색) |
| `rl_train_allocate` | 정적 배치 RL 학습 |
| `rl_infer_allocate` | 정적 배치 RL 추론 |
| `rl_train` / `rl_inference` | 시간대별 RL |
| `benchmark` | 벤치마크 평가 |

---

### 테스트 스크립트

```bash
python3 test_plan_allocation.py          # 정적 배치 단위 테스트
python3 test_benchmark_dataset.py      # RL 벤치마크 (환경·DB 필요할 수 있음)
python3 test_benchmark.py
python3 test_rl.py
```

---

## 모드 한눈에 보기

| `run.py` 모드 | 학습 | 결과 |
|---------------|------|------|
| `allocate` | 없음 | 정적 대수표 (빠름) |
| `train-allocate` → `infer-allocate` | PPO | 정적 대수표 (RL) |
| `train` → `infer` | PPO | slot 시뮬 + 전환 |

---

## 데이터 위치

| 용도 | 경로 |
|------|------|
| CSV 샘플 Input 7종 | `test/data/benchmark_dataset/` |
| 정적 배치 결과 | `output/static_allocation_*` |
| 정적 RL 결과 | `output/static_allocation_rl_*` |
| 정적 RL 모델 | `static_allocation_ppo.zip` |
| 시간대 RL 모델 | `scheduler_ppo_model.zip` |

---

## 추가 문서

- **[COMMANDS.md](COMMANDS.md)** — CLI·API·옵션 상세
- **[guide.md](guide.md)** — Input/Output 테이블·시스템 구조
