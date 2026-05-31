# 명령어 참조 (CLI · API · 테스트)

프로젝트 루트에서 실행합니다. Python 3 권장.

```bash
pip install -r requirements.txt
```

---

## 한눈에 보기

| 목적 | 학습 필요 | 명령 |
|------|-----------|------|
| **정적 배치** — 계획제품×공정×장비모델별 **몇 대** (권장) | 없음 | `python3 run.py allocate ...` |
| 강화학습 학습 | 있음 | `python3 run.py train ...` |
| 강화학습 추론 (시간대별 시뮬 후 최종 대수) | 학습된 모델 필요 | `python3 run.py infer ...` |
| 벤치마크 평가만 | 학습된 모델 권장 | `python3 run.py benchmark ...` |

`allocate`와 `plan-allocate`는 동일합니다.

---

## 1. 정적 배치 `allocate` (학습 없음)

Input 스냅샷 7종만 사용합니다. 시간대(slot) 시뮬·PPO 학습을 하지 않습니다.

### 기본

```bash
# test/data CSV 사용 (DB 불필요)
python3 run.py allocate --benchmark-dataset benchmark_dataset
```

### DB 스냅샷

```bash
python3 run.py allocate --timekey 20251020070000
```

`--timekey` 생략 시 DB `MAX(RULE_TIMEKEY)` (Oracle 연결 필요).

### 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--benchmark-dataset <ID>` | CSV 시나리오 (`test/data/<ID>/`) | `benchmark_dataset` |
| `--timekey <RULE_TIMEKEY>` | DB 조회 키 (`YYYYMMDDHHMMSS`) | DB MAX |
| `--no-optimize` | 재배치 탐색 없이 **현재 Input 배치**만 평가 | 최적화 수행 |
| `--verbose` | 병목·이동 내역 등 상세 리포트 출력 | 끔 |
| `--output-dir <경로>` | JSON/CSV 저장 폴더 | `output/` |

### 예시

```bash
# 최적화 + 파일 저장
python3 run.py allocate --benchmark-dataset benchmark_dataset

# 현재 배치 달성률만 확인
python3 run.py allocate --benchmark-dataset benchmark_dataset --no-optimize

# 상세 로그 + 다른 저장 경로
python3 run.py allocate --timekey 20251020070000 --verbose --output-dir ./result
```

### 출력

- **콘솔**: `PLAN_PROD_KEY`, `OPER_ID`, `EQP_MODEL_CD`, `EQP_QTY` 표
- **파일**
  - `output/static_allocation_<RULE_TIMEKEY>.json`
  - `output/static_allocation_<RULE_TIMEKEY>.csv`

JSON `allocation_table` 예:

```json
{
  "type": "static_equipment_allocation",
  "rule_timekey": "20251020070000",
  "allocation_table": [
    { "PLAN_PROD_KEY": "P1", "OPER_ID": "OP10", "EQP_MODEL_CD": "MODEL_A", "EQP_QTY": 5 }
  ],
  "achievement": { "overall_percent": 100.0 }
}
```

---

## 2. 강화학습 학습 `train`

1시간(slot) 단위 시뮬레이터에서 장비 이동 정책을 **PPO**로 학습합니다.  
완료 시 프로젝트 루트에 `scheduler_ppo_model.zip` 이 생성됩니다.

### 기본

```bash
python3 run.py train --timekey 20251020070000 --steps 100000
```

### 구간 학습

```bash
python3 run.py train \
  --from-timekey 20251020070000 \
  --to-timekey 20251020120000 \
  --steps 50000
```

### 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--timekey` | 단일 스냅샷 RULE_TIMEKEY | DB MAX |
| `--from-timekey` | 학습 구간 시작 | - |
| `--to-timekey` | 학습 구간 종료 | - |
| `--steps` | PPO 총 타임스텝 | `100000` |
| `--no-init-db` | DB 시나리오 초기화(`init_db_scenario`) 생략 | 초기화 수행 |
| `--no-test-eval` | 학습 후 벤치마크 자동 평가 생략 | 평가 수행 |
| `--benchmark-dataset` | 단일 벤치마크 ID | `benchmark_dataset` |
| `--single-benchmark` | 전체 벤치마크 대신 위 ID 하나만 평가 | 전체 평가 |

### 예시

```bash
python3 run.py train --timekey 20251020070000 --steps 50000 --single-benchmark
python3 run.py train --from-timekey 20251020070000 --to-timekey 20251020120000 --no-init-db
```

---

## 3. 강화학습 추론 `infer`

학습된 모델로 24 slot 시뮬을 돌린 뒤, 최종 대수·전환(`RTD_CONV`) 등을 출력합니다.  
모델이 없으면 무작위 액션으로 동작합니다.

```bash
python3 run.py infer --timekey 20251020070000
python3 run.py infer
```

| 옵션 | 설명 |
|------|------|
| `--timekey` | Input·출력 공통 RULE_TIMEKEY (생략 시 DB MAX) |
| `--no-init-db` | DB 시나리오 초기화 생략 |

---

## 4. 벤치마크 평가 `benchmark`

저장된 PPO 모델로 `test/data/` 벤치마크 시나리오를 평가합니다.

```bash
# 전체 시나리오
python3 run.py benchmark

# 하나만
python3 run.py benchmark --single-benchmark --benchmark-dataset benchmark_dataset
```

---

## 5. API (FastAPI)

서버 기동:

```bash
python3 main.py
# 또는: uvicorn main:app --host 0.0.0.0 --port 8000
```

작업 등록:

```bash
curl -X POST http://127.0.0.1:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "rule_timekey": "20251020070000",
    "action": "static_allocate",
    "parameters": {
      "scenario": "benchmark_dataset",
      "optimize": true,
      "output_dir": "output"
    }
  }'
```

### action 목록

| action | 설명 | 주요 parameters |
|--------|------|-----------------|
| `static_allocate` | 정적 배치 (학습 없음) | `scenario`, `optimize`, `output_dir`, `verbose`, `max_iterations` |
| `plan_allocation` | `static_allocate` 와 동일 | 위와 동일 |
| `rl_train` | PPO 학습 | `total_timesteps`, `rule_timekey`, `from_rule_timekey`, `to_rule_timekey`, `benchmark_dataset` |
| `rl_inference` | PPO 추론 | `rule_timekey` |
| `benchmark` | 벤치마크 평가 | `benchmark_datasets` (배열, 생략 시 전체) |
| `db_check` | DB 연결 확인 | - |

`rule_timekey` 는 요청 본문 최상위 또는 `parameters` 에 넣을 수 있습니다.  
`N/A` 또는 빈 값이면 DB `MAX(RULE_TIMEKEY)` 를 사용합니다.

---

## 6. 테스트·유틸

```bash
# 정적 배치 단위 테스트
python3 test_plan_allocation.py

# RL·벤치마크 (DB/모델 환경 필요)
python3 test_benchmark_dataset.py
python3 test_benchmark.py
```

---

## 7. 데이터 소스

| 방식 | 조건 | allocate 동작 |
|------|------|----------------|
| CSV | `--benchmark-dataset <ID>` 지정 | `test/data/<ID>/` 의 7개 CSV |
| DB | `--benchmark-dataset` 없이 `--timekey` 또는 MAX | `RTS_LINEDSDB_INF` EAV 조회 |

CSV 시나리오 목록: `test/data/*/ground_truth.json` 이 있는 폴더  
예: `benchmark_dataset`, `benchmark_07h_boundary`, …

---

## 8. 도움말

```bash
python3 run.py -h
python3 run.py allocate -h   # mode별로는 동일 parser
```

문의·상세 설계: `guide.md`  
Input 테이블 정의: `guide.md` [4]절, `biz/services/rl/db/README.md`
