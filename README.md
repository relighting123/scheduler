# Scheduler — 장비 배치 · 계획 달성

계획제품(PLAN_PROD_KEY) × 공정(OPER_ID) × 장비모델(EQP_MODEL_CD) 별 **몇 대 배치할지** 분석하는 프로젝트입니다.

## 빠른 시작 (학습 없음)

```bash
pip install -r requirements.txt
python3 run.py allocate --benchmark-dataset benchmark_dataset
```

결과: 콘솔 표 + `output/static_allocation_*.json`, `*.csv`

## 문서

- **[COMMANDS.md](COMMANDS.md)** — CLI·API·옵션 전체
- **[guide.md](guide.md)** — 시스템 구조·Input/Output·배경

## 실행 모드

| 모드 | 설명 |
|------|------|
| `allocate` | Input만으로 **정적** 최적 대수 (권장) |
| `train` / `infer` | 1시간 slot **강화학습** (선택) |

```bash
python3 run.py -h
```
