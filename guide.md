해당 프로젝트는 장비 전환 스케줄링 시스템이다.

[1] 인프라 구성
FastAPI 기반으로 구성하며 통신을 위해 Queue 관리를 한다.
Queue에 쌓인 요청에 대해 FIFO에 하나씩 꺼내어 프로세스에 할당한다.
해당 프로세스 갯수는 config 방식으로 구성하여 mutli Process로 동작한다.
기본적으로 위와 같은 통신 구조는 별도 core 항목으로 관리하며 개발자는 여기에 관심을 두지 않게 독립적인 구성이 되어야 한다.


[2] 프로세스 동작 구조
해당 프로세스가 생성시 DB 연결을 진행하여 커넥션을 유지한다. 
DB는 Oracle DB이고 접속 정보는 config 파일로 별도 관리한다.
이러한 DB 연결 방식등 역시 별도 Core 형태로 관리하며 사용자는 사용법만 알면 가져다 쓰게 한다.
DB 연결은 여러 DB에 연결될 수 있기 떄문에 이러한 것도 고려해야 한다.
해당 DB에는 조회/저장/업데이트/삭제/MERGE INTO 기능이 있고 실제 사용자는 간편한 방식으로 이러한 기능을 활요할 수 있다.

서비스 이름 : XEPDB1
IP와 포트 : localhost:1521
계정/암호 : dispatcher/dispatcher


[3] 비즈니스 로직 확장 가이드
본 프로젝트는 인프라(Core)와 비즈니스 서비스(Biz)가 철저히 분리되어 있습니다. 새로운 작업(Task)이나 추론/학습 기능을 추가할 때는 다음 단계를 따릅니다.

1. **서비스 생성**: `biz/services/` 폴더에 새로운 `{name}_service.py` 파일을 생성하고 실행할 로직을 함수로 정의합니다.
2. **로직 구현**: 서비스 함수 내에서 `BaseRepository`를 사용하여 DB 작업을 수행하거나 외부 엔진을 호출합니다.
3. **서비스 등록**: `biz/task_processor.py`에서 새로 만든 서비스를 임포트하고, `process` 메서드의 분기문에 `action`과 매핑하여 등록합니다.
4. **API 호출**: FastAPI `/tasks` 엔드포인트에 정의한 `action` 명을 포함하여 요청을 보냅니다.

상세 예제 코드는 `examples/add_task_example.py`에 작성되어 있습니다.

[4] 시스템 세부 정보
해당 비즈니스는 강화학습과 모방학습을 결합한다.
학습 모듈과 추론 모듈이 있다.
강화학습은 Stablebaseline을 활용한다.
강화학습은 모방학습으로 초기 정책을 학습하고 이후 PPO 모델을 통해 학습한다.

시물레이터는 1시간 단위로 구동되며 slot이라고 의미한다.

input 데이터의 형태는 하기와 같다.
학습(Input) 테이블 7종 모두 **RULE_TIMEKEY** 컬럼을 포함한다 (형식: `YYYYMMDDHHMMSS`, 예: `20251020070000`).
동일 테이블에 서로 다른 RULE_TIMEKEY 스냅샷이 여러 기간·시점으로 공존할 수 있으며, 학습·추론 시 해당 키로 조회한다.

1. 제품별 공정 수순 정보 및 재공 정보
 RULE_TIMEKEY | PLAN_PROD_KEY  | OPER_ID | OPER_SEQ | WIP_QTY

2. 제품별 공정별 장비 모델별 시간당 생산량
 RULE_TIMEKEY | PLAN_PROD_KEY  | OPER_ID | EQP_MODEL_CD | UPH 

3. 제품별 공정별 장비 모델별 시간대별 댓수
 RULE_TIMEKEY | BATCH_ID | EQP_MODEL_CD | TIME_SLOT | EQP_QTY

5. 제품별 공정별 장비 모델별 처리가능여부
 RULE_TIMEKEY | PLAN_PROD_KEY  | OPER_ID | EQP_MODEL_CD | AVAIL_YN

6. Tool 교체 단위 정보
 RULE_TIMEKEY | BATCH_ID | PLAN_PROD_KEY | OPER_ID
Batch id는 plan prod key와 oper id에 의해 정의된다. pla prod key||oper_I와 batch id는 N:1 관계이다.

7. Tool 갯수 정보
 RULE_TIMEKEY | BATCH_ID | EQP_MODEL_CD | TOOL_QTY

8. 계획 정보
 RULE_TIMEKEY | PLAN_PROD_KEY  | OPER_ID | START TIME | END TIME | PLAN_QTY
계획 제품 Key / Oper 별 계획이 있고 세부 일자 시간대별 계획이 있다. 가령 P1 / PT1H / 2026051707 | 2026051708 | 100 이면 2026년 5월 17일 07시부터 08시까지 10000개를 생산하라는 계획이야. 그리고 2026051708 | 2026051807 | 300 이면 2026년 5월 17일 08시부터 18시까지 300개를 생산하라는 계획이야.
이런식으로 동일 제품에 대해 여러 계획이 있을 수 있따.


처리로직은
위와 같은 데이터를 기반으로 장비 배치를 변경하여 시간대별 계획 달성을 Plan prod key|oper 별로 고르게 수행하는 것과 장비 Idle을 최소화하는 목표로 운영한다. 또한 공정별 move 계획 달성 외에도 최종 plan prod key | last oper 기준에서의 공정 달성율도 고려해야 한다. 이를 강화학습으로 운영하도록 할거야

output

RULE_TIMEKEY | FROM_BATCH | FROM_PLAN_PROD_KEY | FROM_OPER_ID | EQP_MODEL_CD | TO_BATCH_ID | TO_PLAN_PROD_KEY | TO_OPER_ID |  EQP_MODEL_CD | START_CONV_TIME | EQP_QTY
형태로 데이털르 생성한다.
2026051723020000 형태가 Rule TImekey이고 현재 수행시간을 의미한다. 테이블 명은 RTD_CONV_INF / RTD_CONV_HI 이고 삭제후 insert하게 된다.

위 input/output 데이터는 위 내용 기반으로 테이블 설계해서 지정된 오라클 db에 생성 및 테스트 데이터 생성해서 만들어줘

[5] 시물레이터
PLAN PROD KEY와 OPER ID별로 배치된 장비들의 UPH를 통해 UPH만큼 재공을 소진한다. 또한 그 재공은 PLAN PROD KEY의 OPER SEQ를 통해 다음 공정의 재공으로 유입된다. 마지막 OPER SEQ를 완료하면 최종 생산량이 되면 해당 OPER SEQ를 수행하면 OPER의 계획 달성에 기여한다. 이러한 흐름을 로그로 테이블화하여 모니터링에 기여해얗한다.

[6] 배경지식
 batch id가 달라지는 경우는 tool 교체가 일어나며 to batch id의 tool은 소진하고 from batch id의 tool은 반환한다. 또한 1시간동안은 비가용 상태가 되고 1시간 후 to batch id로 가용상태가 된다.
  동일 batch id이나 plan prod key인 경우는 tool 교체를 하지 않아도 되며 시간소요도 없다

[7] 학습·추론 RULE_TIMEKEY 운영

**학습**
- `from_rule_timekey` ~ `to_rule_timekey` 구간의 스냅샷을 DB에서 조회하여 학습 (구간에 여러 키가 있으면 에피소드마다 무작위 스냅샷).
- 단일 스냅샷만 지정할 때는 `rule_timekey` 또는 `from`/`to`에 동일 값 지정.
- 학습 완료 후 `test/data/<시나리오>/` 테스트 데이터로 Optimal·휴리스틱·RL 성능 비교 (`evaluate_on_test_data`).
- 테스트 데이터: 입력 7종 CSV + `ground_truth.json`(정답 지표).

**추론**
- Input: `rule_timekey` 지정 시 해당 스냅샷, 미지정·N/A 시 `MAX(RULE_TIMEKEY)` (WIP_INFO).
- Output(RTD_CONV): `output_rule_timekey` 지정, 미지정 시 현재 시각 `YYYYMMDDHHMMSS`.

**CLI 예시**
```bash
python run.py train --from-timekey 20251020070000 --to-timekey 20251020120000 --steps 50000
python run.py infer --timekey 20251020070000
python run.py infer   # Input=DB MAX, Output=현재 시각
```

**API parameters (rl_train)**
- `from_rule_timekey`, `to_rule_timekey`, `rule_timekey`, `run_test_eval`, `test_scenario`

**API parameters (rl_inference)**
- `rule_timekey` (task 또는 parameters), `output_rule_timekey`

[8] 고민사항
이제 실제로 강화학습 구현해야 하는 데 고민사항이 있어.  장비는 장비모델이 있고 그 속에 장비ID 들이 있어.   그리고 재공이 있는데 각 재공은 PROD ID를 가지고 있어 또한 계획이 있고 이는 계획제품이라는 여러 PROD ID를 묶은 GROUPING된 조건이 있어. 하나의 PROD ID는 하나의 계획제품에만 들어갈 수 있고 하나의 계획제품 안에 여러 PROD ID가 있는 상태이지. 이 계획 제품 단위로 계획량을 관리하고 있어. 그리고 각 PROD ID 기준으로 공정 수순인 FLOW를 가지고 있어. FLOW는 OPER와 SEQ로 구성된 개념이야
PLAN PROD KEY / OPER /EQP MODEL 별로 시간당 처리량이 있는데 이건 분산이 존재해. 또한 해당 기준으로 처리가능할지 여부가 있으나 이것 역시 실제로는 백프로 보장하지 않아.
실제로는 EQP_ID/LOT_ID 기준으로 해야 완전 정확한 거지.
여기서 고민 지점이 있어 EQP_ID/LOT_ID로 하면 너무데이터가 많아져서 문제가 되는 거야. 또 PLAN PROD KEY/OPER/EQP MODEL별로 그루핑하자니 세부 레벨 단위 놓치는 게 있을 거 같은게 첫번째 고민이야

PLAN PROD KEY는 여러개이고 EQP_MODEL도 여러개야

두번째는 PLAN PROD KEY와 MODEL은 양이 많아지기 떄문에 강화학습에서 상태 문제가 생길 거 같아
내가 하려는 건 PAL PROD KEY에 MODEL별로 몇대가 적당하기 댸문에 1대 추가하고 다른 PLAN PROD KEY/OPER에서 MODEL 중 1대 뺴라 이렇게 해서 가동율 최대화하면서 계획달성을 유지하려는 강화학습 모델이야.
이러한 Stete action 차원문제를 해결할 방법(GNN,Attention?,혹은 정말 단순한 방법(클러스터링?) 이 있을까?