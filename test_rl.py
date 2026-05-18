import pandas as pd
from biz.services.rl_scheduler_service import RLSchedulerService

class TestRLSchedulerService(RLSchedulerService):
    def fetch_data(self):
        # 1. 제품별 공정 수순 정보 및 재공 정보 (WIP_INFO)
        wip_data = pd.DataFrame([
            {'PLAN_PROD_KEY': 'P1', 'OPER_ID': 'OP10', 'OPER_SEQ': 10, 'WIP_QTY': 1000},
            {'PLAN_PROD_KEY': 'P2', 'OPER_ID': 'OP10', 'OPER_SEQ': 10, 'WIP_QTY': 500}
        ])
        # 2. 제품별 공정별 장비 모델별 시간당 생산량 (UPH_INFO)
        uph_data = pd.DataFrame([
            {'PLAN_PROD_KEY': 'P1', 'OPER_ID': 'OP10', 'EQP_MODEL_CD': 'MODEL_A', 'UPH': 100},
            {'PLAN_PROD_KEY': 'P1', 'OPER_ID': 'OP10', 'EQP_MODEL_CD': 'MODEL_B', 'UPH': 120},
            {'PLAN_PROD_KEY': 'P2', 'OPER_ID': 'OP10', 'EQP_MODEL_CD': 'MODEL_A', 'UPH': 80}
        ])
        # 3. 제품별 공정별 장비 모델별 시간대별 댓수 (EQP_QTY_INFO)
        eqp_qty_data = pd.DataFrame([
            {'BATCH_ID': 'B1', 'EQP_MODEL_CD': 'MODEL_A', 'TIME_SLOT': '2026051707', 'EQP_QTY': 5},
            {'BATCH_ID': 'B2', 'EQP_MODEL_CD': 'MODEL_A', 'TIME_SLOT': '2026051707', 'EQP_QTY': 3}
        ])
        # 5. 처리가능여부 (AVAIL_INFO)
        avail_data = pd.DataFrame([
            {'PLAN_PROD_KEY': 'P1', 'OPER_ID': 'OP10', 'EQP_MODEL_CD': 'MODEL_A', 'AVAIL_YN': 'Y'},
            {'PLAN_PROD_KEY': 'P1', 'OPER_ID': 'OP10', 'EQP_MODEL_CD': 'MODEL_B', 'AVAIL_YN': 'Y'},
            {'PLAN_PROD_KEY': 'P2', 'OPER_ID': 'OP10', 'EQP_MODEL_CD': 'MODEL_A', 'AVAIL_YN': 'Y'}
        ])
        # 6. Tool 교체 단위 정보 (BATCH_TOOL_INFO)
        batch_tool_data = pd.DataFrame([
            {'BATCH_ID': 'B1', 'PLAN_PROD_KEY': 'P1', 'OPER_ID': 'OP10'},
            {'BATCH_ID': 'B2', 'PLAN_PROD_KEY': 'P2', 'OPER_ID': 'OP10'}
        ])
        # 7. Tool 갯수 정보 (TOOL_QTY_INFO)
        tool_qty_data = pd.DataFrame([
            {'BATCH_ID': 'B1', 'EQP_MODEL_CD': 'MODEL_A', 'TOOL_QTY': 10},
            {'BATCH_ID': 'B2', 'EQP_MODEL_CD': 'MODEL_A', 'TOOL_QTY': 5}
        ])
        # 8. 계획 정보 (PLAN_INFO)
        plan_data = pd.DataFrame([
            {'PLAN_PROD_KEY': 'P1', 'OPER_ID': 'OP10', 'START_TIME': '2026051707', 'END_TIME': '2026051708', 'PLAN_QTY': 400},
            {'PLAN_PROD_KEY': 'P2', 'OPER_ID': 'OP10', 'START_TIME': '2026051707', 'END_TIME': '2026051708', 'PLAN_QTY': 200}
        ])
        
        return {
            'wip_info': wip_data,
            'uph_info': uph_data,
            'eqp_qty_info': eqp_qty_data,
            'avail_info': avail_data,
            'batch_tool_info': batch_tool_data,
            'tool_qty_info': tool_qty_data,
            'plan_info': plan_data
        }

if __name__ == "__main__":
    print("[1단계] 테스트 서비스 초기화 중...")
    # DB 매니저 없이 더미 서비스 생성
    test_service = TestRLSchedulerService(db_manager=None)
    
    print("\n[2단계] 모델 사전학습(모방학습) 및 강화학습 진행 중...")
    # 빠른 테스트를 위해 total_timesteps=1000 지정
    test_service.train_model(total_timesteps=1000, pretrain_bc=True)
    
    print("\n[3단계] 모델 추론 진행 중...")
    results = test_service.run_inference()
    
    print("\n[최종 결과]")
    if results:
        print(f"총 {len(results)}건의 장비 전환(Action)이 도출되었습니다.")
        for i, r in enumerate(results[:5]):
            print(f"[{i+1}] {r}")
        if len(results) > 5:
            print("... (이하 생략)")
    else:
        print("도출된 전환 액션이 없습니다.")
