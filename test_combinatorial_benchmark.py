import pandas as pd
from biz.services.rl_scheduler_service import RLSchedulerService

class TestCombinatorialBenchmarkService(RLSchedulerService):
    def init_combinatorial_scenario(self):
        # DB 연결이 없으므로 PASS (fetch_data에서 더미 데이터 반환)
        print("\n[DB 설정] 모의(Mock) 조합최적화 벤치마크 시나리오 초기화 (DB 연결 없음)")

    def fetch_data(self):
        # 1. 제품별 공정 수순 및 재공 정보 (WIP_INFO)
        wip_data = pd.DataFrame([
            {'PLAN_PROD_KEY': 'P1', 'OPER_ID': 'OP10', 'OPER_SEQ': 10, 'WIP_QTY': 15000},
            {'PLAN_PROD_KEY': 'P1', 'OPER_ID': 'OP20', 'OPER_SEQ': 20, 'WIP_QTY': 2000},
            {'PLAN_PROD_KEY': 'P2', 'OPER_ID': 'OP10', 'OPER_SEQ': 10, 'WIP_QTY': 10000},
            {'PLAN_PROD_KEY': 'P2', 'OPER_ID': 'OP20', 'OPER_SEQ': 20, 'WIP_QTY': 1000},
            {'PLAN_PROD_KEY': 'P3', 'OPER_ID': 'OP10', 'OPER_SEQ': 10, 'WIP_QTY': 6000},
            {'PLAN_PROD_KEY': 'P3', 'OPER_ID': 'OP20', 'OPER_SEQ': 20, 'WIP_QTY': 500}
        ])
        
        # 2. 장비 모델별 시간당 생산량 (UPH_INFO) - UPH 분산 및 전용모델 특성 반영
        uph_data = pd.DataFrame([
            {'PLAN_PROD_KEY': 'P1', 'OPER_ID': 'OP10', 'EQP_MODEL_CD': 'MODEL_A', 'UPH': 100},
            {'PLAN_PROD_KEY': 'P1', 'OPER_ID': 'OP20', 'EQP_MODEL_CD': 'MODEL_A', 'UPH': 100},
            {'PLAN_PROD_KEY': 'P1', 'OPER_ID': 'OP10', 'EQP_MODEL_CD': 'MODEL_B', 'UPH': 70},
            {'PLAN_PROD_KEY': 'P1', 'OPER_ID': 'OP20', 'EQP_MODEL_CD': 'MODEL_B', 'UPH': 70},
            {'PLAN_PROD_KEY': 'P2', 'OPER_ID': 'OP10', 'EQP_MODEL_CD': 'MODEL_A', 'UPH': 80},
            {'PLAN_PROD_KEY': 'P2', 'OPER_ID': 'OP20', 'EQP_MODEL_CD': 'MODEL_A', 'UPH': 80},
            {'PLAN_PROD_KEY': 'P2', 'OPER_ID': 'OP10', 'EQP_MODEL_CD': 'MODEL_B', 'UPH': 120},
            {'PLAN_PROD_KEY': 'P2', 'OPER_ID': 'OP20', 'EQP_MODEL_CD': 'MODEL_B', 'UPH': 120},
            {'PLAN_PROD_KEY': 'P3', 'OPER_ID': 'OP10', 'EQP_MODEL_CD': 'MODEL_B', 'UPH': 60},
            {'PLAN_PROD_KEY': 'P3', 'OPER_ID': 'OP20', 'EQP_MODEL_CD': 'MODEL_B', 'UPH': 60},
            {'PLAN_PROD_KEY': 'P3', 'OPER_ID': 'OP10', 'EQP_MODEL_CD': 'MODEL_C', 'UPH': 100},
            {'PLAN_PROD_KEY': 'P3', 'OPER_ID': 'OP20', 'EQP_MODEL_CD': 'MODEL_C', 'UPH': 100}
        ])
        
        # 3. 장비 댓수 정보 (EQP_QTY_INFO) - 초기 비효율적 할당 (병목 유발)
        eqp_qty_data = pd.DataFrame([
            {'BATCH_ID': 'B1', 'EQP_MODEL_CD': 'MODEL_A', 'TIME_SLOT': '2026051800', 'EQP_QTY': 5},
            {'BATCH_ID': 'B3', 'EQP_MODEL_CD': 'MODEL_A', 'TIME_SLOT': '2026051800', 'EQP_QTY': 5},
            {'BATCH_ID': 'B2', 'EQP_MODEL_CD': 'MODEL_B', 'TIME_SLOT': '2026051800', 'EQP_QTY': 4},
            {'BATCH_ID': 'B5', 'EQP_MODEL_CD': 'MODEL_B', 'TIME_SLOT': '2026051800', 'EQP_QTY': 4},
            {'BATCH_ID': 'B6', 'EQP_MODEL_CD': 'MODEL_C', 'TIME_SLOT': '2026051800', 'EQP_QTY': 5}
        ])
        
        # 4. 처리가능여부 (AVAIL_INFO)
        avail_data = pd.DataFrame([
            {'PLAN_PROD_KEY': 'P1', 'OPER_ID': 'OP10', 'EQP_MODEL_CD': 'MODEL_A', 'AVAIL_YN': 'Y'},
            {'PLAN_PROD_KEY': 'P1', 'OPER_ID': 'OP20', 'EQP_MODEL_CD': 'MODEL_A', 'AVAIL_YN': 'Y'},
            {'PLAN_PROD_KEY': 'P1', 'OPER_ID': 'OP10', 'EQP_MODEL_CD': 'MODEL_B', 'AVAIL_YN': 'Y'},
            {'PLAN_PROD_KEY': 'P1', 'OPER_ID': 'OP20', 'EQP_MODEL_CD': 'MODEL_B', 'AVAIL_YN': 'Y'},
            {'PLAN_PROD_KEY': 'P1', 'OPER_ID': 'OP10', 'EQP_MODEL_CD': 'MODEL_C', 'AVAIL_YN': 'N'},
            {'PLAN_PROD_KEY': 'P1', 'OPER_ID': 'OP20', 'EQP_MODEL_CD': 'MODEL_C', 'AVAIL_YN': 'N'},
            {'PLAN_PROD_KEY': 'P2', 'OPER_ID': 'OP10', 'EQP_MODEL_CD': 'MODEL_A', 'AVAIL_YN': 'Y'},
            {'PLAN_PROD_KEY': 'P2', 'OPER_ID': 'OP20', 'EQP_MODEL_CD': 'MODEL_A', 'AVAIL_YN': 'Y'},
            {'PLAN_PROD_KEY': 'P2', 'OPER_ID': 'OP10', 'EQP_MODEL_CD': 'MODEL_B', 'AVAIL_YN': 'Y'},
            {'PLAN_PROD_KEY': 'P2', 'OPER_ID': 'OP20', 'EQP_MODEL_CD': 'MODEL_B', 'AVAIL_YN': 'Y'},
            {'PLAN_PROD_KEY': 'P2', 'OPER_ID': 'OP10', 'EQP_MODEL_CD': 'MODEL_C', 'AVAIL_YN': 'N'},
            {'PLAN_PROD_KEY': 'P2', 'OPER_ID': 'OP20', 'EQP_MODEL_CD': 'MODEL_C', 'AVAIL_YN': 'N'},
            {'PLAN_PROD_KEY': 'P3', 'OPER_ID': 'OP10', 'EQP_MODEL_CD': 'MODEL_A', 'AVAIL_YN': 'N'},
            {'PLAN_PROD_KEY': 'P3', 'OPER_ID': 'OP20', 'EQP_MODEL_CD': 'MODEL_A', 'AVAIL_YN': 'N'},
            {'PLAN_PROD_KEY': 'P3', 'OPER_ID': 'OP10', 'EQP_MODEL_CD': 'MODEL_B', 'AVAIL_YN': 'Y'},
            {'PLAN_PROD_KEY': 'P3', 'OPER_ID': 'OP20', 'EQP_MODEL_CD': 'MODEL_B', 'AVAIL_YN': 'Y'},
            {'PLAN_PROD_KEY': 'P3', 'OPER_ID': 'OP10', 'EQP_MODEL_CD': 'MODEL_C', 'AVAIL_YN': 'Y'},
            {'PLAN_PROD_KEY': 'P3', 'OPER_ID': 'OP20', 'EQP_MODEL_CD': 'MODEL_C', 'AVAIL_YN': 'Y'}
        ])
        
        # 5. Tool 교체 단위 정보 (BATCH_TOOL_INFO)
        batch_tool_data = pd.DataFrame([
            {'BATCH_ID': 'B1', 'PLAN_PROD_KEY': 'P1', 'OPER_ID': 'OP10'},
            {'BATCH_ID': 'B2', 'PLAN_PROD_KEY': 'P1', 'OPER_ID': 'OP20'},
            {'BATCH_ID': 'B3', 'PLAN_PROD_KEY': 'P2', 'OPER_ID': 'OP10'},
            {'BATCH_ID': 'B4', 'PLAN_PROD_KEY': 'P2', 'OPER_ID': 'OP20'},
            {'BATCH_ID': 'B5', 'PLAN_PROD_KEY': 'P3', 'OPER_ID': 'OP10'},
            {'BATCH_ID': 'B6', 'PLAN_PROD_KEY': 'P3', 'OPER_ID': 'OP20'}
        ])
        
        # 6. Tool 갯수 정보 (TOOL_QTY_INFO)
        tool_qty_data = pd.DataFrame([
            {'BATCH_ID': b, 'EQP_MODEL_CD': m, 'TOOL_QTY': 20}
            for b in ['B1', 'B2', 'B3', 'B4', 'B5', 'B6']
            for m in ['MODEL_A', 'MODEL_B', 'MODEL_C']
        ])
        
        # 7. 계획 정보 (PLAN_INFO) - 24시간 기준 목표
        plan_data = pd.DataFrame([
            {'PLAN_PROD_KEY': 'P1', 'OPER_ID': 'OP10', 'START_TIME': '2026051800', 'END_TIME': '2026051824', 'PLAN_QTY': 12000},
            {'PLAN_PROD_KEY': 'P1', 'OPER_ID': 'OP20', 'START_TIME': '2026051800', 'END_TIME': '2026051824', 'PLAN_QTY': 12000},
            {'PLAN_PROD_KEY': 'P2', 'OPER_ID': 'OP10', 'START_TIME': '2026051800', 'END_TIME': '2026051824', 'PLAN_QTY': 8000},
            {'PLAN_PROD_KEY': 'P2', 'OPER_ID': 'OP20', 'START_TIME': '2026051800', 'END_TIME': '2026051824', 'PLAN_QTY': 8000},
            {'PLAN_PROD_KEY': 'P3', 'OPER_ID': 'OP10', 'START_TIME': '2026051800', 'END_TIME': '2026051824', 'PLAN_QTY': 5000},
            {'PLAN_PROD_KEY': 'P3', 'OPER_ID': 'OP20', 'START_TIME': '2026051800', 'END_TIME': '2026051824', 'PLAN_QTY': 5000}
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
    print("[1단계] 조합최적화 벤치마크 테스트 서비스 초기화 중...")
    # DB 매니저 없이 더미 서비스 생성
    test_service = TestCombinatorialBenchmarkService(db_manager=None)
    
    print("\n[2단계] 조합최적화 벤치마크 실행 (정답지 vs 휴리스틱 vs RL)...")
    # 빠른 테스트를 위해 total_timesteps=1000 지정
    test_service.run_combinatorial_benchmark(total_timesteps=60000)
    print("\n[성공] 모든 벤치마크 테스트가 완료되었습니다.")
