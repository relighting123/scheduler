"""벤치마크 시나리오 ID 규칙 및 레지스트리.

명명 규칙: bench_{순번2자리}_{slug}
  - bench_01_multiproduct : 다품종·다모델 표준 (구 benchmark_dataset)
  - bench_02_initial_conv : 초기 장비 전환 필수 (구 benchmark_initial_conv)
"""
from typing import Dict

DEFAULT_BENCHMARK_SCENARIO = "bench_01_multiproduct"

# 구 ID → 신 ID (하위 호환)
LEGACY_SCENARIO_ALIASES: Dict[str, str] = {
    "benchmark_dataset": "bench_01_multiproduct",
    "benchmark_initial_conv": "bench_02_initial_conv",
}

SCENARIO_REGISTRY: Dict[str, str] = {
    "bench_01_multiproduct": "다품종·다모델 표준 벤치마크",
    "bench_02_initial_conv": "초기 장비 전환(재배치) 필수 벤치마크",
}


def resolve_scenario_id(scenario: str) -> str:
    """시나리오 ID 정규화 (레거시 별칭 지원)."""
    return LEGACY_SCENARIO_ALIASES.get(scenario, scenario)
