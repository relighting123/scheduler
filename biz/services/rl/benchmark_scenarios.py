"""벤치마크 시나리오 ID 규칙 및 레지스트리.

명명 규칙: bench_{순번2자리} (slug 없음)
  - bench_01 : 다품종·다모델 표준
  - bench_02 : 초기 장비 전환 필수
"""
from typing import Dict, List, Union

DEFAULT_BENCHMARK_SCENARIO = "bench_01"

# 구 ID → 신 ID (하위 호환)
LEGACY_SCENARIO_ALIASES: Dict[str, str] = {
    "benchmark_dataset": "bench_01",
    "benchmark_initial_conv": "bench_02",
    "bench_01_multiproduct": "bench_01",
    "bench_02_initial_conv": "bench_02",
}

SCENARIO_REGISTRY: Dict[str, str] = {
    "bench_01": "다품종·다모델 표준 벤치마크",
    "bench_02": "초기 장비 전환(재배치) 필수 벤치마크",
}


def resolve_scenario_id(scenario: str) -> str:
    """시나리오 ID 정규화 (레거시·slug 포함 구 ID 지원)."""
    return LEGACY_SCENARIO_ALIASES.get(scenario, scenario)


def parse_benchmark_scenarios(
    spec: Union[str, List[str], None] = None,
) -> List[str]:
    """평가할 시나리오 ID 목록.

    - None / 빈 문자열 → 기본 시나리오 1개
    - ``all`` / ``*`` → 레지스트리 전체
    - 쉼표 구분 문자열 또는 문자열 리스트
    """
    if spec is None:
        return [DEFAULT_BENCHMARK_SCENARIO]
    if isinstance(spec, list):
        raw = [str(s).strip() for s in spec if str(s).strip()]
    else:
        text = str(spec).strip()
        if not text:
            return [DEFAULT_BENCHMARK_SCENARIO]
        if text.lower() in ("all", "*"):
            return list(SCENARIO_REGISTRY.keys())
        raw = [part.strip() for part in text.split(",") if part.strip()]

    resolved = [resolve_scenario_id(item) for item in raw]
    unknown = [s for s in resolved if s not in SCENARIO_REGISTRY]
    if unknown:
        known = ", ".join(SCENARIO_REGISTRY.keys())
        raise ValueError(
            f"알 수 없는 벤치마크 시나리오: {unknown}. 사용 가능: {known} (또는 all)"
        )
    return resolved
