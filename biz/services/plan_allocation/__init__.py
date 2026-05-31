"""Input-only static plan capacity and equipment allocation optimizer."""

from biz.services.plan_allocation.optimizer import PlanAllocationOptimizer
from biz.services.plan_allocation.report import format_allocation_report

__all__ = ["PlanAllocationOptimizer", "format_allocation_report"]
