"""Official empty-patch differential rewards; importing never starts compute."""

from .baseline import BASELINE_SCHEMA_VERSION, EmptyPatchBaselineStore, TaskKey
from .critic import ProcessCritic, build_process_evidence
from .differential import compute_differential, pair_node_reports
from .engine import REWARD_RECORD_FIELDS, RewardEngine, compute_reward_record, spo_reward_record
from .progress import (
    DEFAULT_EFFICIENCY_WEIGHT,
    DEFAULT_SCALES,
    OBJECTIVE_VERSION,
    composite_reward,
    efficiency_evidence,
)


__all__ = [
    "BASELINE_SCHEMA_VERSION",
    "DEFAULT_EFFICIENCY_WEIGHT",
    "DEFAULT_SCALES",
    "EmptyPatchBaselineStore",
    "OBJECTIVE_VERSION",
    "ProcessCritic",
    "REWARD_RECORD_FIELDS",
    "RewardEngine",
    "TaskKey",
    "build_process_evidence",
    "composite_reward",
    "compute_differential",
    "compute_reward_record",
    "efficiency_evidence",
    "pair_node_reports",
]
