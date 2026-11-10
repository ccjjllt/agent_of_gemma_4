"""Single-stream Policy Optimization core; torch remains lazy until loss use."""

from .value_tracker import ValueObservation, ValueRecord, ValueTracker
from .advantage import TokenStatistics, action_token_statistics, freeze_advantages
from .sampler import SPOSampler
from .loss import SPOLoss, masked_ppo_clip_loss, microbatch_weight, spo_ppo_loss
from .trainer import SPO_KL_ESTIMATOR, SPOTrainer, make_spo_trainer

__all__ = [
    "SPOLoss", "SPOSampler", "SPOTrainer", "SPO_KL_ESTIMATOR", "TokenStatistics",
    "ValueObservation", "ValueRecord", "ValueTracker", "action_token_statistics",
    "freeze_advantages", "make_spo_trainer", "masked_ppo_clip_loss",
    "microbatch_weight", "spo_ppo_loss",
]
