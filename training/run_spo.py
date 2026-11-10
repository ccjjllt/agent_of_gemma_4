"""SPO launch validator (CPU-safe). A800 start uses the same config checks.

Usage (server):
  python -m training.run_spo --config configs/spo_a800_2x80.toml --run-dir /root/autodl-tmp/gemma4/runs/spo-next [--check-only]
"""
from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path


REQUIRED_SECTIONS = ("policy", "run", "inference", "spo", "optimizer", "rewards", "adapter")
FORBIDDEN_KEYS = ("num_generations", "groups_per_batch", "max_candidate_groups_per_batch",
                  "num_iterations", "loss_type", "scale_rewards", "advantage_std_floor")


def load_config(path: Path) -> dict:
    with open(path, "rb") as handle:
        cfg = tomllib.load(handle)
    if cfg.get("algorithm") != "single_stream_spo":
        raise ValueError("algorithm must be single_stream_spo")
    for section in REQUIRED_SECTIONS:
        if section not in cfg or not isinstance(cfg[section], dict):
            raise ValueError(f"missing config section: {section}")
    text = Path(path).read_text()
    for key in FORBIDDEN_KEYS:
        if key in text:
            raise ValueError(f"legacy GRPO key must not appear in SPO config: {key}")
    policy = cfg["policy"]
    if (policy.get("lora_rank"), policy.get("lora_alpha")) != (128, 256):
        raise ValueError("SPO requires lora_rank=128, lora_alpha=256")
    if float(policy.get("lora_dropout", 0.0)) != 0.0:
        raise ValueError("SPO requires lora_dropout=0")
    spo = cfg["spo"]
    if int(spo.get("target_action_tokens", 0)) <= 0 or int(spo.get("max_policy_lag", -1)) < 0:
        raise ValueError("invalid spo token budget or lag")
    if cfg["adapter"].get("allow_resident_pair") is not False:
        raise ValueError("allow_resident_pair must be false until VRAM is measured")
    if cfg["rewards"].get("objective_version") != "official_differential_spo_v1":
        raise ValueError("wrong rewards objective_version")
    return cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(Path(args.config))
    run_dir = Path(args.run_dir)
    if args.check_only:
        print(f"spo_config_ok sections={sorted(cfg.keys())}")
        return 0
    from .trajectory.ledger import TrajectoryLedger
    from .spo.value_tracker import ValueTracker
    from .spo.sampler import SPOSampler
    run_dir.mkdir(parents=True, exist_ok=True)
    ledger = TrajectoryLedger(run_dir / "spo_stream.db")
    tracker = ValueTracker(
        ledger, prior_value=float(cfg["spo"].get("value_prior", 0.5)),
        prior_success=float(cfg["spo"].get("success_prior", 0.5)),
        prior_strength=float(cfg["spo"].get("prior_strength", 1.0)),
        kl_half_life=float(cfg["spo"].get("kl_half_life", 0.05)),
    )
    sampler = SPOSampler(tracker, epsilon=float(cfg["spo"].get("sampling_epsilon", 0.05)))
    state = ledger.recover()
    print(f"spo_run_ready run_dir={run_dir} buffer={state['buffer']} "
          f"kl_cursor={state['kl_cursor']} sampler_epsilon={sampler.epsilon}")
    ledger.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
