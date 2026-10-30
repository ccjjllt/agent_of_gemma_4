"""Durable collection/update transitions for one explicitly budgeted run."""
from __future__ import annotations

import copy
import time

from .recovery import CollectionInterrupted, file_digest


class TrainingSession:
    def __init__(self, *, cfg, sampler, policy, endpoint, ledger, store, run_dir,
                 collect, collect_kwargs, emit, capture_rng, restore_rng, expired,
                 deadline, reward_summary, torch):
        self.cfg, self.sampler, self.policy = cfg, sampler, policy
        self.endpoint, self.ledger, self.store = endpoint, ledger, store
        self.run_dir, self.collect, self.collect_kwargs = run_dir, collect, collect_kwargs
        self.emit, self.capture_rng, self.restore_rng = emit, capture_rng, restore_rng
        self.expired, self.deadline, self.reward_summary, self.torch = expired, deadline, reward_summary, torch
        self.state = None
        self.checkpoint_id = None
        self.trainer = None

    def check_budget(self):
        if self.expired.is_set() or time.monotonic() >= self.deadline:
            raise CollectionInterrupted("Run budget exhausted; resume its committed checkpoint")

    def save(self, state, training=None):
        export_started = time.perf_counter()
        training = self.trainer.export_training_state() if training is None else training
        export_seconds = time.perf_counter() - export_started
        commit_started = time.perf_counter()
        checkpoint_id = self.store.save("checkpoint", {"run": state, "trainer": training})
        commit_seconds = time.perf_counter() - commit_started
        self.state, self.checkpoint_id = state, checkpoint_id
        self.emit("checkpoint_committed", checkpoint_id=checkpoint_id, phase=state["phase"],
                  batch=state["batch"], optimizer_steps=training["completed_optimizer_steps"],
                  state_export_seconds=export_seconds, durable_commit_seconds=commit_seconds,
                  checkpoint_bytes=(self.store.root / checkpoint_id / "state.pt").stat().st_size,
                  export_timing_scope="session export only; trainer callback snapshots are prepared before this call")

    def checkpoint_training(self, training):
        """Called only by the trainer at a fully recoverable optimizer boundary."""
        state = copy.deepcopy(self.state)
        active = training["active_batch"]
        if active is None or active["offset"] != state["policy_version"]:
            raise ValueError("Optimizer checkpoint does not belong to the collected policy batch")
        state["phase"] = "updating"
        state["ledger_updates"][str(state["batch"])] = {
            "optimizer_steps": active["completed_reuses"], "adapter_changed": None,
        }
        self.save(state, training)

    def run(self, trainer, *, resume):
        self.trainer = trainer
        saved = self.store.load("checkpoint") if resume else None
        if saved is not None:
            self.checkpoint_id, payload = saved
            self.state = payload["run"]
            if (self.state.get("schema_version") != 1
                    or self.state.get("phase") not in ("collecting", "update_pending", "updating", "finished")):
                raise ValueError("Unsupported run recovery state")
            trainer.restore_training_state(payload["trainer"])
            self.sampler.load_state_dict(self.state["sampler"])
            active = trainer.has_pending_batch
            if active != (self.state["phase"] == "updating"):
                raise ValueError("Run phase and pending optimizer cohort disagree")
            expected_steps = self.state["policy_version"]
            if active:
                expected_steps += self.state["ledger_updates"][str(self.state["batch"])]["optimizer_steps"]
            if trainer.completed_optimizer_steps != expected_steps:
                raise ValueError("Run policy and optimizer checkpoint counters disagree")
            self.ledger.restore_updates(self.state["ledger_updates"], self.checkpoint_id)
            self.emit("run_resumed", checkpoint_id=self.checkpoint_id,
                      phase=self.state["phase"], batch=self.state["batch"],
                      optimizer_steps=trainer.completed_optimizer_steps)
        else:
            if self.ledger.groups or self.ledger.batch_updates:
                raise ValueError("Existing reward evidence has no committed model baseline")
            initial = self.run_dir / "adapters" / "initial"
            self.policy.save_adapter(initial)
            self.save({"schema_version": 1, "phase": "collecting", "batch": 0,
                       "policy_version": 0, "sampler": self.sampler.state_dict(),
                       "ledger_updates": {},
                       "previous_adapter_hash": file_digest(initial / "adapter_model.safetensors")})

        while self.state["phase"] != "finished":
            self.check_budget()
            batch = self.state["batch"]
            if self.state["phase"] == "collecting":
                small = self.store.load("collection")
                collection_state = None
                if small is not None and small[1]["checkpoint_id"] == self.checkpoint_id:
                    collection_state = small[1]["state"]
                    self.restore_rng(small[1]["rng"])
                    self.emit("collection_resumed", batch=batch, collection_checkpoint_id=small[0],
                              completed_groups=collection_state["completed_groups"],
                              completed_slots=sum(slot["status"] == "complete" for group in collection_state["groups"]
                                                  for slot in group["slots"]))

                baseline_id = self.checkpoint_id

                def persist_collection(snapshot):
                    rng = self.endpoint.generation_barrier(self.capture_rng)
                    self.store.save("collection", {"checkpoint_id": baseline_id, "state": snapshot, "rng": rng})

                collection = self.collect(
                    cfg=self.cfg, batch=batch, policy_version=self.state["policy_version"],
                    sampler=self.sampler, endpoint=self.endpoint, run_dir=self.run_dir,
                    reward_ledger=self.ledger, recovery_state=collection_state,
                    persist_callback=persist_collection, **self.collect_kwargs)
                state = copy.deepcopy(self.state)
                state.update(phase="update_pending", sampler=self.sampler.state_dict(),
                             ledger_updates=copy.deepcopy(self.ledger.batch_updates),
                             trajectories=[item for group in collection["accepted"] for item in group],
                             collection={key: collection[key] for key in (
                                 "completed_groups", "failed_groups", "candidate_successes",
                                 "candidate_trajectories", "failure_limit_reached")})
                state["collection"].update(accepted_groups=len(collection["accepted"]),
                                           attempted_tasks=sorted(collection["attempted"]))
                # This baseline makes a failure before probability freezing
                # recoverable without re-collecting any accepted trajectory.
                self.save(state)

            trajectories, summary = self.state["trajectories"], self.state["collection"]
            previous_hash = self.state["previous_adapter_hash"]
            if trajectories:
                self.emit("grpo_update_started", batch=batch, policy_version=self.state["policy_version"],
                          accepted_groups=summary["accepted_groups"], trajectories=len(trajectories),
                          candidate_groups=summary["completed_groups"],
                          resume_pending=trainer.has_pending_batch,
                          reuse_iterations=int(self.cfg["grpo"]["num_iterations"]))
                self.check_budget()
                stats = trainer.resume_batch() if trainer.has_pending_batch else trainer.optimize_batch(trajectories)
                # Use the absolute count. A restored cohort's optimizer_steps
                # includes updates that were already committed before the crash.
                policy_version = int(stats["total_optimizer_steps"])
                updates = copy.deepcopy(self.state["ledger_updates"])
                updates[str(batch)] = {"optimizer_steps": int(stats["optimizer_steps"]), "adapter_changed": None}
                self.ledger.restore_updates(updates, self.checkpoint_id)
                self.check_budget()
                destination = self.run_dir / "adapters" / f"batch-{batch + 1:04d}"
                self.policy.save_adapter(destination)
                current_hash = file_digest(destination / "adapter_model.safetensors")
                self.ledger.mark_batch_update(batch, int(stats["optimizer_steps"]),
                                              adapter_changed=current_hash != previous_hash)
                self.emit("batch_finished", batch=batch, policy_version=policy_version,
                          accepted_groups=summary["accepted_groups"], candidate_groups=summary["completed_groups"],
                          sampled_official_success_rate=summary["candidate_successes"] / summary["candidate_trajectories"],
                          selected_official_success_rate=sum(item["official_success"] for item in trajectories) / len(trajectories),
                          **self.reward_summary(trajectories), adapter_changed=current_hash != previous_hash,
                          adapter_sha256=current_hash, adapter_path=str(destination), grpo=stats,
                          peak_memory_gib=[self.torch.cuda.max_memory_allocated(i) / 2**30
                                           for i in range(self.torch.cuda.device_count())])
            else:
                self.ledger.mark_batch_update(batch, 0, adapter_changed=False)
                policy_version, current_hash = self.state["policy_version"], previous_hash
                self.emit("batch_no_learning_signal", batch=batch, **summary)
            self.sampler.set_policy_version(policy_version)
            finished = batch + 1 >= int(self.cfg["run"]["max_batches"])
            self.save({"schema_version": 1, "phase": "finished" if finished else "collecting",
                       "batch": batch + 1, "policy_version": policy_version,
                       "sampler": self.sampler.state_dict(), "ledger_updates": copy.deepcopy(self.ledger.batch_updates),
                       "previous_adapter_hash": current_hash})
            if summary["failure_limit_reached"] and not finished:
                self.emit("run_stopped_after_worker_failures", next_batch=batch + 1,
                          checkpoint_id=self.checkpoint_id, resume_required=True)
                return 75
        self.emit("run_finished", optimizer_steps=trainer.completed_optimizer_steps,
                  checkpoint_id=self.checkpoint_id)
        return 0
