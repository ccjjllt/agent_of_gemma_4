# Real 31B weight A/B carrier preparation

CPU implementation only. No runnable carrier or adapter results have been
fabricated. The builder requires `--adapter-kernel owner/slug` and
`--training-manifest /absolute/path/to/the/real/training_manifest.json` from a
completed future RFT run. It rejects the existing capacity diagnostic signature.
It freezes the real final config/weights SHA256 and complete training manifest
SHA256, and copies only that source's final `adapter/` output at runtime.

Outputs default to this directory's new `kernel/` and `carrier-manifest.json`;
existing outputs are refused. It does not invoke the baseline builder main or
write baseline/next carrier locations. Use explicit new output paths for another
frozen candidate. No remote launch is performed by the builder.

The twelve ordered `final-selection.json.dev_task_ids` are the complete panel.
No train attempts are scheduled. A uses the four frozen base Agent files. B has
identical files except `adapter: p015_rft_lora` added to agent.yaml. This is the
actual official SandboxedLlmAgentConfig field; resolve_model resolves the alias
through the discovered adapter manifest and `adapter:<name>` registry entry.
The unchanged current attempt_worker performs those registrations and the
ordinary official Agent compilation, tools, independent workspaces and host
upstream F2P/P2P judgments. Runtime wire capture verifies the actual requested
model and response.model for every attempt; a route-unverified result cannot
contribute a paired win or verified resolved count.

One official31B CTv2 TP4 service runs both arms with LoRA enabled, exactly one
future RFT adapter, max LoRA rank8, and max sequence concurrency4. The capacity
diagnostic is not remounted: its independent v1 evidence is already available.
The actual VllmServer command is recorded and checked for the one-adapter/rank/
TP/concurrency settings. There is no base-only fallback in a weight A/B: failure
to load the candidate is a failed experiment, not evidence about its quality.

A/B order alternates by issue. Summary pairs only contemporaneous same-issue
outcomes, recording wins, regressions, unchanged successes/failures and invalid
or incomplete pairs. Both arms' dev counts remain separate. Historical5/12 is
not the control and is never subtracted to claim a gain. No pass/fail numbers
exist until the real model, service and experiments have run.

Optional Stage C is off by default. Explicit `--fast-thinking-budget 1024`
adds a third four-file base Agent whose sole change is the actual official
`thinking_config.thinking_budget` scalar, from4096 to1024. A/B remain byte-for-
byte unchanged and retain4096. Official generation resolution forwards this
field as `extra_body.thinking_token_budget`; captured runtime requests audit
that actual value as well as the requested/response model. All other sampling,
max output, prompts, tools and Agent budgets remain unchanged.

Stage C cannot launch until all24 A/B attempts have ended with valid judge and
routing observations and no A/B worker is active. A valid failed repair remains
a valid observation. It then runs the same12 dev issues on the warm service, requesting the
base model without an adapter while the one RFT adapter remains mounted. C is
reported separately as mechanism exploration, with per-task times and actual
thinking budgets. It contributes no weight-pair counts and uses neither the
historical5/12 nor a staged A/B arm as a simultaneous causal control.

The explicit `--wall-seconds` option accepts3600 (default) or4800. The manifest
records `platform_seconds=wall_seconds+600`:4200 seconds by default, or5400
seconds for the extended candidate. At the current2x quota multiplier these
platform bounds represent at most about2.33 or3.0 quota hours. The experiment
deadline includes service startup; the separate600 seconds reserve covers
bootstrap and final artifacts. The root must apply the recorded platform bound
when launching; the builder does not start or extend a job.

The root may select4800 only after real SFT has completed and the actual balance
has been checked, while retaining at least1.5 quota hours for deployment and
staying within the session's14-hour plan. Otherwise the default3600 remains.
This budget choice does not change A/B, per-attempt timeouts, Stage C admission
or the stage barrier. C still uses the same warm service, with no separate C
job or automatic retry. Its admission requires enough remaining time for three
four-worker waves at the full Agent budget, both F2P and P2P timeouts, and
cleanup reserve. Otherwise summary records the actual reason for not executing
C. The offline builder never assumes a balance. The motivation is v1 evidence
of API waiting and long completions; thinking time was not directly measured.
