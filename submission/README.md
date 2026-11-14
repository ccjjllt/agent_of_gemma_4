# Submission Directory

Put the actual Kaggle submission root here. Expected layout follows the official `adk-submission` contract:

```text
submission/
├── agent.yaml                 # exactly one root config
├── eval_config.yaml           # optional evaluation budget
├── sub_agents/                # optional nested agent YAML
├── skills/                    # optional skill files
└── adapters/                  # optional PEFT LoRA adapters (rank <= 128)
```

Do not add Python entrypoints. All agents, tools, callbacks and model bindings must resolve through the official host registries. Keep the unpacked tree below 3 GiB including `adapters/`.

This directory is intentionally a skeleton on `main`; model training and adapter research live on `codex/training-plan-refactor`.
