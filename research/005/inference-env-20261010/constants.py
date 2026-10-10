"""Shared pins for the P005 CPU inference-env kernel. Keep in sync with Global Constraints."""

STARTER_DOCKER_IMAGE = (
    "gcr.io/kaggle-private-byod/python@sha256:"
    "37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
)
KERNEL_ID = "ccjjllt/p005-infer-env-cpu"
KERNEL_TITLE = "P005 Infer Env CPU"
WHEELHOUSE_DATASET = "metric/gemma-4-developer-agent-wheelhouse"
COMPETITION = "gemma-4-developer-agent"
CODE_FILE = "probe.py"
