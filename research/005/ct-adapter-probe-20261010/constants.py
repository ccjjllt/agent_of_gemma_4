"""Pins for the P005 official-CT diagnostic adapter probe."""

STARTER_DOCKER_IMAGE = (
    "gcr.io/kaggle-private-byod/python@sha256:"
    "37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
)
FORBIDDEN_DOCKER_IMAGE = (
    "gcr.io/kaggle-images/python@sha256:"
    "81b1e7c4b4f0f2a7e0a33943c95754b1b7af6b30d65221633a284b6b601cd84a"
)
KERNEL_ID = "ccjjllt/p005-ct-adapter-probe"
KERNEL_TITLE = "P005 CT Adapter Probe"
WHEELHOUSE_DATASET = "metric/gemma-4-developer-agent-wheelhouse"
COMPETITION = "gemma-4-developer-agent"
CT_MODEL = "google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2"
DDP_PROBE_KERNEL = "ccjjllt/p005-training-ddp-probe"
CODE_FILE = "probe.py"
ADAPTER_NAME = "diag_ddp_step4"
EXPECTED_ADAPTER_CONFIG_SHA256 = "6e25de4c3846d89643f85b5607fffe9a436eb3668bdf148d806ce5ee16298b6a"
EXPECTED_ADAPTER_WEIGHTS_SHA256 = "a870230ea9ac96abf90e998203c8b9c423e494cccb7f0faffda33b319fe78cf0"
EXPECTED_ADAPTER_WEIGHTS_BYTES = 212261592
T0_SHA256 = {
    "agent.yaml": "ae12f5f5911c855970b075a72fa2571726efe86f90130b24bd139ce47906e6d0",
    "configs/sampling.yaml": "5a6acb429b95163d65f68ff483ebe83031b025eb818840b39f6d5b718f09645e",
    "eval_config.yaml": "2a0205536c9d418d8502be4b13b3c063e1c0800d591e38c72d651a092de2b147",
    "prompts/system.md": "6c2e8f4c90ec07fa973be9cefa9c04176f99acdef3ab29a985302e04203a17b5",
}
CLASSES = (
    "adapter_not_loaded",
    "inference_error",
    "tool_call_ok_no_repair_claim",
)
ROUND_GPU_USED_AT_START = 7011
ROUND_GPU_CAP_DELTA_S = 21600
WALL_TIMEOUT_S = 1500
L4_BILLING_MULTIPLIER = 2
