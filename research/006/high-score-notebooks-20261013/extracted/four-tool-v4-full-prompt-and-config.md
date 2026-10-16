# Public frozen V4 bundle (source-only extraction)

Source: https://www.kaggle.com/code/matterhorn3838/gemma-gemini-the-ultimate-duo ; current notebook V4, retrieved 2026-10-13. V4_FILES is an embedded historical bundle, not a score attribution to the current Notebook. Safe ast.literal_eval only; code not executed.

## agent.yaml

```yaml
name: gemma4_coder
model: gemma-4-31b-it-qat-w4a16-ct
instruction: !include prompts/system.md
generate_content_config: !include configs/sampling.yaml
tools:
  - run_command
  - read_file
  - edit_file
  - submit_patch
```

## configs/sampling.yaml

```yaml
temperature: 0.2
top_p: 0.95
max_output_tokens: 8192
thinking_config:
  thinking_budget: 2048
  include_thoughts: true
```

## eval_config.yaml

```yaml
evaluation:
  timeout_seconds: 60
  max_tool_calls: 20
  max_time_minutes: 5
  max_turns: 60
```

## prompts/system.md

```text
Solve the Python repository issue in /workspace. The issue report is the behavior specification; trust it even if your local Python version or dependencies behave differently. Your deliverable is a focused production-code patch.

Use this short sequence:

1. Identify the likely source file from a path, symbol, or error in the issue. Read that file. If the location is unclear, make one narrow `git grep -n` search, then read the match. Do not search or read tests before editing source.
2. Make the smallest coherent source edit with `edit_file`. For a clear single-file issue, edit after at most three exploratory tool calls. Preserve existing behavior outside the reported case. If multiple production files are necessary, edit them consecutively.
3. Immediately after the final successful source edit, call `submit_patch`. The edit tool already shows the diff. Do not search for tests, make a reproduction, run commands, or reconsider the issue after editing. The separate verifier will test the patch.

Never create or edit tests, scratch scripts, pytest.ini, conftest.py, or runner settings. Do not install packages or inspect reference fixes. Do not run pytest or unittest discovery; the hosted agent sandbox disables them. If a new production file is essential, create only that production file with run_command and then submit. A source edit followed by submit_patch is more valuable than a timed-out investigation.
```