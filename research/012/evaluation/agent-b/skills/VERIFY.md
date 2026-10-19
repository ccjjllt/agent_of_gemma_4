# P012 skills VERIFY：候选绑定运行记录扩展（一次性手动验证）

日期：2026-10-19。验证对象：`research/012/agent/skills/working-state/scripts/working_state.py`
（P012 扩展版，`--candidate/--stage` 命名空间），以及 `source-edit/scripts/patch_store.py`
（原样复用）冒烟。未读 heldout/预留题，未动 `research/005` 原件，未提交 GPU 作业。

## 1. 复制核对（sources 见 research/005 原目录）

复制命令：

```sh
mkdir -p research/012/agent/skills
cp -R research/005/candidates/single_state/agent/skills/working-state research/012/agent/skills/working-state
cp -R research/005/candidates/single_state/agent/skills/source-edit research/012/agent/skills/source-edit
cp -R research/005/candidates/single_state/agent/skills/source-probe research/012/agent/skills/source-probe
```

复制时刻 `research/012` 与 `research/005` SHA 一致（除本轮唯一改动文件外，
以下 11 个文件复制前后 SHA 相同，复制后又与 sources 逐一比对一致）：

| 文件 | SHA256（完整值，复制前后与 source 一致） |
|---|---|
| `working-state/SKILL.md` | `4acc59bb23ce9f542f7d2fc9fe7a5b972c47686947befa5ed57887c86943cfb2` |
| `working-state/scripts/workspace_paths.py` | `5dc0f650555961fe6a237d3bb86ab15b931c6493d31b2b6b50b72295c0479b0a` |
| `source-edit/SKILL.md` | `0da4dacb770dba7f24611c6323a01888cfd6a64119298bfe6da58e4f1d9fb9be` |
| `source-edit/scripts/patch_store.py` | `ddc96d8c9dcb4940193b9f959ef7e2255bc4199d6f4ef552273b756f79e0cb36` |
| `source-edit/scripts/source_edit.py` | `466847b9c964465396fb8a55ceac09abcfb0aaed4371f9813b2569e4c8b99bd1` |
| `source-edit/scripts/span_store.py` | `5a0172743b67eab2b1ff80fc2d47d5be143bd3e785b70cbde6ca9cf52dd89edd` |
| `source-edit/scripts/workspace_paths.py` | `5dc0f650555961fe6a237d3bb86ab15b931c6493d31b2b6b50b72295c0479b0a` |
| `source-probe/SKILL.md` | `b827f33668219344e14f6513a70c78fec33163560e819951eac7a649f764b204` |
| `source-probe/scripts/source_probe.py` | `621aa91fcd6c1570b3f2f5bee805ebc8dfae633798a431a44ddcc3b412d4fa86` |
| `source-probe/scripts/workspace_paths.py` | `5dc0f650555961fe6a237d3bb86ab15b931c6493d31b2b6b50b72295c0479b0a` |
| `working-state/scripts/working_state.py`（复制时） | `4393d21201f53583756a4989ba71c52f799778f225259824b7e4ef2af2`（= source，原件未动） |

本轮唯一改动：`working-state/scripts/working_state.py`
`4393d21201f53583756a4989ba71c52f799778f225259824b7e4ef2af2` →
`b22e601b7dd1afa2e3b0edeecd299988f13243db2f699f3fd3ecc29ebefca24c`
（P012 `--candidate/--stage` 扩展，见 `REUSE.md` §2）。

## 2. 验证环境（一次性临时工作区）

```sh
export W=/tmp/p012_verify_ws && rm -rf $W && mkdir -p $W && cd $W \
  && git init -q . && git config user.email t@t.t && git config user.name t \
  && printf 'def f():\n    return 1\n' > a.py && git add a.py && git commit -qm init \
  && echo "ISSUE TEXT cand demo" > issue.txt && echo WS_READY
```

输出：`WS_READY`。说明：`--allow-non-workspace-root` 显式传入；
state 目录绑定该临时仓库 toplevel（`root_e848523ae3823dfc`）。

## 3. 候选命名空间写入（修复 A / 修复 B / 验收 A）

命令（`WS=research/012/agent/skills/working-state/scripts/working_state.py`）：

```sh
python3 $WS --root $W --allow-non-workspace-root record --candidate A --stage repair \
  --trigger tA --expected eA --preserve pA --fact "f1|span:x|fact A" \
  --hypothesis "h1|transformer|claim A|f1|" --chosen h1 --unknown uA \
  --next-kind inspect --next-target a.py --next-purpose discriminate-A
python3 $WS --root $W --allow-non-workspace-root record --candidate B --stage repair \
  --trigger tB --expected eB --preserve pB --fact "f9|span:y|fact B" \
  --hypothesis "h9|producer|claim B|f9|" --chosen h9 --unknown uB \
  --next-kind probe --next-target b.py --next-purpose discriminate-B
```

输出（原文）：

```json
{"status": "ok", "command": "record", "candidate": "A", "stage": "repair", "changed_fields": ["(first-record: all fields new)"], "next_action": {"kind": "inspect", "target": "a.py", "purpose": "discriminate-A"}, "hypotheses": ["h1"], "note": "stored only; no root cause was inferred by this tool"}
{"status": "ok", "command": "record", "candidate": "B", "stage": "repair", "changed_fields": ["(first-record: all fields new)"], "next_action": {"kind": "probe", "target": "b.py", "purpose": "discriminate-B"}, "hypotheses": ["h9"], "note": "stored only; no root cause was inferred by this tool"}
```

## 4. 读回确认互不覆盖

命令：

```sh
python3 $WS --root $W --allow-non-workspace-root show --candidate A --stage repair
python3 $WS --root $W --allow-non-workspace-root show --candidate B --stage repair
```

输出（原文，A 与 B 各自独立）：

```json
{"status": "ok", "command": "show", "empty": false, "candidate": "A", "stage": "repair", "state": {"schema_version": 1, "issue_contract": {"trigger": "tA", "expected": "eA", "preserve": "pA"}, "facts": [{"id": "f1", "source_ref": "span:x", "claim": "fact A"}], "hypotheses": [{"id": "h1", "layer": "transformer", "claim": "claim A", "support": ["f1"], "contradiction": []}], "chosen": "h1", "unknown": "uA", "next_action": {"kind": "inspect", "target": "a.py", "purpose": "discriminate-A"}, "current_patch_id": "none"}, "changed_fields": ["(first-record: all fields new)"], "next_action": {"kind": "inspect", "target": "a.py", "purpose": "discriminate-A"}}
{"status": "ok", "command": "show", "empty": false, "candidate": "B", "stage": "repair", "state": {"schema_version": 1, "issue_contract": {"trigger": "tB", "expected": "eB", "preserve": "pB"}, "facts": [{"id": "f9", "source_ref": "span:y", "claim": "fact B"}], "hypotheses": [{"id": "h9", "layer": "producer", "claim": "claim B", "support": ["f9"], "contradiction": []}], "chosen": "h9", "unknown": "uB", "next_action": {"kind": "probe", "target": "b.py", "purpose": "discriminate-B"}, "current_patch_id": "none"}, "changed_fields": ["(first-record: all fields new)"], "next_action": {"kind": "probe", "target": "b.py", "purpose": "discriminate-B"}}
```

继续写入验收 A 并生成三份 handoff：

```sh
python3 $WS --root $W --allow-non-workspace-root record-issue --candidate A --stage repair --issue-file issue.txt --source test-A
python3 $WS --root $W --allow-non-workspace-root record-issue --candidate B --stage repair --issue-file issue.txt --source test-B
python3 $WS --root $W --allow-non-workspace-root record --candidate A --stage acceptance \
  --trigger tA-acc --expected eA-acc --preserve pA-acc --fact "f2|span:z|acc fact A" \
  --hypothesis "h2|consumer|acc claim A|f2|" --chosen h2 --unknown uA2 \
  --next-kind inspect --next-target a.py --next-purpose acc-A
python3 $WS --root $W --allow-non-workspace-root record-issue --candidate A --stage acceptance --issue-file issue.txt --source test-A-acc
python3 $WS --root $W --allow-non-workspace-root write-handoff --candidate A --stage repair --code "a.py:1-2"
python3 $WS --root $W --allow-non-workspace-root write-handoff --candidate B --stage repair --code "a.py:1-2"
python3 $WS --root $W --allow-non-workspace-root write-handoff --candidate A --stage acceptance --code "a.py:1-2"
```

输出（原文，handle 各不相同）：

```json
{"status": "ok", "command": "record-issue", "chars": 21, "candidate": "A", "stage": "repair", "source": "test-A", "note": "issue stored verbatim as passed by the model; the skill never reads host task files and no host auto-copy is claimed"}
{"status": "ok", "command": "record-issue", "chars": 21, "candidate": "B", "stage": "repair", "source": "test-B", "note": "issue stored verbatim as passed by the model; the skill never reads host task files and no host auto-copy is claimed"}
{"status": "ok", "command": "record", "candidate": "A", "stage": "acceptance", "changed_fields": ["(first-record: all fields new)"], "next_action": {"kind": "inspect", "target": "a.py", "purpose": "acc-A"}, "hypotheses": ["h2"], "note": "stored only; no root cause was inferred by this tool"}
{"status": "ok", "command": "record-issue", "chars": 21, "candidate": "A", "stage": "acceptance", "source": "test-A-acc", "note": "issue stored verbatim as passed by the model; the skill never reads host task files and no host auto-copy is claimed"}
{"status": "ok", "command": "write-handoff", "candidate": "A", "stage": "repair", "handle": "/var/folders/9s/sdk1x32s58g8dycwh6c74bqc0000gp/T/p005_working_state/root_e848523ae3823dfc/handoff_A_repair.json", "sha256": "737449e2d2b0a843640c9267025cf8a847783941bb39f84238e0659f1e38ca7c", "chars": 881, "token_estimate": 220, "handoff_trimmed": false, "note": "explorer final text must be this handle only; output_key saves that text, not an auto-merged JSON"}
{"status": "ok", "command": "write-handoff", "candidate": "B", "stage": "repair", "handle": "/var/folders/9s/sdk1x32s58g8dycwh6c74bqc0000gp/T/p005_working_state/root_e848523ae3823dfc/handoff_B_repair.json", "sha256": "d4503d7e67f86fc24f7625d4ec7996d2dbba4450f3f4b6049703270d2b51fb19", "chars": 876, "token_estimate": 219, "handoff_trimmed": false, "note": "explorer final text must be this handle only; output_key saves that text, not an auto-merged JSON"}
{"status": "ok", "command": "write-handoff", "candidate": "A", "stage": "acceptance", "handle": "/var/folders/9s/sdk1x32s58g8dycwh6c74bqc0000gp/T/p005_working_state/root_e848523ae3823dfc/handoff_A_acceptance.json", "sha256": "cf067494c76c8cdf043e34419357a8d89d38ef350594f61c22e9a8f062ec96c2", "chars": 894, "token_estimate": 223, "handoff_trimmed": false, "note": "explorer final text must be this handle only; output_key saves that text, not an auto-merged JSON"}
```

 state 目录共存 9 个命名空间文件，无覆盖（`ls` 原文）：

```text
handoff_A_acceptance.json  handoff_A_repair.json  handoff_B_repair.json
issue_A_acceptance.json    issue_A_repair.json    issue_B_repair.json
state_A_acceptance.json    state_A_repair.json    state_B_repair.json
```

handoff 内容与候选绑定（直接读文件）：

```text
A_repair     -> trigger: tA     | hyps: ['h1']
B_repair     -> trigger: tB     | hyps: ['h9']
A_acceptance -> trigger: tA-acc | hyps: ['h2']
```

向后兼容（缺省候选行为不变）：不带 `--candidate/--stage` 的 `record`
仍写 legacy `state.json`，`show` 读回 `trigger: legacy`，且重读
`show --candidate A --stage repair` 仍为 `tA/h1`（未被 legacy 覆盖）。
非法命名空间被拒绝：`record --candidate 'A/B'` →
`{"status": "error", "error": "bad --candidate (want 1-32 chars of [A-Za-z0-9_-])"}`。
`read-handoff --handle <B handle>` 返回 `ok` 且 `trigger: tB`（按 handle 读取，
无需命名空间参数）。

## 5. patch_store 冒烟（snapshot / list / restore）

命令（`PS=research/012/agent/skills/source-edit/scripts`，`a.py` 先改为 `return 2`）：

```sh
PYTHONPATH=$PS python3 $PS/patch_store.py --root $W --allow-non-workspace-root snapshot --note "candidate A fix" --parent none
```

输出（原文，`patch_id` 每次随机，本轮为 `p_db7088e3f412d30c`）：

```json
{"status": "ok", "command": "snapshot", "patch_id": "p_db7088e3f412d30c", "parent_patch_id": "none", "head": "4228c17f1349876fc7621472222920d29740e092", "files": [{"path": "a.py", "status": "modified", "sha256": "b7fdeefd2ff2fd36afb5919c77890537a8d74c15b1fc316059fc69dfb527a93f"}], "snapshots_kept": ["p_db7088e3f412d30c"], "note": "stored only; nothing here judges which patch is correct"}
```

（注：`files` 另含未跟踪的 `issue.txt`，其 sha256 为
`47157bab72a37a164218e250f31845a362e674b002b05f4574637cbb67346b22`。）

```sh
printf 'def f():\n    return 3\n' > a.py
PYTHONPATH=$PS python3 $PS/patch_store.py --root $W --allow-non-workspace-root snapshot --note "candidate A followup" --parent p_db7088e3f412d30c
PYTHONPATH=$PS python3 $PS/patch_store.py --root $W --allow-non-workspace-root list
```

`list` 确认 `parent_patch_id` 链（候选绑定由调用方传 parent 链保证，
存储本身未改）：

```text
[('p_db7088e3f412d30c', 'none', 'candidate A fix'),
 ('p_5a62d35a87ffe697', 'p_db7088e3f412d30c', 'candidate A followup')]
```

```sh
PYTHONPATH=$PS python3 $PS/patch_store.py --root $W --allow-non-workspace-root restore --patch-id p_db7088e3f412d30c
```

输出（原文）：

```json
{"status": "ok", "command": "restore", "restored_patch_id": "p_db7088e3f412d30c", "head_now": "4228c17f1349876fc7621472222920d29740e092", "files": ["a.py", "issue.txt"], "current_diff_stat": " a.py | 2 +-\n 1 file changed, 1 insertion(+), 1 deletion(-)\n", "note": "Requested worktree snapshot restored; correctness is not inferred."}
```

`cat a.py` 确认回到首个快照内容（`return 2`）。

## 6. 结论

- 候选 A（修复）、B（修复）、A（验收）的 state/issue/handoff 各自独立文件，
  读回内容与写入一致，互不覆盖；legacy 单份路径缺省行为不变；非法命名空间被拒。
- `patch_store` snapshot → list（parent 链）→ restore 冒烟通过，
  restore 后工作树回到指定快照；存储未改，候选绑定由调用方 `parent_patch_id` 链保证。
- 未新建 tests/ 目录与 pytest/unittest 文件；以上均为一次性手动命令。
- 临时工作区 `/tmp/p012_verify_ws` 及系统 temp 下 `root_e848523ae3823dfc` 为验证残留，
  不在仓库内，不影响正式包。
