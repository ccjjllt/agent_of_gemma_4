# gemma-4-developer-agent 官方提交契约（可执行清单）

整理日期：2026-11-15。整理范围：仓库保留的官方资产 + 历史官方源码副本 + 真实提交样例。
本文只描述**契约与校验规则**，不含训练方案，也不声称任何分数。

## 0. 证据来源与可信度分级

| 代号 | 文件 | 性质 | 说明 |
| --- | --- | --- | --- |
| **[HR]** | `official/HARNESS_README.md`（55,940 B，sha256 `445000…aa1f`，见 `official/provenance.json:10-13`） | 官方文档（2026-10-24 抓取） | 比赛接口说明；版本可能已漂移 |
| **[SW]** | `official/swegemma-0.2.11-py3-none-any.whl`（sha256 `9ad8…a9d3`） | 官方 wheel，成员可 `unzip -p` 直读 | 评测/沙箱/工具/预算实现 |
| **[AE]** | `official/adk_eval_core-0.1.0-py3-none-any.whl`（sha256 `e614…82f3`） | 官方 wheel | 沙箱/预算/ATIF 跟踪 |
| **[AS]** | git 历史 `d61248b3:research/006/official-data-20261013/wheels-source/adk_submission/*`（`__init__.py:118` 声明 `__version__ = "0.2.13"`） | 官方 `adk-submission` 源码副本，已被 `24cb03b8` 退役 | **唯一可直接读到的 schema/compiler 源码**；`swegemma` 依赖 `adk-submission` 但未随 wheel 打包（见 `[SW] swegemma-0.2.11.dist-info/METADATA:8`） |
| **[ST]** | `research/competition-intel/official-starter/unpacked/`（同目录 `submission.zip`，9 条目/5,528 B，`agent.yaml` mtime 2026-10-12） | **官方 starter kit（数据集 `sample_submission` 原文）** | 最权威的最小合法包 |
| **[GS]** | git 历史 `d61248b3:research/003/runtime-review-20260930/official-starter-source.txt`（307 行） | 官方 Getting Started notebook 源码存档 | 打包与 `eval_config` 读取路径的确凿证据 |
| **[R14]** | git 历史 `d61248b3:research/014/submission/submission-manifest.json` | 真实提交包指纹 | 3,089 B 的真实 `submission.zip` 成员清单 |
| **[R15]** | git 历史 `d61248b3:research/015/inference/evidence-v1/capacity-adapter-diagnostic.json` | 真实 GPU 运行记录 | adapter 真实命名/rank/服务命令 |
| **[TP]** | git 历史 `research/007/official-data-20261017/topic-*.json` | 官方论坛存档 | 官方 staff 答复与参赛者故障报告 |

**分级**：`[契约]` = 代码/官方文档直接约束，违反必失败；`[经验]` = 历史真实故障或社区报告，不保证当前版本仍存在。

---

## 1. 提交布局与可接受的根文件名

### 1.1 两层结构（先 notebook，再 zip）

Kaggle 侧提交的是 **notebook 版本**；notebook 在 `/kaggle/working` 下产出 `submission.zip`，评分端解压后用 `compile_submission` 编译：

```text
notebook 重跑  ──写──▶  /kaggle/working/submission.zip
                              │ 解压（unpacked dir = submission root）
                              ▼
                        compile_submission(submission_dir, ToolRegistry, ModelRegistry, ...)
                              │
                              ▼
                        ADK BaseAgent 树 ── 逐题跑 Container A ── agent_patch ── Container B 判分
```

- 打包方式（官方 starter）：`shutil.make_archive(str(zip_base), 'zip', root_dir=AGENT_DIR)`，因此 **`agent.yaml` 位于 zip 根，不能多套一层目录**。`[GS:293-295]`
- 官方 starter 自带打包断言：累计 `file_size` ≤ `MAX_SUBMISSION_SIZE_BYTES`；除目录条目外每个成员扩展名必须 ∈ `ALLOWED_SUBMISSION_EXTENSIONS`。`[GS:298-305]`
- 目录条目（`configs/` 这类 0 字节条目）是允许的：官方 starter 的 `submission.zip` 就含 3 个目录条目。`[ST] unzip -l`
- 评分端另有容错：`generate_standard_submission` 会在 `submission_dir`、`/kaggle/working`、`/kaggle/tmp` 递归 glob `**/agent.yaml` 与 `**/root_agent.yaml`，取“层级最浅 + 名字优先 agent.yaml”的一个。`[SW] swegemma/submission.py:42-69` 但这只是兜底，**不要依赖**。

### 1.2 目录树（官方 starter 原文）

```text
<submission root>/
├── agent.yaml                  # 必需：根 agent 配置
├── eval_config.yaml            # 可选：Stage 1 每题预算（不参与 compile）
├── configs/
│   └── sampling.yaml           # 可选：被 !include 的采样参数
├── prompts/
│   ├── system.md               # 可选：被 !include 的提示
│   └── analyzer.md
├── sub_agents/
│   └── code_analyzer.yaml      # 可选：子 agent / AgentTool
├── adapters/                   # 可选：PEFT LoRA
│   └── main_lora/
│       ├── adapter_config.json
│       └── adapter_model.safetensors
└── skills/
    └── repo_navigation/
        └── SKILL.md
```
`[HR:74-95]`；目录树与 `[ST]` 实测一致。

### 1.3 根文件名：到底认哪几个？

| 来源 | 接受的根名 | 备注 |
| --- | --- | --- |
| 文档 | `agent.yaml`、`agent.yml`、`root_agent.yaml`、`root_agent.yml`，且**恰好一个** | 0 个 → `MissingRootConfigError`；>1 → `MultipleRootConfigsError` `[HR:97]` |
| 代码 `[AS] discovery.py:24` | `_ROOT_CONFIG_NAMES = ["agent.yaml", "root_agent.yaml"]` | **只有 `.yaml`，不含 `.yml`** |
| 代码 `[AS] discovery.py:180-251` | 先找 `agent.yaml`，再 `root_agent.yaml`；都没有时**回退到“根目录只有一个 yaml/yml”**；0 个或 ≥2 个报错 | 回退分支才可能用到 `*.yml` |
| 代码 `[SW] swegemma/submission.py:46` | 只 glob `**/agent.yaml`、`**/root_agent.yaml` | 不含 `.yml` |
| 官方样例 | `agent.yaml` `[ST]` | — |

> **结论（照此执行）**：根配置**只叫 `agent.yaml`**，扩展名只 `.yaml`。`root_agent.yaml` 在 `find_root_config` 里也能命中，但 `[SW] swegemma/submission.py:46` 的兜底 glob 只认这两个名字，命名越少越好排错。
> **关键坑**：`eval_config.yaml` 一旦存在，根目录就有 2 个 yaml —— 只有当其中一个叫 `agent.yaml`/`root_agent.yaml` 时 `find_root_config` 才在第一步命中；若把根配置改成 `foo.yaml` 且保留 `eval_config.yaml`，就会 `Ambiguous root config` 直接失败。`[AS] discovery.py:226-251`

### 1.4 允许的文件扩展名（严格白名单）

```text
[SW] swegemma/config.py:20-31
ALLOWED_SUBMISSION_EXTENSIONS = {'.yaml', '.yml', '.md', '.txt', '.py', '.json', '.safetensors'}
ALLOWED_ADAPTER_EXTENSIONS    = {'.safetensors'}
```
校验点：`[AS] discovery.py:121-127`（`p.suffix.lower()` 必须在集合内，否则 `SubmissionValidationError: File has disallowed extension`）。
`adk_submission` 默认白名单更宽松（含 `.bin/.pt/.gguf/.csv/.sh/...`，`[AS] limits.py:103-135`），**但比赛用 `build_submission_limits()` 覆盖为上面的 7 项** `[SW] swegemma/config.py:44-58`。

**由此产生的硬性后果（`[契约]`）**：
- `.jsonl`、`.csv`、`.toml`、`.sh`、`.patch`、`.zip` **一律禁止**（哪怕只在 `adapters/` 里当元数据也不行）。
- 大小写不敏感（`suffix.lower()`），但**隐藏文件会失败**：`.DS_Store` 的 `suffix` 为空串 → 不在白名单；`__MACOSX/` 同理。**macOS 打包必须清理。**

---

## 2. `agent.yaml` 完整字段契约

### 2.1 四类 agent（判别式联合）

`agent_class` 缺省/为空 → `LlmAgent`。`[AS] schema.py:374-397`

| `agent_class` | 允许字段 | 必须字段 |
| --- | --- | --- |
| `LlmAgent`（默认） | 见 2.2 | `name`、`instruction` |
| `SequentialAgent` | `name`、`description`、`sub_agents`、`before_agent_callbacks`、`after_agent_callbacks` | `name`、`sub_agents` |
| `ParallelAgent` | 同上 | 同上 |
| `LoopAgent` | 同上 + `max_iterations`（1..500，缺省取 `limits.max_loop_iterations`=500） | `name`、`sub_agents` |

`[AS] schema.py:326-366`；`[AS] builders/workflows.py:123-142`
**没有** `if/switch/condition/selector` 之类编排原语；条件分支只能由 LLM 在运行时选择调用哪个 `agent_tool`。`[AS] schema.py:394-412`

### 2.2 `LlmAgent` 字段逐项

| 字段 | 类型/取值 | 必填 | 说明与证据 |
| --- | --- | --- | --- |
| `name` | `str` | **是** | 也是 `AgentTool` 暴露给模型的名字；重名会冲突 `[AS] schema.py:249` + `[AS] resolvers/tools.py:117` |
| `agent_class` | `"LlmAgent"` | 否 | 缺省 `LlmAgent` `[AS] schema.py:277,301-304` |
| `model` | `str` | 实质必填 | 4 个允许 id 之一（见 2.3）；不写则 `resolve_model` 返回 `None`，ADK `Agent(model=None)` 无法运行 `[AS] resolvers/models.py:77-80` |
| `adapter` | `str` | 否 | 必须是 `adapters/` 下**发现到的名字**；且必须同时给 `model`，否则 `SubmissionValidationError: specifies adapter ... without a base model alias` `[AS] resolvers/models.py:46-58` |
| `instruction` | `str`（≤1,000,000 字符） | **是（无默认值）** | 支持 `{problem_description}`/`{hints}`/自定义 `output_key` 状态插值；`!include *.md` 读原文 `[AS] schema.py:280-282`；`[HR:112,297-301]` |
| `description` | `str`（≤1,000,000 字符，按 instruction 上限校验） | 否 | 父 agent 委派时可见 `[AS] schema.py:250`；`[AS] context.py:156-179` |
| `tools` | `list[str \| {agent_tool: {...}}]` | 否 | 见 2.5 |
| `skills` | `list[str]` | 否 | 相对 submission root 的技能目录；名字要求 kebab-case（`_`→`-`）见 2.7 |
| `sub_agents` | `list[{config_path: ...}]` | 否 | 只允许 `config_path`，无 `code` `[AS] schema.py:45-59` |
| `output_key` | `str` | 否 | 末轮文本写入 `session.state[key]`，可被其它 agent 的 `{key}` 读到 `[AS] builders/llm.py:84` |
| `include_contents` | `"default" \| "none"` | 否 | `none` = 不带历史 `[AS] schema.py:284` |
| `disallow_transfer_to_parent` | `bool` | 否 | 仅在非 None 时下传 `[AS] builders/llm.py:90-93` |
| `disallow_transfer_to_peers` | `bool` | 否 | 同上 |
| `generate_content_config` | `dict` | 否 | 见 2.4 |
| `before_agent_callbacks` / `after_agent_callbacks` | `list[str]` | 否 | 见 2.8（swegemma 路径下被静默忽略） |
| `before_model_callbacks` / `after_model_callbacks` / `before_tool_callbacks` / `after_tool_callbacks` | `list[str]` | 否 | 同上 |

**`extra="forbid"`（全部模型）**：任何未知键 → `SubmissionSchemaError`。常见误写：把 `temperature:`/`max_output_tokens:` 直接放 agent 顶层（必须放 `generate_content_config`）、写 `system_instruction`（被 schema 永久排除）、写 `if`/`when`、写 `tools: [{name: ...}]` 这种 ADK 原生但非本 schema 的形状。`[AS] schema.py:247,202`

### 2.3 允许的 base model（单模型规则）

**允许列表（仅 4 个）**`[HR:174-181]`：

```text
gemma-4-31b-it-qat-w4a16-ct     # 官方 starter 旧默认，历史稳定
gemma-4-12b-it-qat-w4a16-ct     # 当前官方 starter 默认
gemma-4-e4b-it-qat-w4a16-ct
gemma-4-e2b-it-qat-w4a16-ct
```

- 非量化名字（`gemma-4-31b-it`、`gemma-4-27b-it`、`gemma-4-26b-a4b-it`、`gemma-4-9b-it` …）**只**注册在本地 CLI registry，**不能**用于比赛提交。`[HR:178,192-200]`
- 单模型校验：`validate_single_declared_model(agent_dir)` 遍历根配置、`sub_agents[*].config_path`、`tools[*].agent_tool.config_path`，以及目录内**所有**独立 agent YAML；要求 `len(models) == 1`，否则 `ParticipantVisibleError`。`[HR:186-189]`；`[AS] discovery.py:659-814`
- 前缀归一化：`openai/`、`google/`、`hosted_vllm/`、`custom/` 会被剥离后比较。`[HR:188]`
- **单模型校验的隐蔽陷阱（`[契约]`）**：`discover_declared_models` 会把目录里任何**未被引用**的 `*.yaml` 也读一遍，只要该文件同时含 `model` 和 `agent_class|instruction|tools|sub_agents` 之一，就计入已声明模型。`[AS] discovery.py:785-804`
  → 因此 `configs/` 下**任何** YAML 都不要写 `model:`；被弃用的候选 cluster 配置要**移出提交包**，否则“declared multiple base models”。

### 2.4 `generate_content_config`（采样 / thinking）

允许字段（全部可见，`allowed_fields=None`）`[SW] swegemma/config.py:59-67`；`[HR:146]`：

| 字段 | 范围 | 默认（未写时套用） |
| --- | --- | --- |
| `temperature` | ≥ 0.0，有限数，非 bool | 无（走 vLLM/模型默认） |
| `top_p` | 0.0–1.0 | — |
| `top_k` | ≥ 1 | — |
| `max_output_tokens` | **1–32,768** | **16,384** |
| `presence_penalty` / `frequency_penalty` | 有限数 | — |
| `stop_sequences` | `list[str]` | — |
| `response_mime_type` | `str` | — |
| `seed` | `int`（会一路透传到请求体） | — |
| `thinking_config.thinking_budget` | **0–32,768** | **4,096** |
| `thinking_config.include_thoughts` | `bool` | — |
| `thinking_config.thinking_level` | `MINIMAL/LOW/MEDIUM/HIGH/NONE` | — |

证据：`[AS] limits.py:227-259,309-440`；`[AS] schema.py:155-227`；默认值 `[SW] swegemma/config.py:63-66`。

**永久禁止字段**（schema 层就没有）：`safety_settings`、`tools`、`system_instruction`、`http_options`、`response_schema`。写入即校验失败。`[HR:153-154]`；`[AS] schema.py:186-202`

**`thinking_level` 的实操警告（`[经验]+[契约]`）**：经 vLLM OpenAI 兼容端点 + LoRA 时，`thinking_level` 会被映射成 `reasoning_effort`，`swegemma` 又设了 `litellm.drop_params = True`，可能被静默丢弃。官方 starter 的 `configs/sampling.yaml` 就**只写** `thinking_budget` + `include_thoughts`，不写 `thinking_level`。`[HR:149]`；`[ST] configs/sampling.yaml:1-6`；`[GS:75-84]`

### 2.5 `tools`：注册表里只有 9 个名字

`SwegemmaContext.create_tools()` 注册的**全部**工具名 `[SW] swegemma/tools/__init__.py:34-46`：

```text
run_command   read_file   write_file   edit_file
submit_patch  get_status
get_code_neighbors   search_similar_code   get_code_subgraph
```

- 写别的名字 → `ToolNotFoundError`（`available=[...]` 会列全表）`[AS] registry.py:59-75`。
- 技能包额外注入 3 个工具名（由 `SkillToolset` 提供，不需要在 `tools:` 里写）：`load_skill`、`load_skill_resource`、`run_skill_script`。`[SW] swegemma/harness/agent_runner.py:61`
- `agent_tool` 形状（**唯一**合法的“子 agent 当工具”写法）：

```yaml
tools:
  - agent_tool:
      config_path: sub_agents/code_analyzer.yaml   # 相对“当前文件所在目录”，回退 submission root
      skip_summarization: true                     # 默认 false
```
`[AS] schema.py:84-137`；官方 starter 正是这个写法 `[ST] agent.yaml:14-16`。

- 工具子集是**自由裁剪**的：只声明你真正需要的工具能减少 prompt 噪声，且 harness 会按声明过滤提示文本（未声明的工具的说明不会出现）。`[HR:304,326-332]`；`[SW] swegemma/harness/agent_runner.py:332-337,364-373`

**9 个工具的硬行为（写 prompt 时必须按这个假定）**：

| 工具 | 计入 `tool_calls` | 关键限制 |
| --- | --- | --- |
| `run_command` | 是 | 单次上限 `min(300s, max(5, 剩余时间))`；stdout/stderr 各截断 5,000 字符 `[HR:386-390]`；`[SW] swegemma/budget.py:29-32` |
| `read_file` | 是 | **150 行 / 10,000 字符**双截断 `[HR:441-444]` |
| `edit_file` | 是 | 3 级匹配（exact/flexible/regex）；`old_string` 必须唯一（`allow_multiple=False`）、文件不能为空、`old_string` 不能为空 `[HR:458-465]` |
| `write_file` | 是 | 自动 `mkdir -p` `[HR:478-484]` |
| `submit_patch` | **否** | `git add -N . && git diff --binary _swegemma_baseline \|\| HEAD`，返回 `patch_size`/`files_changed` `[HR:406-413]`；`[SW] swegemma/tools/execution.py:76-114` |
| `get_status` | **否** | 免费查询预算 `[HR:415-433]` |
| `get_code_neighbors` / `search_similar_code` / `get_code_subgraph` | 是 | 需该 repo 有预计算 graph/embeddings；无则提示不列出 `[HR:488-533]` |

### 2.6 `sub_agents` 与路径解析语义

- 形状：`sub_agents: [{config_path: xxx.yaml}]`，**没有** `code`。`[AS] schema.py:45-59`
- 解析：`ctx.resolve_sub_config_path()` 先按**当前文件所在目录**拼路径；若不存在则回退到 **submission root** 再拼。`[AS] context.py:98-124`
  → 所以 `sub_agents/code_analyzer.yaml` 里写 `!include ../prompts/analyzer.md` 合法（仍在 root 内），官方 starter 就这么用。`[ST] sub_agents/code_analyzer.yaml:4-10`
- 绝对路径、`..` 越界、symlink → `PathTraversalError`。`[AS] paths.py:22-56,59-128`
- 深度上限 50（`sub_agents` 与 `agent_tool` 都计）。`[AS] discovery.py`/`[AS] builders/sub_agents.py:42-48`

### 2.7 `skills`

- 每个技能目录必须含 `SKILL.md`；目录大小 ≤ 50 MiB；总数 ≤ 1,000。`[AS] discovery.py:562-656`；`[AS] resolvers/tools.py:150-238`
- 技能名由目录名派生：`_` → `-`；重名时用相对路径再消歧。`[AS] discovery.py:635-649`
  → **目录名用 kebab-case**（`working-state` 而不是 `working_state`）。历史记录里存在“含下划线的技能名被官方编译拒绝”的先例（git 历史 `d61248b3:research/012/agent/architecture.md` §2，属 `[经验]`），与上面的 `_`→`-` 派生规则一致。
- `skills:` 里写路径，可以写 `skills/source-edit` 或裸名，且存在 `skill_registry` 时按 4 种候选键匹配。`[AS] resolvers/tools.py:161-183`
- 声明 `skills` 时，`tools:` 里的工具会被塞进 `SkillToolset(additional_tools=...)`，不是丢掉。`[AS] resolvers/tools.py:240-260`

### 2.8 `callbacks`（重要：在 swegemma 路径下等于空操作）

- schema 接受 6 个回调字段；但 `swegemma` 调 `compile_submission` 时**没有传 `callback_registry`**，因此 `ctx.callback_registry is None` → `resolve_callback_fields` 直接返回 `{}`，回调被**静默忽略**（不报错、也不生效）。`[AS] resolvers/callbacks.py:83-95`；`[SW] swegemma/harness/agent_runner.py:616-628`
- 结论：不要在提交里依赖任何 callback 语义（包括“用 callback 强制提交补丁”这类设计）。

### 2.9 结构上限（一次看全）

| 参数 | 值 | 证据 |
| --- | --- | --- |
| `max_total_size_bytes` | **3 GiB = 3,221,225,472**（解压后，含 `adapters/`） | `[SW] swegemma/config.py:19,45`；`[AS] limits.py:184`；`[HR:134]` |
| `max_file_count` | 10,000 | `[SW] swegemma/config.py:48` |
| `max_yaml_files` | 1,000 | 同上 `:49` |
| `max_yaml_size_bytes` | 每文件/**include 累计** 50 MiB | `:46`；`[AS] yaml_loader.py:173-190,295-299` |
| `max_skill_size_bytes` | 每技能目录 50 MiB | `:47` |
| `max_instruction_chars` | 每 agent 1,000,000 | `:50` |
| `max_total_instruction_chars` | 全部 agent 合计 10,000,000 | `:51` |
| `max_agents` | 500 | `:52` |
| `max_sub_agent_depth` | 50 | `:53` |
| `max_loop_iterations` | 500 | `:54` |
| `max_skills` | 1,000 | `:55` |
| `allowed_file_extensions` | 7 项（1.4） | `:56` |
| `adapter_extensions` | `.safetensors` only | `:57` |
| `!include` 深度 | 10（带环检测/节点上限） | `[AS] yaml_loader.py:29,140-153,194-213` |

`!include` 规则：`.md/.txt` → 原始字符串；`.yaml/.yml` → 递归解析；**`*.json`、`*.py` 等一律拒绝**（`!include does not support '.json' files`）。`[AS] paths.py:163-175`；`[AS] yaml_loader.py:24-28`

---

## 3. LoRA Adapter 规则

### 3.1 目录布局与发现

```text
adapters/
└── main_lora/                      # 目录名即 adapter 名（推荐）；也可放裸 .safetensors
    ├── adapter_config.json
    └── adapter_model.safetensors
```
- 只有 `adapters/` 下、后缀 ∈ `{'.safetensors'}` 的文件才算 adapter。`[SW] swegemma/config.py:31,57`；`[AS] discovery.py:416-421`
- 名字派生规则 `[AS] discovery.py:423-495`：
  - 根级文件 → `p.stem`；
  - 子目录里的 PEFT 组 → `str(dir.relative_to(adapters)).replace("/", "_")`，例如 `adapters/a/b/` → `a_b`；
  - 只含 `adapter_model*.safetensors` 或同目录有 `adapter_config.json` 的组会被合并成 **一个** adapter（但 `size_bytes` 是该组所有文件之和）；
  - `training_args`/`optimizer`/`scheduler`/`scaler`/`rng_state`/`trainer_state` 这些 stem 被排除 `:372-381`；
  - 名字冲突先消歧，仍冲突 → `SubmissionValidationError: Adapter name collision`。
- 命名约束（文档）：`[A-Za-z0-9][A-Za-z0-9._-]*` `[HR:206]`。代码用 `_` 拼路径，所以**目录名只用 `[a-z0-9_-]`** 最安全。
- **`adapters/` 里不能放 `manifest.json` 之外的非白名单文件**：`.json` 允许，但 `.jsonl/.bin/.pt` 会被 1.4 的白名单直接拒绝。本仓库 `training/adapter/exporter.py:20` 与 `manager.py:19` 会额外写 `manifest.json`（`.json`，可通过），但它**不参与**官方发现逻辑。

### 3.2 `adapter_config.json` 要求

官方侧（`adk_submission` + vLLM）**只要求**：
- 文件存在，`r` 是正整数 → 用它推断 rank（`infer_adapter_rank`，`[AS] discovery.py:288-347`）；推断不出则回落到 `.safetensors` header 里的 `lora_A/lora_B` 形状。
- 若推断失败，rank 记为 `None`，vLLM 端 `max_lora_rank` 回落 128。`[HR:169]`

vLLM 启动参数（比赛侧）：`enable_lora = bool(adapter_map)`、`max_loras = max(1, len(adapter_map))`、`max_lora_rank = max(detected_ranks) if detected_ranks else 128`。`[HR:169,208]`
官方 starter 则是硬编码 `enable_lora=True, max_loras=8, max_lora_rank=128`。`[GS:176-183]`
→ **`r ≤ 128` 当作硬规则**（超 128 在 starter 配置下无法加载）。

本仓库**内部**更严（训练/导出侧）：`export()` 强制 `r=128, lora_alpha=256, lora_dropout=0.0, bf16`，并校验 `adapter_config.json` 里这几个值。`training/adapter/exporter.py:90-117`；`manager.py:123-127`
这**不是**官方要求 —— 历史真实跑通过的服务 adapter 是 `r=8, lora_alpha=16`（git 历史 `d61248b3:research/009/execution-20261018/gpu-output-v1/agent_exec/run-009-.../new_adapter/adapters/trained_adapter/adapter_config.json`），且能被 vLLM 正常加载。

### 3.3 target module 规则（Gemma 4 + vLLM 特有）

真实 per-layer 命名（`research/015` 的容量诊断 adapter，412 个 target）：`model.language_model.layers.<N>.{self_attn.q_proj|self_attn.o_proj|mlp.gate_proj|mlp.up_proj|mlp.down_proj}`。`[R15]`
本仓库的兼容性审计（`manager.py:183-283`）明确了几条**会导致 fail-closed** 的规则：

| 禁止项 | 原因 |
| --- | --- |
| `full_attention` 层的 `k_proj` | `k_eq_v` 打包问题需要 vLLM PR 61267；否则 V 的 delta 被静默丢掉 `manager.py:260-264,277-278` |
| `full_attention` 层的 `v_proj` | `v_proj_on_k_eq_v_layer` `manager.py:265-268` |
| `kv_shared` 层（`num_kv_shared_layers`）的 `k_proj`/`v_proj` | `kv_shared_layer_k_or_v_target` `manager.py:249-258,273-274` |
| `use_dora` / `use_rslora` / `modules_to_save` / `bias != none` | `unsupported_peft_variant` / `modules_to_save_present` / `bias_training_present` `manager.py:218-223` |
| target 数量与 `expected_target_count`（默认 400）不符 | `target_count_mismatch` `manager.py:189,275-276` |

**结论**：训练 adapter 时只打 `q/o/up/down/gate`（必要时按层白名单），不要打 `k_proj`/`v_proj`。

### 3.4 模型绑定（`adapter:` 到底怎么生效）

1. `model:` 必须是 4 个允许 id 之一（供单模型校验与 `ModelRegistry.get`）。
2. `adapter:` 必须是发现到的 adapter 名。
3. 运行时 `resolve_swegemma_adapter(base_model, info, models)` 把该 agent 的模型**替换成 `openai/<adapter_name>`**（不是 `<base>:<adapter>`，也不是 base 本身）。`[SW] swegemma/harness/agent_runner.py:64-94`；调用点 `:625-627`
4. vLLM 侧以 `--lora-modules <name>=<path>` 注册同名 LoRA，于是请求 `openai/<name>` 命中“base + 该 adapter”。`[HR:208]`；真实命令行见 `[R15]`（`--max-loras 1 --max-lora-rank 8 --lora-modules p015_capacity_diag=/tmp/.../adapters/p015_capacity_diag`）
5. 同一 submission 里**不同 agent 可用不同 adapter**，但只有一个 base model。`[HR:205-208]`

```yaml
# 推荐形状（真实运行配置：git 历史 d61248b3:research/014/eval-12b/agents/adapter/repair.yaml）
name: p014_12b_repair
model: gemma-4-12b-it-qat-w4a16-ct
adapter: p013_behavior_lora          # == adapters/<this>/
instruction: !include prompts/repair.md
output_key: repair_handoff
```

### 3.5 体积预算

3 GiB 总额（含权重）；官方给的规划参考（31B，bf16，7 类投影全打）：`r=16 ~110–220 MB`、`r=32 ~220–450 MB`、`r=64 ~450–900 MB`、`r=128 ~0.9–1.8 GB`。`[HR:209-214]`
实测参照：`r=8`、412 个 target 的 31B adapter = **212,261,592 B (≈202 MiB)**。`[R15]`

---

## 4. 硬限制总表（提交 + 运行）

### 4.1 提交包

见 §2.9（3 GiB / 10k 文件 / 1k YAML / 50 MiB / 白名单扩展名 / 无 symlink）。

### 4.2 运行环境（评分机）

| 项 | 值 | 证据 |
| --- | --- | --- |
| GPU | 4 × NVIDIA L4，24 GB/卡，共 96 GB | `[HR:160-162]` |
| vLLM | `tensor_parallel_size=4`、`gpu_memory_utilization=0.90`、`max_model_len=32768`、`enable_auto_tool_choice=True`、`tool_call_parser="gemma4"`、`reasoning_parser="gemma4"` | `[HR:163-170]`；`[GS:169-185]` |
| 沙箱 | Docker `swebench-sandbox:latest`，`network_mode=none`，**4 GiB RAM / 2 vCPU**，工作目录 `/workspace` | `[HR:251-258]` |
| 12B 特例 | 必须带 `--hf-overrides '{"architectures":["Gemma4ForCausalLM"],"model_type":"gemma4"}'`；官方称评分端已内置 | `[HR:170]`；`[TP] topic-748287.json`（Ryan Holbrook 答复） |

### 4.3 每题预算（`eval_config.yaml`）与 harness 固定值

`eval_config.yaml`（可选，根级，`evaluation:` 键）——**不参与 `compile_submission`，只被 Stage 1 的 `scripts/inference.py`/starter 读取**：

```yaml
evaluation:
  timeout_seconds: 60      # 单条命令超时（默认 300）
  max_tool_calls: 10       # 默认 100（CLI 下 None）
  max_time_minutes: 1      # 默认 60.0
  max_turns: 50            # 默认 500
```
`[ST] eval_config.yaml:1-7`；`[GS:221-230]`；`[HR:98,540-563]`

| 参数 | 默认 | 可覆盖键 | 环境变量 |
| --- | --- | --- | --- |
| 会话墙钟 | 60.0 min | `evaluation.max_time_minutes` | `SWE_MAX_TIME_MINUTES` |
| 工具调用 | 100（CLI `None`） | `evaluation.max_tool_calls` | `SWE_MAX_TOOL_CALLS` |
| LLM 轮次 | 500 | `evaluation.max_turns` | `SWE_MAX_TURNS` / `SWE_MAX_LOOP_ITERATIONS` |
| 单命令超时 | 300 s | `evaluation.timeout_seconds` | `SWE_TIMEOUT_SECONDS` |
| Phase2 pytest 超时 | 300 s（固定） | — | `SWE_TIMEOUT_SECONDS` |
| 命令输出上限 | 5,000 字符 | — | — |
| `read_file` | 150 行 / 10,000 字符 | — | — |
`[HR:552-563]`；默认值常量 `[SW] swegemma/budget.py:14-33`

- **容器准备时间不计入 agent 时间预算**（`start_agent_session()` 在解压/安装之后才起算）。`[HR:566]`
- 上下文管理：`compaction_interval=15, overlap_size=2, token_threshold=14,336, event_retention_size=5`；`context cache min_tokens=2,048, ttl=1,800s`；瞬态错误重试 5 次指数退避。`[HR:569-578]`
- 续跑 nudge 上限 **3**（连续无有效工具调用的轮次）。`[HR:357-367]`；`[SW] swegemma/harness/agent_runner.py:784-785,965-975`

### 4.4 整场限制（最容易被忽略的一条）

- **官方原文**（Competition Overview，归档 git d61248b3:research/014/submission/failure-20261021/pages.json）："Your agent has a limit of 12 hours to submit patches for all tasks, inclusive of sandbox setup time, but excluding time for patch validation. You may, but are not required, to set per-task time limits in your submission's `eval_config.yaml`." —— 这是搞垮过一次真实提交的规则。
- **题量**：同页原文 "There are about 120 tasks in the test set, evenly divided between the public and private splits." → 约 120 题，public/private 各半。
- **真实失败案例**：提交 57438326，API `status=COMPLETE`、`public_score=""`、`error_description="Your submission notebook exceeded the allowed runtime."`（`d61248b3:research/014/submission/submission-manifest.json`）。
- **算一遍**：120 题 × 8 min ≈ 16 h > 12 h（仅 agent 循环，还没算 setup）。这就是历史有效提交把 `max_time_minutes` 压到 **5**（≈10 h）、`timeout_seconds` 压到 **60** 的原因。
- 官方 staff 明确：4 个 variant 使用**同一套**硬件、同一 12 小时总限、同一 `eval_config` 预算。`[TP] topic-748287.json`

### 4.5 评分口径

- 单题 `resolved = (pytest exit_code == 0) AND JUnit XML 校验通过`：`passed_tests > 0`、`failures == 0`、`errors == 0`、所有必需节点（FAIL_TO_PASS/PASS_TO_PASS/test_patch 提取出的函数）显式通过且未被 skip。`[HR:624-627]`
- 总分 = **Resolution Rate** = 解决题数 / 评测题总数 ∈ [0,1]。`[HR:628-629]`
- 保护文件（改了就必被剥离/回滚）：`test_*.py`、`*_test.py`、`tests|test|testing` 下的 `.py`、`conftest.py`、`pytest.ini`、`pyproject.toml`、`tox.ini`、`setup.cfg`、`.pytest.ini`、`sitecustomize.py`、`usercustomize.py`、`_swegemma_stubs.py`、`*.pth`。`[SW] swegemma/harness/verification.py:62-85`；`[HR:591,607,614-615]`
- 未调用 `submit_patch` **不等于** 0 分：harness 会回退抓 `git add -N . && git diff --binary`，只剥离保护文件。`[HR:592-594]`

---

## 5. 提交前本地预检清单（只读命令）

> 本机（macOS，python3.11）**没有** 安装 `adk_submission`/`swegemma`（已核对 `importlib.util.find_spec` 全为 None），也没有 Docker 沙箱镜像。因此本地只能做**静态契约校验**；真正 `compile_submission` 必须在有 `google-adk` 的主机/服务器上做。
> 按项目约定，下面都是一次性只读命令，不新增测试文件、不写仓库内临时产物（临时文件写 `/private/var/folders/.../T/opencode` 或 `/tmp`）。

### 5.1 一条命令跑完的静态检查

```bash
SUB=./agent_dir            # 解压后的 submission root（含 agent.yaml）
python3 - "$SUB" <<'PY'
import sys, pathlib, zipfile, yaml, os
root = pathlib.Path(sys.argv[1]).resolve()
ALLOWED = {'.yaml','.yml','.md','.txt','.py','.json','.safetensors'}
MAX = 3*1024*1024*1024
files = [p for p in root.rglob('*') if p.is_file()]
sym   = [p for p in root.rglob('*') if p.is_symlink()]
size  = sum(p.stat().st_size for p in files)
print('files      =', len(files), '(<=10000)', len(files) <= 10_000)
print('unpacked   =', size, '(<=3GiB)', size <= MAX)
print('symlinks   =', [str(p.relative_to(root)) for p in sym], '(must be [])')
bad = [str(p.relative_to(root)) for p in files if p.suffix.lower() not in ALLOWED]
print('bad ext    =', bad, '(must be [])')
roots = [n for n in ('agent.yaml','agent.yml','root_agent.yaml','root_agent.yml') if (root/n).exists()]
print('root cfgs  =', roots, '(推荐恰好 ["agent.yaml"])')
# YAML 可解析 + 顶层是 mapping（!include 用 SafeLoader 无法解析，这里只看根文件结构）
for n in roots:
    txt = (root/n).read_text(encoding='utf-8')
    print(n, 'has !include:', '!include' in txt)
PY
```

### 5.2 zip 布局检查（打包后）

```bash
unzip -l submission.zip           # 期望：agent.yaml 在根；无 __MACOSX/、无 .DS_Store
python3 - submission.zip <<'PY'
import sys, zipfile, pathlib
ALLOWED = {'.yaml','.yml','.md','.txt','.py','.json','.safetensors'}
MAX = 3*1024*1024*1024
zf = zipfile.ZipFile(sys.argv[1])
infos = zf.infolist()
total = sum(i.file_size for i in infos)
names = [i.filename for i in infos]
print('entries    =', len(names), 'total =', total, '<=3GiB', total <= MAX)
print('root files =', [n for n in names if '/' not in n.rstrip('/')])
bad = [i.filename for i in infos
       if not i.is_dir() and pathlib.Path(i.filename).suffix.lower() not in ALLOWED]
print('bad ext    =', bad, '(must be [])')
print('hidden/junk=', [n for n in names if '__MACOSX' in n or '.DS_Store' in n], '(must be [])')
print('agent.yaml at root:', 'agent.yaml' in names)
PY
```
官方 starter 的等价断言：`[GS:298-305]`。

### 5.3 schema 级校验（不需要 GPU / 不需要 google-adk）

历史已执行过的做法（`research/012/agent/compiler-precheck.md`）：只把 `errors/paths/limits/schema/yaml_loader` 五个纯依赖模块放到**仓库外临时目录**，用 pydantic+pyyaml 做 `SandboxedAgentConfig.model_validate(load_yaml(...))`：`!include` 沙箱解析 + 字段/类型/越界全部覆盖。
 服务器侧的**完整编译**请直接照抄官方 starter 第 4/5 节（VllmServer → create_model_registry → EvalConfig+Evaluator，评分为真：会校验工具/模型/adapter 注册表与真实 vLLM 加载）[GS:152-185,234-270]。

可行的**静态**替代（无需安装，仅读 git 历史里的官方源码 + 系统 pyyaml/pydantic）：

```bash
# 1) 取出官方 adk_submission 纯 schema 子集到临时目录（不落仓库）
TMP=$(mktemp -d) && for m in errors paths limits schema yaml_loader; do
  git show d61248b3:research/006/official-data-20261013/wheels-source/adk_submission/$m.py > "$TMP/$m.py"; done
# 2) 在 $TMP 下 import schema，逐文件 model_validate
PYTHONPATH="$TMP" python3 - "$SUB" <<'PY'
import sys, pathlib, yaml
from yaml_loader import load_yaml
from schema import SandboxedAgentConfig
root = pathlib.Path(sys.argv[1]).resolve()
for p in sorted(root.rglob('*.yaml')) + sorted(root.rglob('*.yml')):
    try:
        cfg = SandboxedAgentConfig.model_validate(load_yaml(p, root))
        print('OK  ', p.relative_to(root), type(cfg.root).__name__, getattr(cfg.root,'name',''))
    except Exception as e:
        print('FAIL', p.relative_to(root), type(e).__name__, str(e)[:300])
PY
```
> 注意：第 2 步需要 `pydantic`+`pyyaml`。若要连 `limits`/`yaml_loader` 的默认上限一起校验，还需把 `limits.py` 里被 `schema.py` 引用的 `ensure_no_traversal_components` 一并带上（`paths.py` 已包含）。

### 5.4 人工必查项（逐条打勾）

- [ ] 根目录**只有**一个根配置，名叫 `agent.yaml`。
- [ ] 每个 `LlmAgent` 都有 `name` + `instruction`（含子 agent、AgentTool 引用的 YAML）。
- [ ] 所有 agent（含未被引用的 YAML）的 `model:` 完全一致，且 ∈ 4 个允许 id。
- [ ] 每个 `adapter:` 名都能在 `adapters/` 里发现；每个带 adapter 的 agent 都写了 `model`。
- [ ] 每个 `adapter_config.json` 的 `r ≤ 128`；`target_modules` 不含 `k_proj`/`v_proj`（Gemma4 全注意力层）。
- [ ] `tools:` 里只有那 9 个名字 + `agent_tool:` 结构。
- [ ] `generate_content_config` 只出现 2.4 表中的键；**没有** `thinking_level`（除非你确认要走 `reasoning_effort`）。
- [ ] `max_output_tokens + thinking_budget` 的合计留白：上下文总长 32,768。
- [ ] `eval_config.yaml` 的四项预算与 12 小时总量自洽（题数 × `max_time_minutes` + setup 余量）。
- [ ] `!include` 目标都是 `.md/.txt/.yaml/.yml`，且都在 root 内、无 symlink。
- [ ] 无 `.DS_Store`、`__MACOSX`、`.jsonl`、`.sh`、`.patch`、`.zip`、`.pyc`、`__pycache__`。
- [ ] zip 里 `agent.yaml` 在**根**（不是 `xxx/agent.yaml`）。
- [ ] 清理 `/workspace` 里的临时脚本要靠 agent 自律（见 §6.9），包里不需要任何运行时脚本。

---

## 6. 已知会变成 "Notebook Threw Exception" / 空分的坑

按“真实发生过”的证据排列。

### 6.1 12B 模型在评分端启动失败（历史最集中的一类）`[经验]`

- 2026-10 多名参赛者报告：**同一份包，31B 有分，换成 12B 就 `Notebook Threw Exception`**；本地 wheelhouse 正常。`[TP] topic-749259.json`；`topic-748287.json`
- 根因链（官方 staff 已确认并修复）：
  1. 12B 的 `architectures` 是 `Gemma4UnifiedForConditionalGeneration`，vLLM 回退到 Transformers 后端 → `ValueError: Expected hidden_size to be 3840, but found: 256`；官方要求加 `--hf-overrides '{"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"}'`，并称 **scorer 已内置**。`[HR:170]`；`[TP] topic-748287.json`
  2. checkpoint 含 9 个 `model.vision_embedder.*` 张量，`Gemma4ForCausalLM.load_weights` 不跳过 → `KeyError: 'vision_embedder.patch_dense.weight'`；需 wheelhouse v31+（跳过该前缀）。
- **操作建议**：先用 31B 打通全链路；切 12B 前必须在**同一 wheelhouse 版本**上本地起一次 `VllmServer` 并成功 `compile_submission` + 一次真实请求，再提交。

### 6.2 整场 12 小时超时（`COMPLETE` 但无分）

- 报错原文：`Your submission notebook exceeded the allowed runtime.`；API `status=COMPLETE`、`public_score=""`。`research/014/submission/submission-manifest.json`；`failure-20261021/README.md`
- 关键点：**`COMPLETE` ≠ 成功**，必须同时读 `error_description` 与 `public_score`。
- 触发条件：`eval_config` 的 `max_time_minutes` 太大 × 题量（~120）× 串行。
- 有效修复（已在 57471497 上验证拿到分数）：`max_time_minutes: 5`、`timeout_seconds: 60`、`max_turns: 40`、`max_tool_calls: 28`。`research/014/submission/runtime-v2/README.md`

### 6.3 `ParallelAgent` + 工具调用在 ADK 1.36 上损坏 `[经验]`

- 社区报告 adk-python issue 7324：`ParallelAgent` 带 tool call 会失败；官方表示“此时迁移 ADK v2 风险太大”，未迁移。`[TP] topic-748287.json`
- **操作建议**：不要用 `ParallelAgent` 承载会调工具的 agent；用 `SequentialAgent` 或 `agent_tool` 串行。

### 6.4 根配置歧义 / 缺失 `[契约]`

- 0 个：`No root YAML config found. Expected 'agent.yaml' or 'root_agent.yaml'`。
- ≥2 个根级 yaml 且都不叫这两个名字：`Ambiguous root config: found 2 YAML files ...`。`[AS] discovery.py:240-251`
- 0 个时的兜底报错（notebook 流）：`No agent.yaml or root_agent.yaml found in submission. Searched directories: ...`。`[SW] swegemma/submission.py:60-67`

### 6.5 未知字段 / 缺 `instruction` `[契约]`

- `extra="forbid"` → 多写一个 `metadata:`、`version:`、`prompt:` 就 `SubmissionSchemaError`。`[AS] schema.py:247`
- `instruction` 无默认值 → 漏写直接 pydantic 报错。`[AS] schema.py:280-282`

### 6.6 `!include` 打到不允许的扩展名

- `!include configs/foo.json` → `!include does not support '.json' files (allowed: .md, .txt, .yaml, .yml)`。`[AS] paths.py:163-175`
- 想复用 YAML 片段只能放 `.yaml/.yml`；想嵌 JSON 只能改写成 YAML 或直接内联。

### 6.7 多个 base model（含“误伤”）

- 明确写两个不同 `model:` → `ParticipantVisibleError`（多模型）。
- 隐蔽情形：`configs/` 下残留一个带 `model:` 的旧采样/候选 YAML → 被 `discover_declared_models` 的“全目录扫描”计入。`[AS] discovery.py:785-804`
- 另一隐蔽情形：`openai/gemma-4-31b-it-qat-w4a16-ct` 与 `gemma-4-31b-it-qat-w4a16-ct` 混用时，前缀会归一化，**不会**报错；但混用两个真正的不同 id 一定失败。`[HR:188]`

### 6.8 adapter 引用错误

- `AdapterNotFoundError(name, available=[...])`：名字与 `discover_adapters` 派生的名字不一致（最常见：目录 `adapters/main_lora/` 但 YAML 写 `adapter: main-lora`，`-` vs `_`）。`[AS] resolvers/models.py:46-53`
- `specifies adapter ... without a base model alias`：只写 `adapter` 不写 `model`。`[AS] resolvers/models.py:54-58`
- adapter 目录里没有 `adapter_config.json`：rank 推断失败 → 仍能注册，但 vLLM 侧加载/对齐更易失败；且本仓库导出器要求必须有。`[AS] discovery.py:288-347`；`training/adapter/exporter.py:107-109`

### 6.9 补丁为空的“静默 0 分”

- agent 只要**没**调 `submit_patch`，harness 会回退抓工作树 diff；但如果工作树也没改动 → `agent_patch` 为空 → 该题直接 0。`[HR:592-594]`；`[SW] swegemma/harness/agent_runner.py:1082-1096`
- agent 在 `/workspace` 里留下的临时复现脚本**会进补丁**（`git add -N .`）。要放 `/tmp`。`[HR:687-688]`
- 改 `pytest.ini`/`conftest.py`/测试文件都会被剥离，等于白做且浪费预算。`[HR:685-686,614-615]`

### 6.10 macOS 打包噪声 `[契约]`

- `.DS_Store`：`suffix` 为空 → 不在 7 项白名单 → `File has disallowed extension ''`。
- `__MACOSX/` 里的 `._*` 文件同理。
- zip 里用 `-x` 排除，或打包前 `find . -name '.DS_Store' -delete`（对外部交付物执行，不动仓库）。

### 6.11 callback 写了但不生效 `[契约]`

- `swegemma` 编译时不传 `CallbackRegistry` → 回调被静默忽略（不报错）。`[AS] resolvers/callbacks.py:83-95`；`[SW] swegemma/harness/agent_runner.py:616-628`
- 因此“用 `before_tool` 强制 `submit_patch`”“用 `after_agent` 兜底提交”这类方案在比赛环境**不可能生效**。

### 6.12 `thinking_level` 被丢弃 / 截断 `[契约]+[经验]`

- `thinking_level` → `reasoning_effort` → `litellm.drop_params=True` 下可能被丢。`[HR:149]`；`[SW] swegemma/__init__.py:83-88`
- `max_output_tokens` 打满 + 大 `edit_file` → `<|tool_call>` 未闭合，nudge 提示“拆小编辑”。`[HR:683-684,363-367]`

### 6.13 `eval_config.yaml` 写成 agent schema 的一部分

- `eval_config.yaml` 只有 `evaluation:` 一个键，**不要**把它 `!include` 进 `agent.yaml`（`evaluation` 不是 schema 字段 → `extra=forbid` 报错）。
- 它也不会被 `compile_submission` 读取，只影响 Stage 1 预算。`[HR:98]`；`[GS:221-230]`

---

## 7. 最小可用模板（6 工具单 agent，等价于历史已提交并通过编译的包）

> 官方 starter 本身是 12B + 9 工具 + 1 个 `agent_tool` 子 agent（`[ST] agent.yaml:1-17`）。下面是最小可运行子集；用 31B 是为了避开 §6.1 的 12B 服务端风险。

```yaml
# agent.yaml
name: my_agent
model: gemma-4-31b-it-qat-w4a16-ct
instruction: !include prompts/system.md
tools:
  - run_command
  - read_file
  - edit_file
  - write_file
  - get_status
  - submit_patch
generate_content_config: !include configs/sampling.yaml
```
```yaml
# configs/sampling.yaml
temperature: 0.2
top_p: 0.95
max_output_tokens: 8192
thinking_config:
  thinking_budget: 4096
  include_thoughts: true
```
```yaml
# eval_config.yaml
evaluation:
  timeout_seconds: 60
  max_tool_calls: 28
  max_time_minutes: 5
  max_turns: 40
```
体积参照：该形状的真实提交包 **3,089 B / 4 个文件**（`agent.yaml`、`configs/sampling.yaml`、`eval_config.yaml`、`prompts/system.md`）。`[R14]`

---

## 8. 与训练侧（本仓库）的接口约定

- 导出 adapter 时目标目录必须长成 `adapters/<name>/`，且含 `adapter_config.json` + `adapter_model.safetensors`；本仓库 `training/adapter/exporter.py:106-109` 与 `manager.py:113-127` 已强制这一点。
- 若要在提交里用 `r=128/alpha=256/dropout=0/bf16` 之外的配置，**必须绕过本仓库导出器**（它会拒绝）；官方对 `r` 只要求 `r ≤ 128` 能被 vLLM 加载（`max_lora_rank`）。历史真实提交用过 `r=8`。
- `manifest.json`（`training/adapter/manager.py:19`）不进 `adapters/<name>/` 更安全（虽然 `.json` 在白名单内，但它对官方发现逻辑无意义，且会增加一个“独立 yaml 扫描”无关但会占体积的文件）。
- 提交包只允许 7 种扩展名，因此**训练日志、数据副本、`.jsonl` 轨迹都不能进包**。
