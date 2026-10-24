# 官方评分机制复核（2026-10-24，Asia/Shanghai）

## 结论

比赛的得分单位是完整任务，每个任务只有 resolved / unresolved；总分为该评测 split 内 resolved 任务数除以全部任务数。没有按照仓库、题型、测试条数或修改量加权，没有单题部分分。129 个四仓库任务是公开开发/训练集，并不是公开排行榜的题目。官方当前说明仍称排行榜测试集约 120 题，公榜/私榜约各半，来自私有仓库。因此，129 题只能用来研究和训练修复能力，不能据此断言隐藏评测的类型分布。

## 当前原始来源及版本

1. 2026-10-24 通过已认证 Kaggle API `competition_list_pages` 直接读取的官方页面：
   - Evaluation：https://www.kaggle.com/competitions/gemma-4-developer-agent/overview/evaluation
   - 模型与执行规则：https://www.kaggle.com/competitions/gemma-4-developer-agent/overview/model-selection-budget-and-harness-rules
   - Data：https://www.kaggle.com/competitions/gemma-4-developer-agent/data
   - 三页原文保存在 `sources/official-pages-20261024.json`。JSON 内保留 API 返回的原始页面名称与完整 content。
2. 当日逐文件下载的官方 `HARNESS_README.md`，保存在 `sources/`。它是公开说明文档，并非我方研究结论。该文档第 624–629 行明确写出 resolution 条件与最终公式。
3. 当日从官方 https://www.kaggle.com/datasets/metric/gemma-4-developer-agent-wheelhouse 下载两个小 wheel，只提取通用框架源码；当前证据目录的四份重点源码保存在 `sources/framework/`，其余可由 `sources/` 中原始 wheel 提取：
   - `swegemma-0.2.11-py3-none-any.whl`，119670 bytes，SHA256 `9ad876581f3acde8ec3c369858f0834e2b9f41a3cffce3ca42f09cb090aea9d3`。
   - `adk_eval_core-0.1.0-py3-none-any.whl`，90060 bytes，SHA256 `e614e1b2fe8d886a634871e4c57f9887e1cc66e8c27300f8503a43c6e3ca82f3`。
   - 当前官方文件清单还列出 `adk_submission-0.2.13`、`google_adk-1.36.1`、`vllm-0.19.1`；本次未下载这些大依赖。
4. 本次没有读取隐藏任务、rerun 或 secret 数据内容，没有启动 GPU，没有执行比赛任务验证，也没有新增测试文件。

所有代码实现结论均以本次重新下载且有 SHA256 的官方原包为准。重点文件：[verification.py](sources/framework/swegemma/harness/verification.py)、[agent_runner.py](sources/framework/swegemma/harness/agent_runner.py)、[execution.py](sources/framework/swegemma/tools/execution.py)、[runner.py](sources/framework/adk_eval_core/runner/runner.py)。

## 单题究竟如何判定

以下代码路径相对于官方 wheel 内的包根目录，行号来自当日下载 wheel 的实际源码。四份重点文件可通过上面的链接查看；其余文件保留于原始 wheel。

- `swegemma/evaluate.py:298–322`：Phase 1 调用 agent，在 Container A 生成补丁。
- `swegemma/evaluate.py:324–351`：有 agent_error 且没有补丁，直接 unresolved。有补丁即使 agent 出错/超时，仍进入验证；不能把超时标签直接等同于 0 分。
- `swegemma/evaluate.py:354–378` 与 `swegemma/harness/verification.py:359,384–427`：Phase 2 在新的 Container B 从原任务 snapshot 重建环境，并建立 clean baseline。A 中的进程状态、临时依赖等不会随补丁带入 B。
- `swegemma/harness/verification.py:429–441`：验证目标文件取 task.test_files，通常由 test_patch 所触及路径提取；不是简单按 FAIL_TO_PASS 列表逐节点执行。
- `swegemma/harness/verification.py:443–489`：应用有效 agent_patch；补丁无法应用直接 unresolved。
- `swegemma/harness/verification.py:493–551`：将目标测试与 agent 触及的受保护测试/配置文件恢复 baseline。
- `swegemma/harness/verification.py:553–599`：应用官方 test_patch；应用失败同样 unresolved。
- `swegemma/harness/verification.py:618–644`：pytest 对目标测试文件执行；无目标时 fallback 到 `.`。因此实际执行同一文件中的其它测试，而不是只跑新加的测试。conftest.py 通常从 positional targets 排除。
- `swegemma/harness/verification.py:646–683`：首先要求 pytest exit_code == 0，然后检查 JUnit XML 和 stdout 是否确有通过测试。缺失 XML、no tests ran、0 passed 等不能骗过得分。
- `swegemma/harness/verification.py:266–301`：XML 必须可解析，存在 testsuite，至少有一个实际 passing test，且 failure/error 总数为 0。
- `swegemma/harness/verification.py:303–340`：required nodes = FAIL_TO_PASS + PASS_TO_PASS。若两列表都空且存在 test_patch，则从 test_patch 抽取测试函数名作为 fallback。每个 required node 必须有匹配 passing 记录，且不能有匹配失败/skip 记录。允许无关测试 skip，因此“全套任何 skip 都失败”不准确。
- `swegemma/harness/verification.py:208–249`：fallback 优先抽取新增 test_ 函数名；没有新增函数名时再抽取 hunk/context 中修改的函数并排除删除项。匹配方法在 252–263 行，会匹配参数化后缀及部分重命名形式，不是严格字符串相等。
- `swegemma/harness/verification.py:363–382`：完全缺失测试规格时不会凭空判成功。

## 受保护文件与补丁提取

`swegemma/harness/verification.py:62–85` 定义受保护文件：conftest.py、pytest.ini、pyproject.toml、tox.ini、setup.cfg、.pytest.ini、sitecustomize.py、usercustomize.py、_swegemma_stubs.py、*.pth；test_*.py、*_test.py，以及 tests/test/testing 目录内的 .py。额外还保护官方 test_patch 涉及路径（429–438 行）。`strip_protected_files_from_patch` 会移除触及这些路径的整个 diff section（88–205 行）。因此修改 pyproject.toml 等并不能保证随 patch 进入评分。

`swegemma/tools/execution.py:76–116`：submit_patch 捕获相对于 _swegemma_baseline（fallback HEAD）的 git diff，并剥离受保护修改。可以多次提交，以最终保存的 patch 为准。`swegemma/harness/agent_runner.py:1048–1103`：如果从未显式 submit_patch，结束/耗尽预算后仍尝试从磁盘提取有效修改。因此“没调用 submit_patch 就必然 0 分”也不准确；稳定策略仍应显式提交。

## 汇总公式与预算

`adk_eval_core/runner/runner.py:24–36`：total = len(task_results)，resolved = sum(r.resolved)，resolution_rate = resolved / total。官方 HARNESS_README.md:628–629 将同一公式指向正式 score()。仓库级统计只是分析视图，未成为四仓库等权宏平均。

当前官方 Evaluation 页面明示：所有题生成补丁合计限 12 小时，包含 sandbox setup，排除 patch validation，可自行配置每题时间限制。HARNESS_README.md:554–566 区分每题 agent loop 预算与总预算：默认每题 60 min，agent loop 预算不含容器安装；Phase 2 验证默认 300 s，scorer 固定为 300 s。源码 `swegemma/budget.py:18,29–33` 有相应默认值，`verification.py:634–643` 优先独立 verification timeout。不能混淆“每题 loop 不含 setup”和“全题 12 小时包含 setup”。预算影响能否生成/验证成功，而非将耗时作为连续分数扣分。

当前官方模型页面已经支持四个量化 variant：31B、12B、E4B、E2B，分别为 gemma-4-{31b,12b,e4b,e2b}-it-qat-w4a16-ct。同一 submission 的 agent/subagent 必须使用同一个 base variant，可附不同 LoRA。官方 HARNESS_README.md:160–189 记载 4 × L4、32768 context 与同一 variant 规则。模型尺寸本身不参与得分权重；选择通过吞吐/资源约束影响最终解决数量。不能沿用 10-13 页面快照中的“仅支持 31B”。

## 不应混同数据构建与线上逐题判分

官方 Data 页面中的 baseline 必须 fail、gold patch 后必须 pass，是任务筛选与验证管线的描述。本次检查的 `verify_task` 在 agent patch 上直接跑验收，并不在每次评分时先重跑 baseline 再比对增量。Data 页面所谓“整个 test suite 通过”也不能代替源码实际的 target-file 范围描述。

## 当前核实的局限

本次直接核实的是当前公开官方页面、说明文档与已发布 wheel。正式私有 scorer 的 `scripts/metric.py` / `scripts/inference.py` 原文件没有在这些 wheel 中发布，不能声称已逐行审计线上私有 wrapper，亦不能断言所有线上环境细节必然与公开包完全相同。公式与 12 小时总预算有当日官方页面确认；具体 XML、protected stripping 等实现有当日官方 wheel 和当日官方 README 双重支持。未获取隐藏任务数量精确值，官方只写 about 120，不能把论坛推算的 58 题当作官方承诺。

## 独立跨题观察：固定修法会互相冲突

仅观察当日下载的公开开发集 `/private/tmp/gemma4-reset-official/tasks.jsonl`（SHA256 `e4b3fd60f69dbc2b9213e54eeb9636db78aefe92c1d06269d73d9f5f8f3c8ad6`）。没有查看隐藏集。

- 第 96 行 requests_6644，base_commit `382fc2c0c6c0ef0874bc65bc1175f97c073e5086`：problem 是 urllib3 会把多个 leading slash 的 request URI 误解释成带 host/port 的完整 URI。参考补丁在 HTTPAdapter.request_url 增加 `if url.startswith("//"): url = f"/{url.lstrip('/')}"`；官方验收要求 `http://127.0.0.1:10000//v:h` 输出 `/v:h`。
- 第 15 行 requests_7315，base_commit `6360477c52303c9445b45fa8744b02d05a2f0905`：problem 是折叠 leading slash 会破坏 S3 presigned URL 的签名。参考补丁恰好删除同一个条件块；官方验收把同一 URL 例子的期望从 `/v:h` 改成 `//v:h`。这是同一库、同一函数、同一表面输入，两个快照和任务契约要求相反操作。不能训练成“URL 双斜杠一律归一化”，也不能归成“同一修法类型”。
- 第 94 行 requests_6757，base_commit `0e322af87745eff34caffe4df68456ebc20d9068`：题目标题是 Test on urllib3 1.26.x，但参考代码改动不是单纯“补测试”，而是显式识别 urllib3 major，并只在 2.x+ 对 str 按 UTF-8 编码计算长度；1.x 保留既有语义。验收同时把字符串长度测试设置为 1.x 下 skip。根因边界是下游依赖如何把字符串编码成请求体，Content-Length 必须随该契约一致，不能概括成“字符串长度永远按 UTF-8 bytes 计算”。

这三个公开题共同证明：分类应保留需要恢复的行为契约、依赖/版本边界和推理操作。可以将它们置于同一“边界表示与外部协议契约”机制家族，但实际修复方向必须由 problem_statement + base snapshot + dependency semantics 决定。单按仓库、标题中的 bug/feature/test，或者“加条件/删条件/编码/URL规范化”等固定动作分类，会抹掉真正决定得分的条件。上述证据是源码/验收静态阅读，并非新实验收益或已运行通过的宣称。

## 补充：修改位置与公开 oracle 的覆盖边界

1. `docs_src/`、`scripts/` 不在保护目录名单内。正常 Python 示例或维护脚本，只要路径没有触发具体测试文件规则、特定配置文件名规则，也不在 task.test_patch 涉及的额外保护路径内，修改就能进入 agent_patch。不能把“修底层库”解释成“只允许包目录下的代码”；源码真正实现的是路径保护规则。FastAPI 的 docs_src-only / scripts-only 修复不应仅因目录位置被排除为非功能任务。
2. 公开集第 78 行 `httpx_3672`（base `4acf5c2c37714cc63b5cf71b3e284fca83c90311`）需求包括 parser keepalive、keepalive 请求读完、complete→reset、server exit 关闭 streams、避免退出异常。金补丁也跨 sync/async parser、pool、server 以及 network 层。但它的公开 test_patch 仅修改 `tests/test_parsers.py`，把 5 个既有测试中的 `p.complete()` 改成 `p.reset()`，没有新增直接针对请求读完、stream 清理或 KeyboardInterrupt 的断言。评分会执行整个 tests/test_parsers.py；required fallback 包括这 5 个修改函数（test_parser、test_parser_server、test_parser_repr、test_client_connection_close、test_server_connection_close）。这能观测 API 重命名与既有 parser 状态回归；不能把该 oracle 覆盖范围等同于完整需求范围。未实际跑验证，因此本报告也不声称“只加 reset 别名一定能通过”。在分类表中应分别保留 requirement_mechanism 与 oracle_coverage，避免把金补丁所有修改都当作得分必需动作。

## 公开 required / skip 冲突的手动控制流复核

`rich_3486` 的 `test_patch` 新增 `test_traceback_finely_grained_missing` 和 `test_traceback_finely_grained`；前者在 Python >=3.11 跳过，后者在 Python <3.11 跳过。官方公开 129/129 行都没有 FAIL_TO_PASS/PASS_TO_PASS 字段，当前公开 helper 因此把两者都抽为 required；任意 Python 版本至少有一个 required skip。

本轮用一次性 Python 从当天官方源码的 AST 提取 `_extract_test_functions_from_patch`、`_node_matches_requirement`、`_validate_junit_xml`，以合成的一过一跳 JUnit XML 复核：模拟 3.10、3.11、3.13 的 required 均未全部通过，返回拒绝。这个证据验证的是公开 helper 的分支逻辑，没有在真实环境执行该任务，也没有审计私有 wrapper 的处理。应标为“公开 harness 的 required-skip 冲突”，不能计作某个 Agent 的能力失败或宣称线上必挂。

## 额外的版本条件审阅

扫描129题公开test_patch并对相关装饰器检查精确公开base_commit，记录在 [oracle-caveats.json](oracle-caveats.json)。其中 `fastapi_14186` 的 fallback 同时含 `test_model_field_default_required`（needs_pydanticv2）和 `test_model_description_escaped_with_formfeed`（needs_pydanticv1）；后者标记在精确基线 `6e49dc029541aaa896fa63c47647ccd3f066df67` 中存在，公开patch未删除。该基线utils将两标记分别定义为 skipif(not PYDANTIC_V2) 与 skipif(PYDANTIC_V2)，故单一主版本环境必有required skip。源URL、摘录及完整下载文件指纹已保留。

总计2处静态互斥、20项条件性候选、3项已排除误报。条件性候选可能在比赛实际环境完全满足，不能擅自从分母删掉，亦不能作为模型失败结论。再次强调，没有执行这些真实任务，也没有验证私有scorer wrapper如何处理这些情况。
