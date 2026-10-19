# 任务环境准备：扩展确认 6/12 + pr_1981 无效判定（CPU，可复核）

日期：2026-10-19。延续 `task-prep-pilot.md`（schema + cookiecutter-1669 已确认）。
本轮独立重跑确认 feedparser×4、cookiecutter-1779/2010、conan×3 的修前失败与参考修复通过；
cookiecutter-1981 判定为无区分性任务（见 §5），拟按同仓哈希规则替补。

## 1. feedparser×4（全部确认）

| 任务 | buggy commit | PRE（独立重跑） | POST（独立重跑） |
|---|---|---|---|
| pr_334 | `66c91af3…` | `parse()` 抛 `NameError: datetime`（import 时即触发） | 加 `import datetime` + `import time` 后正常解析，bozo False |
| pr_469 | `842710dc…` | `PrefixFileWrapper(b'abc',BytesIO(b'def'))` 先读 2 字节后 `read(-1)` 得 `b'abcdef'`（应为 `b'cdef'`） | 得 `b'cdef'` |
| pr_417 | `ca151773…` | `parse()` 抛 `NameError: name 'cgi' is not defined` | bozo False |
| pr_344 | `8c438e84…` | `import feedparser` 即 `ImportError: cannot import name 'convert_to_idn'` | bozo False，`convert_to_idn` 可导入 |

- 参考补丁：由各自轨迹生产 `str_replace` 重建（`applied=N failed=0` 逐字匹配），
  存于各 `task-env/<iid>/reference.patch`；全部不是 `patch` 列
  （该列是 3MB 的 site-packages 垃圾，已逐字节比对确认不同）。
- 环境依赖：feedparser 在此 era 需 `sgmllib`（py2 模块），评测环境须提供
  （本轮用 `sgmllib3k` 验证通过）。`parse()` 判据用本地字符串，不依赖网络。

## 2. cookiecutter-1779/2010（确认）；1981（无效，见 §5）

- pr_1779（drop jinja2-time）：PRE `StrictEnvironment(context={})` 抛
  `UnknownExtension: No module named 'jinja2_time'`；POST 补丁（新增
  `TimeExtension` + 默认扩展替换）后 `{% now 'utc','%Y' %}` 渲染当年年份。
  **区分性依赖评测环境不装 jinja2-time**（本轮两台 python 均未装，判据成立）。
- pr_2010（嵌套 dict 合并）：PRE `apply_overwrites_to_context` 丢 subkey3/4；
  POST 全部合并。注意首次自写 repro 误调顶层 context 导致 POST 假失败，
  已按调用约定（内层 dict）纠正并双向复测通过。
- 轨迹 diff 含 docs 等无关文件（112KB 级污染）；
  `reference.patch` 只取生产文件 hunks（1779：environment.py+extensions.py；
  2010：generate.py）。

## 3. conan×3（全部确认）

| 任务 | buggy commit | PRE（独立重跑） | POST（独立重跑） |
|---|---|---|---|
| pr_16559（`&:` options 过滤） | `9c84a975…` | `RecipeReference.matches('&:', True)` 为 False | 为 True（`&` 基线 True 不变） |
| pr_13365（子目录自定义命令报错） | `bf8825d6…` | 坏 import 导致 `ModuleNotFoundError` 裸崩，无友好信息 | `ERROR: Error loading custom command mycompany.cmd_hello: …`，不崩且继续加载 |
| pr_13112（remote --force 回归） | `499a7cb7…` | 重复 URL 提示用 `--force`，但 `--force` 参数未识别（exit 2） | `--force` 接受，重复 URL 建仓成功，`remote list` 可见 |

- conan 本地导入需 `colorama/fasteners/requests/patch-ng`（已装用户级，仅环境依赖）。
- pr_13112 轨迹第 2 处生产编辑（remotes.py）经核对是注释级无行为变更；
  参考补丁只取 remote.py 的 CLI 修复（`--force` 透传 + `update` 替代 `move`）。
- pr_13365 的 PRE/POST 差异是“裸崩 vs 友好报错+继续”，判据成立；
  报告原文“silently ignored”指缺少可定位信息，与裸崩观测一致（无命令级提示）。

## 4. 当前 12 题状态

- 已确认 11/12：schema1 + cookiecutter3（1669/1779/2010）+ feedparser4 + conan3。
- pr_1981 无效（§5），待替补 1 题后 12/12。

## 5. pr_1981 无效判定（非 outcome-based 换题，而是任务本身无行为区分）

- 轨迹唯一生产修改：prompt.py 加注释 + docstring 行，零行为变更；
  buggy 代码本来就对非 dict 抛 TypeError（独立重跑确认）。
- 上游 `resolved=true` 是在该注释级修改下通过的 → 与纯 buggy 代码行为一致 →
  隐藏测试在 buggy 上同样通过（确定性测试；无 flakes 证据）。
  故该题不满足“修前失败与参考修复通过的区分”，不能进入 48 次主结果。
- 替补规则：cookiecutter 剩余候选已用尽（4/4），按同仓哈希后延无题可替；
  交接的 pandas/modin 替补需新定顺序与重型环境。最贴近原规则的替补是
  **conan 哈希 rank 3**（同一冻结排序、同仓、conan 充裕，pr ном待定）。
  待执行：取 conan 第 4 顺位 issue，走同一 stage+PRE/POST 流程后冻结替补。

## 6. conan pr_14852（替补 pr_1981，确认）

- 轨迹 2 处生产编辑：`conan/api/subapi/local.py`（`source()` 新增 `remotes` 参数并透传
  给 `load_consumer`）+ `conan/cli/commands/source.py`（CLI 透传 `conan_api.remotes.list()`）。
  第 2 处 anchor 有 1 字节尾换行差异（文件无尾换行，轨迹多 `\n`），已按文件实际构造，
  `git apply --check` 通过。参考补丁 1760 字节，存于对应 task-env。
- PRE（buggy，全新 CONAN_HOME + local-recipes-index remote，py_req2 仅在 remote）：
  `Cannot resolve python_requires 'py_req2/1.0': No remote defined`。
- POST（patched，同环境）：`Downloaded recipe revision …` + `Running source in consumer2`。
- 弯路如实记录：local-recipes-index 布局需 `recipes/<name>/config.yml`（初版缺 `recipes/` 层）；
  conancenter 在离线环境必须 disable，否则 PATCHED 也会因连不上公网 remote 而失败
  （与补丁无关，是环境事实；评测 env 须声明 remote 配置）。
- 替补状态见 `dev12-substitution.json`（`confirmed`）。最终 12 题：
  cookiecutter3 + feedparser4 + schema1 + conan4。

## 7. 全部 12 判据测试双向验证通过（2026-10-19）

每个 `criteria/test_p012_*.py` 均在“buggy 克隆 → 必须 FAIL”与
“参考补丁应用后 → 必须 PASS”双向试跑通过（一次性 pytest，非测试套件；
本地 LibreSSL 告警用 `-W ignore` 压住，GPU 镜像为 OpenSSL 不受影响；
含 pytest-cov addopts 的仓库用 `-o addopts=` 覆盖）：

| 任务 | FAIL_TO_PASS 节点 | PRE（buggy） | POST（patched） |
|---|---|---|---|
| schema pr_331 | test_none_uses_const | `{'type': None}` ≠ 预期 | `{'const': None}` |
| cookiecutter 1669 | test_keep_project_on_failure | 默认删除/keep 缺失 | 默认删、keep 留 |
| cookiecutter 1779 | test_no_jinja2_time_needed | UnknownExtension（无 jinja2_time） | `{% now %}` 渲染当年 |
| cookiecutter 2010 | test_nested_dict_merge_keeps_new_keys | subkey3/4 丢失 | 全合并 |
| feedparser 334 | test_parse_basic_feed | NameError datetime | bozo False |
| feedparser 469 | test_prefix_read_negative_size | 得 `b'abcdef'` | 得 `b'cdef'` |
| feedparser 417 | test_parse_no_cgi_name_error | NameError cgi | bozo False |
| feedparser 344 | test_idn_import_and_parse | ImportError convert_to_idn | 可导入且 bozo False |
| conan 16559 | test_ampersand_colon_matches_consumer | `&:` 不匹配 | 匹配（`&` 基线不变） |
| conan 13365 | test_subfolder_import_error_surfaced | 裸 ModuleNotFoundError 崩溃、无提示 | 友好报错且不崩溃 |
| conan 13112 | test_remote_add_force | `--force` 未识别（exit 2） | 接受且重复 URL 建仓成功 |
| conan 14852 | test_source_with_remote_python_requires | `No remote defined` | 从 localstore 下载并 source 成功 |

- feedparser 判据测试自带 `tests/p012_lib/sgmllib.py`（BSD，test_patch 内聚，
  不污染 agent diff；评测时由 test_patch 应用）。
- 1779 判据含环境断言（eval 环境不得装 jinja2_time），conan 13112/14852 用带
  独立 CONAN_HOME 的子进程 CLI（与独立重跑一致）。
- 参考补丁全部只取生产文件 hunks（trajectory 顺带 docs 变更已剔除）；
  cookiecutter-1981 因零行为差异未入选（§5），由 conan-14852 按同仓哈希顺位替补。
