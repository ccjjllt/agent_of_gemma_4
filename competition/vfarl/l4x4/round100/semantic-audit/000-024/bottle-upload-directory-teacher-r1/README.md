# Bottle4：目录保存主契约的新真实示范

本次 fresh teacher 补齐固定索引4、`bottlepy__bottle.a8dfef30.func_basic__dzcdqzqa` 的工具观察缺口。旧示范真实复现了 overwrite=False，但没有调用题面主分支 directory destination；旧 patch 同时修了该分支。这里保留同一题、原题面、snapshot、生产 patch 和完整原 oracle，没有覆盖旧 trace 或当前100的 supervision-r2/trainer-input-r2。

唯一准入 run 是 `actual-run-r2`。相同的有限 inline API 命令先分别观察两条分支，最后再断言，避免第一个异常遮住第二条观察：

| 分支 | baseline 真实观察 | 修复后真实观察 |
| --- | --- | --- |
| FileUpload.save(directory) | IsADirectoryError，目标文件不存在、目录为空 | `<directory>/incoming.txt` 存在且字节等于上传内容 |
| 已有文件，overwrite=False | 无异常，原字节被覆盖 | OSError，原字节保留 |

两个命令都绑定 workspace 的 bottle.py 导入路径与源码 SHA；baseline 源码为 `2e36abac5d2999429efaee26cff3532b8801e45d543cb0f0ae163b8fa1bd6f04`，修复后为 `6b32ce1f938c89877107e320dc4d0b9d3561eeda93e10e12bd4032eb814c8817`。原完整 oracle 实际执行：baseline 365例中2失败，修复后365通过，官方0→1；提交 patch 与旧 teacher 字节完全相同。没有隐藏 oracle 命令进入模型 history，没有新增测试文件。

source-inventory 中每个原文件 hash/mode 已恢复，source_tree_sha256 逐字保持 `7e6f05801df65c76cea23d1590a9b0d0c009518bd3161e812385e0031f05c612`。只去掉归档中原清单未含的 setup conftest.py、pytest.ini 和排除项 .git 元数据。预算保持原 int 60分钟、100 tool calls、500 turns，command/verification timeout各300秒。实际 prompt SHA 仍为 `311ab9e6b8da37c03d498e8ec83dbecd1425b3e185d874da9999a5cfcd26bdca`。

旧 .venv-sfr 依赖目录已不存在，本次使用已存在的 macOS CPython3.12 official-host cache；新 run-summary 如实记录该路径和发行 metadata，没有伪装成旧 profile 或 Docker 等价环境。source_bytecode_policy 使用当前实际 LEGACY 常量 `legacy-default`。准备阶段曾因系统 Python tar API、未更新命令审阅 hash，以及误写 policy 名称而停止；这些均发生在任何工具／oracle 执行前。保留的 `actual-run/README.md` 说明该未执行目录，不能消费为示范。

已通过现有 normalize_results 严格规范化，并由固定 formatter（SHA `5a9ce116…`）和已验证 CT tokenizer/template 导出独立 action-view：7条真实工具动作加1条实际 terminal，1378监督 token，最长4306，不截断。它是 Codex 编写并审阅、实际工具执行的 off-policy 示范，thinking=0，无模型采样／行为 logprob，不是 RL pair 或新算法收益。

组装入口是 `export-bindings.json`，完整 teacher/canonical/messages/action 路径与 SHA 均在其中。推荐独立输入 `gemma-sft/sft-actions.jsonl`，SHA `1cd6410a3fdddebe8055af892f7df61779b5a6a4542981e0201aaf772af711d1`；manifest SHA `63c3382d060de76331d4bca6cc4c2e522ee209d14abfa8f5265cadf596078f55`。主100新消费版本由 root 另行组装，本目录未自动混入。
