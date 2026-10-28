# 唯一待提交CPU r5：uv窄修＋一次非门控Linux进程观测

此最终包沿用[保留的UV-only r5](../carrier-r5/README.md)，其run SHA`4c945f17c83961b45e03e2054d6ea557db54e1399f9a2e66043eb83489b52f2a`不回写。只有本目录计划提交为`jiyujia/vfarl-current-four-cpu-runtime-repair-r5`；新run SHA`4936acd37fb5102e426b79c952e20f8f607b6e144b7afb1a2f5544e18ca0b21a`，metadata SHA`2390aeded4eb08fe5e7fe2fbf651a067c25d88156ec1fdf2317f299020136f39`。目前未提交。

四题、当前R3 task／prompt投影、原snapshot／patch／oracle、legacy-default与32个SOURCE_BUNDLE文件完全沿用UV候选。private、GPU/TPU关闭、models=[]；uv只复用真实可执行PATH binary或同输入清单SHA匹配wheel，保持原安装版本／CLI flags并明确UV_OFFLINE=1，不下载新包。原r4实际失败、无oracle结果及所有历史null不改。

新增一次Linux进程机制观测，调用位点在main创建本次out后，仅MODE=prevalidate执行，早于material展开、uv修复及四题验收。外层调用和helper均捕获异常；unsupported、timeout或failure只记录真实日志／receipt，**不门控CPU四题**，之后继续原流程。具体差异见[source-diff.md](source-diff.md)，原始inline程序见[budget-observation-inline.txt](budget-observation-inline.txt)。没有新增测试文件、测试开关或通用gate框架。

观测只嵌入经root审阅的readable control source SHA`fd25b6732986acd99ecfe8cb3d0d5f192a994362d8e8739297e258eb8c9412d4`（对应实际GPU wrapper SHA`6796849539e4bdd27d53e0425a052dfb7184a96a9dc775eeb84c5c311cbbddf3`）。在fresh subprocess中，AST仅取imports／literal assignments／class及非main函数，核完整source SHA后编译，不调用main、不含或解码模型payload。调用原`run_bounded_command`，execution0.75秒、cleanup1.5秒；实际子进程再创建setsid孙进程，两者各自sleep2秒，孙进程忽略TERM以观察KILL／adopt／reap。外层等待至多3秒。

真实stdout输出PID／PPID／SID，primitive returncode、cleanup、时间与source SHA保存于`owned-process-budget-observation/`。预期timeout124不是预先填入的结果，实际unsupported／failure亦原样保留；目前尚未Linux执行，不能称硬预算机制已验证。该观测不运行模型、工具策略或oracle，不改变任务source及验收。GPU研究是否可提交由root／comp依据实际证据另定，不由此CPU文件自动启动。
