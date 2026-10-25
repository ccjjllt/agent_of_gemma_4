# 官方六工具 CPU 数据重放

当前数据扩容集中在[S600](../s600/README.md)，400共同与200专科名单互不重复。下段81题数字为该阶段开始前的历史起点；最新运行、对白与课程数量以本目录机器清单和[S600课程清单](../s600/course-manifest.json)为准。

2026-10-25。当前81条可用示范（原96中的61题＋明确补充20题），G28/S16/F18/R19，实际导出716条决策（641工具调用＋75实际终止回答）、89,774个助手监督tokens，最大动作上下文8,973。精确实物以[准入清单](canonical-results-manifest.json)、[重放总览](replay-summary.json)、[对白清单](gemma-sft-manifest.json)和[四份主责/共享课程](adapter-curricula-manifest.json)为准。

`run_official.py`正常导入原版swegemma0.2.11、adk-eval-core0.1.0、adk-submission0.2.13、google-adk1.39.1。真实Context/Budget、六工具、路径保护、verify_task和JUnit matcher未修改。动作由Codex编排审阅，legacy human_reviewed字段不是用户人工标注或Gemma采样。每条命令有审阅hash，原始参数、观察、双轮干净验证、快照/补丁与源绑定均保存。

原F2P/P2P全部保留，ExternalTrainTask只提供其完整原文件并集。缺失测试只按固定上游恢复；没有专用测试套件、断言修改或缩成只跑required节点。81条修复全部满足原版修后验收；其中80条原ID逐字保留，Slugify一个保守unittest映射单列，完整原要求均有覆盖。

环境profile保持显式：source-layout优先当前源码；短路径alias只绑定真实包；preload执行真实initializer/原console入口；task-site-packages分离官方父依赖；task-python恢复源版本。新copy模式把已有依赖物理复制至sandbox正常purelib，排除旧bytecode，不改sysconfig或测试。sandbox-executable仅映射已审阅依赖binary并记录hash，不能覆盖Python/pytest。

PennyLane按原CI框架版本用JAX/TF/Torch CPU完成原全部要求，Linux/Kokkos未复刻。Hydra3.10不实现PYTHONSAFEPATH变量的限制、Moto3.11与原Gym3.12区别都明确保存。当前仍为官方SubprocessManager本地后端，不是Docker等价或安全隔离证明。新版记录父NumPy/Pydantic真实导入版本/路径/hash，旧样本历史边界不补写。

`normalize_results.py`读取实际task/recipe/messages/events/双轮JUnit与固定来源，原报告不改。多次报告按instance ID去重，每题至多一条当前训练episode，全部尝试保留。补充任务只有显式supplemental catalog可入池，不能仅凭report自报资格。两个MonkeyType基线例外仅绑定固定task SHA/基线XML、全部原节点被观察且有真实因果失败：qualified-name样本为52失败/309已通过，Tuple[()]样本为3失败/1已通过；后者源问题提及Python3.7，本地实际3.11差异不能否定历史标签。已通过的节点不充当错误动作证据。默认证据规则不整体放宽，修后原matcher不改。PennyLane基线JUnit相对原tests/pytest.ini解释地址，配置字节由真实快照hash核对，类名和参数仍精确匹配。

题面不允许静默改写。77条保留原发布题面；TinyDB缓存失效、Boltons集合相等、PtyProcess存活查询和Dataset事务退出4条原题面为空，已根据公开行为契约和实际观察写明澄清题面，并以该题面重新执行完整六工具与双轮验证。原task、空题面旧运行、固定source review和新版本全部保留。准入要求report中的完整curator profile与单独审阅catalog完全一致，原/新题面hash、实际task/run绑定均核对；该profile也进入稳定来源指纹。空题面或自报未审阅的新题面不能入池。

`export_prompts.py`只把原可见题面/提示/预算交给原build_agent_prompt。`../format_replays.py`只读eligible canonical，套固定官方模板和六工具声明，逐动作用add_generation_prompt=True；thinking_budget=0、无新增system prompt，空thought块与观察不进loss。标签、来源和baseline证据profile只作metadata。没有补写旧终止回答、thinking、策略概率或偏好对。

`../export_curricula.py`把同一主文件拆成G/S/F/R四份主责及shared_coldstart。共享复用相同任务，不增加独立数；未设训练混合权重或启动训练。路径和hash在课程清单中，大JSONL保留本地ignored artifacts。

```bash
competition/vfarl/replay/.venv/bin/python competition/vfarl/replay/normalize_results.py
competition/vfarl/replay/.venv/bin/python competition/vfarl/assemble_pilot.py
competition/vfarl/replay/.venv/bin/python competition/vfarl/export_curricula.py --manifest competition/vfarl/replay/gemma-sft-manifest.json --out competition/vfarl/replay/artifacts/adapter-curricula
```

原96剩余35题和具体来源/验收冲突见[完整说明](../pilot96/README.md)。未执行GPU、Gemma或额外teacher调用，没有模型on-policy轨迹和已验证关键动作对；人工CPU耗时不能当模型预算优化成绩。
