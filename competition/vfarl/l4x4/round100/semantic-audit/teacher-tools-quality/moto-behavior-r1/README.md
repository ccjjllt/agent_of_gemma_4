**导出器后续缺陷：** 此处真实工具／原oracle记录保留，但旧per-action SFT中的CRLF片段有错位，不能直接训练；修正行已另存[当前监督r2](../../../question-revision-r2/supervision-r2/README.md)。不能只按文件hash／token自一致就认定completion正确。

# Moto行为断言示范：两题真实重放

2026-10-28。固定100原索引8/12，`getmoto__moto-5200` 与 `getmoto__moto-6609`。这是Codex指定、审阅并通过实际官方六工具执行的新off-policy示范；无Gemma采样、logprob、生成thinking、GPU、外发或训练更新。旧recipe、trace、canonical、S600/F600及其manifest均保留，固定100身份没有改变。

两题各执行7个真实动作：get_status/read_file/run_command/edit_file/run_command/get_status/submit_patch。原语法检查命令被替换为来源绑定及真实API数值/集合断言：5200直接调用归档 `secret_arn`，后缀5→6；6609实例化归档 `Key(..., "SM2", ...)` 并读取 `to_dict()`，SigningAlgorithms恢复包含SM2DSA。两题均实际改前exit1、改后exit0；没有stub、AST替身或修改测试/依赖代码。

新完整官方验收在 `008-secret-arn/actual-run-r2` 与 `012-kms-sm2/actual-run-r2`：5200 baseline为1失败/84通过，reference为85通过；6609 baseline为3失败/113通过，reference为116通过。完整原文件范围与原required节点不变（分别1F2P+84P2P、3F2P+109P2P），严格规范化均eligible且reasons为空。这里的通过数量是验收范围，不是模型修复收益。

首次 `actual-run` 的真实六工具API观察成立，但完整官方验收分别因缺mock/freezegun收集失败，完整结果保留且不用于SFT。第二轮使用新独立 `/private/tmp/vfarl-moto-source-api-cache-r2`，未改第一次profile。纯Python依赖和真实dist-info均来自既有jiyujia缓存，mac原生/官方ADK依赖复用deepdiff任务的既有Python3.12缓存，没有下载或覆盖原环境。精确来源及文件哈希见 `runtime-profile-r2-source-manifest.json`。这是本机macOS CPU证据，未声明Kaggle chroot/比赛Docker等价。

原source snapshot在harness.setup之后产生，多出conftest.py/pytest.ini。这里仅按旧source-inventory清单恢复每个原文件字节、symlink及mode，并排除这两额外生成文件；新source_tree_sha逐字等于旧pin（5200: c8be9b4748ba2876cf13484ed870bce64ece365cf8e15879649a9159c2b3881b；6609: ce2a8e84c0b5e3210bb12a22d67a9dd6ce6fb1593d02a58231155c3fdecf5631）。两题保留旧30分钟/100tool/command300s/verification240s配置，新的initial_prompt字节SHA也逐字匹配旧真实prompt。题面里的CRLF原样保留；逐字比较使用bytes，避免read_text的换行转换。

规范化最初发现原acquisition blob已缺失，未绕过准入。随后只按source_review明确的train来源页与row_idx，从本地对应缓存行恢复8条moto原wrapper，序列化后逐字SHA全部等于原catalog/review pin才写原缺失路径；没有读取整页题面、heldout或用adapted task猜source。证据 `source-recovery-exact.json`。首次缺源失败与恢复经过在此保留，不作为新的来源版本。

独立canonical、真实指针raw report和Gemma SFT增量位于 `independent-sft-r2/`；没有把旧成功结果或token轨迹复制为新示范。所有准入仍使用未改动的normalize_results.normalize。

实际运行使用未改动的run_official.py，每题显式传入其source-binding.json中的task/workspace/recipe，参数为：`--verify --verify-baseline --source-layout . --task-site-packages /private/tmp/vfarl-moto-source-api-cache-r2 --time-minutes 30 --tool-calls 100 --command-timeout 300 --verification-timeout 240`。官方host的PYTHONPATH是旧deepdiff实际site-packages；任务source、API和deps绑定保存在本次run-summary。collector host设置了AWS metadata禁用标志；本次后续只读发现官方sandbox allowlist不会继承该变量。此处服务API的离线性来自实际Moto mock context；direct secret_arn/Key调用本身不发网络。
