# 四题 CPU R7：已导出可复用运行时

2026-10-28。唯一实际提交 `jiyujia/vfarl-current-four-cpu-runtime-export-r7/1` 于03:15:53 UTC返回成功，03:24:27 UTC实际状态COMPLETE。冻结源SHA `d29c8e665baf629de5f1765503ba608fafc531ab83621e0666ed8a00e193f78e`，私有／GPU关闭，零模型调用和optimizer更新。

发布产物为 `vfarl-cpu-stage.bin`：

- 实际大小 **925,108,005字节**（约925.1MB／882.3MiB）。
- 生产者实际SHA256：`f13237d48dd0adecb3556334106836002290efa2b02987061f17b606771be44c`。
- 原运行时基底SHA：`0a8e277010c63701b1a163869bd6998732c18307770d7297a5cea524391879a0`。
- 既有消费入口：runtime-kernel使用上述已完成 `/1`，runtime-sha256使用新SHA。没有改变默认运行时或课程。

[归档原始记录](outputs/vfarl-prevalidate/runtime-archive.json)还绑定本次四题准备范围及原验收摘要SHA。平台列出了该文件；HEAD方法返回404，随后同一个官方文件URL的有界GET返回206、`Content-Range: bytes 0-0/925108005`，仅读取1字节，证明发布大小与记录相同。未下载完整归档；新消费时仍由现有runner在解包前实际重算整个文件SHA，不能把本地未下载说成独立重哈希。见[发布读回](published-runtime-readback.json)。

| 当前题目 | baseline | 原已审修复 | 奖励 |
|---|---:|---:|---:|
| Moto5348（9） | 1失败、6通过 | 7通过 | 0 → 1 |
| Moto6159（10） | 1失败、68通过 | 69通过 | 0 → 1 |
| DVC7628（23） | 31失败、3通过 | 34通过 | 0 → 1 |
| Markdownify161（54） | 1失败、74通过 | 75通过 | 0 → 1 |

[本次聚合原件](outputs/vfarl-prevalidate/prevalidation.jsonl)SHA `8d5e56a4b20be4851a4f84aceb515c4731cfdf698a628dcab45bbb4b7d58c1bd`；四行usable=true，八端verification_complete=true。实际task／snapshot／patch、patch字节、public prompt渲染及八XML均已逐项绑定，54使用当前澄清题面。四个runtime config指纹与独立CPU6完全相同，见[实际结果摘要](actual-results-summary.json)。原历史null不改，不能称模型收益或100题新验收。

只安装四题对应的stock-moto／stock-dvc／stock-markdownify，实际55／115／8包；DVC按原声明排除不使用的rerun插件。准备期按原锁定版本访问PyPI，验收期恢复UV_OFFLINE=1。新的prepare-only-selected-tasks显式选项默认false，本次为true，其他原profile保留基底状态。归档仍走原archive.add(stage)，新题目/源码材料位于独立material目录，不加入新源码forest；没有新增overlay或消费协议。

消费边界：当前helper匹配task版本／snapshot／patch／isolation／bytecode，外层另核真实依赖profile及metadata；不能说它统一比较了完整runtime.json哈希，也不能仅凭路径/元数据hash差异认定发生错误复用。未来消费本产物还需实际确认适用环境及题目，尚未进行新GPU采样。

[小输出下载清单](outputs/download-manifest.json)含52项、860,275字节。API请求CPU时限3600秒，但实际时限回读未暴露；本次已真实结束。

此前第一次提交请求被自动审批拒绝，命令未启动，也未产生mutation receipt或远端作业；原因是新增大型runtime持久保存授权不够明确。用户随后明确允许“私有导出运行时，供后续复用”，之后沿同一路径对同一冻结源完成唯一实际提交。该历史拒绝已经解除，未绕过审批。

## 复用参数与当前材料缺口

2026-10-28 再次只读核对维护中的 `competition/vfarl/l4x4/kaggle/build_runner.py` 和 `runner.py`。现有入口可以消费本 R7 运行时，无需 overlay 或新 CLI。将下面**参数片段**并入下一次已审载体的冻结命令，工作目录为仓库根目录；继续显式提供该载体自己的 `--material-build`、与其归档 SHA 匹配的 `--dataset-source`、实验模式及新 `--out`。本次没有执行 builder、上传、CPU/GPU 验证或训练。

```sh
--kernel-owner jiyujia \
--runtime-kernel jiyujia/vfarl-current-four-cpu-runtime-export-r7/1 \
--runtime-sha256 f13237d48dd0adecb3556334106836002290efa2b02987061f17b606771be44c \
--prevalidation-report competition/vfarl/l4x4/round100/prevalidation.jsonl \
--prevalidation-report competition/vfarl/l4x4/round100/cpu-runtime-repair/execution-r7-20261028/outputs/vfarl-prevalidate/prevalidation.jsonl
```

前一报告是原有 100 行聚合 CPU 记录，SHA `9c1c24dd2fe72849ca5233ada078af2cfdb8776f5b1c0bb293773625adaa67cc`；后一报告是本 R7 的 4 行原件，SHA `8d5e56a4b20be4851a4f84aceb515c4731cfdf698a628dcab45bbb4b7d58c1bd`。builder 将两份真实文本独立冻结进 SOURCE_BUNDLE，并把报告 SHA 写入 carrier manifest。runner 优先使用明确冻结的报告；有相同 task 的记录时优先当前完整 task 匹配，其次经旧原件证明的纯题面兼容，同级才由后报告覆盖。这样保存历史原件并让四题使用 R7 记录。

不要只传 R7 的四行报告给完整 100 题 selection：runner 在处理 `--only-task` 的可采样范围之前，仍要求所有冻结 task ID 都有报告行。`--prevalidation-report` 是后续 GPU 载体的验收复用入口；CPU `--mode prevalidate` 会执行真实验收后返回，不会靠这两个参数跳过验收。本片段不指定训练模式，不代表启动正式训练的授权。

`--runtime-kernel` 将精确 `/1` 加入 kernel_sources；runner 找到唯一 `vfarl-cpu-stage.bin` 后先实际重算整个归档 SHA，再解包并依据 preparation 中的真实 profile/package 记录绑定任务环境。不要同时挂载旧运行时 kernel 或 `--runtime-dataset`，以免出现两个同名归档；模型/NF4/adapter输入继续沿各自原有独立绑定。本次新归档 SHA `f132…` 与旧基底 `0a8e2770…` 是两个不同产物，旧基底的可用性不能由新四题结果追认。

**当前 R3 题面兼容还需要 host-only 旧 task 原件。**只读对比 `question-revision-r3/selection.json`：原 100 行报告中有 39 个 task SHA 与新题面不同；R7 四行 task SHA 都与当前 R3 完全一致，因此这四行本身不依赖旧题面证明。加入 R7 后，Markdownify54 改用直接当前证据，其余仍有 38 行需要 statement-only 兼容。现有 builder 会按原 revision 引用自然打包全部 39 份真实旧对象（约 2.89MB），无需为少一行另建格式。

这些文件应位于下一材料包的 `vfarl-input/provenance/cpu-verification-tasks/<旧CPU行task_input_sha256>.json`，由 `kaggle/build.py::package_question_provenance` 从现存旧 adapted-task 原件复制，并由 material manifest 哈希覆盖。消费 helper 会再次核对 bytes SHA，且仅允许 old/current 完整对象除 `problem_statement` 外全部相等；snapshot、patch、isolation、bytecode 和真实环境适用性继续分别判断。这些原件不进入模型题面或源码工作区，不能拿 revision 元数据的声明代替。

初次审阅时没有现成 R3 `material-build.json`，旧 R2 材料也没有 `cpu-verification-tasks` 原件；这是当时观察，已被下述本地构建补齐。**只重建 carrier 或增加 `--prevalidation-report` 不会补入旧材料归档缺失的原件**，因此主任务随后使用当前材料 builder 生成了实际 R3 归档。

2026-10-28 05:51:30 UTC 的[本地消费原件](../../question-revision-r3/material-local-r1/consumption-observation.json)（SHA `2a8344475b0eb0a00ef518b859a5a7f17e42b9068ba94cf77722d0e04e94ffae`）记录：新归档 **125,224,607 字节**，SHA `7bb34f0aa992edb8ee09b89d81db1581e2089597e246abafac57ec9bf6de4468`；归档内 39 份旧 task 共 2,891,406 字节，已逐字读回并核对。实际 runner 报告选择片段与打包 helper 消费上述两份 CPU 报告，得到 **62 条完整当前 task 匹配、38 条纯题面兼容**，R7 四行原样选中。该记录明确没有环境准备、oracle/model 执行或新 Linux 验收。

下一载体的本地材料参数已有确定文件：

```sh
--material-build competition/vfarl/l4x4/round100/question-revision-r3/material-local-r1/material-build.json
```

[材料构建记录](../../question-revision-r3/material-local-r1/material-build.json) SHA `dd2e26f692bc424f94cabaf7adc7ce9075ca1c93cfd7467c25e0f683d0ce1e54`。其本地 dataset metadata 标识为 `jiyujia/vfarl-current100-r3`，**尚未上传、没有远端 version**；所以现在不能给出一个真实可挂载的 `--dataset-source owner/dataset/version`，也不能拿旧 dataset 版本替代此新归档。运行时 R7 和 R3 本地材料均已准备好，剩余远端输入绑定需在未来实际发布后取得版本。此次文档更新只读上述结果，没有重建、重验或上传。

以上是实际源码消费审阅，不是新的 Linux 验收：R7 新实证范围仍仅四题；其它题是否可用由各自历史证据、当前任务语义和真实 profile 共同决定，不能称为 100 题已经在新 R7 运行时全部通过。
