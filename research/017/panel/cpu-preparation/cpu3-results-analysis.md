# CPU3 实际结果与下一轮输入

2026-10-24。作业 `jiyujia/p017-confirmation-cpu-repair-v3-d7d6753a` v1 已完成。14:12:52 北京时间回读为 RUNNING，14:16:41 回读为 COMPLETE；远端源码 SHA 与本地 `244b2632406b0e45eb08855263f46af06bcf8911e92ddb04c682ac111a29b233` 精确一致，GPU/TPU/Internet 均为 false。回读证据在 `research/017/execution/cpu3-carrier-readback/` 及 `confirmation-cpu3-current-*.json`。

小结果优先取回，随后取得全部 47 份日志及输出材料。材料 `cpu3-output/p017-confirmation-preparation/prepared-confirmation-material.bin` 为 **166,519,567 bytes**，实算 SHA256 `75859188c99bcdfaf09f0e987b49b09027cdeae916dc89abb056797ec3b6a2ed`，与 [summary](cpu3-output/p017-confirmation-preparation/summary.json) 一致。

**本轮 5 题全部实际执行，新增 2 题有效。** 原 11 个有效题没有重跑；跨两轮目前有 13 个不同有效题、7 个仓库，其中 11 题有 P2P 前后通过证据，2 题仍为原来的 F2P-only。这个数字不是新的模型修复率，也不表示原 24 题已全部准备成功。

| 本轮题目 | 实际前后结果 | 结论 |
|---|---|---|
| `hgrecco__pint-2089` | F2P exit 1→0，P2P 0→0 | 新增有效；两态真实 Linux collection 证实完整 node |
| `facelessuser__soupsieve-119` | F2P exit 1→0，P2P 0→0 | 新增有效；原源码/PYTHONPATH 导入成功，未执行旧 editable setup |
| `facelessuser__soupsieve-50` | 两态 F2P/P2P 均 exit 4 | 绕过 setup 后暴露生产源码 `from collections import Mapping` 与 Python3.12 不兼容；保留失败，不能添加源码兼容补丁冒称环境修复 |
| `hynek__structlog-586` | backend 支持检查成功，editable metadata-generation-failed | 尚未运行到判据；不再归因于 hatchling 不存在 |
| `facelessuser__soupsieve-263` | backend 支持检查成功，editable metadata-generation-failed | 尚未运行到判据；不预报补 backend 已修好 |

完整记录见 [prevalidation-results.json](cpu3-output/p017-confirmation-preparation/prevalidation-results.json)。Pint 的两个实际映射为：

```text
pint/testsuite/test_quantity.py::TestQuantity::test_unit_canonical_name_parsing[cm
→ pint/testsuite/test_quantity.py::TestQuantity::test_unit_canonical_name_parsing[cm centimeter cm centimeter-expected_unit1]

pint/testsuite/test_quantity.py::TestQuantityWithDefaultRegistry::test_unit_canonical_name_parsing[cm
→ pint/testsuite/test_quantity.py::TestQuantityWithDefaultRegistry::test_unit_canonical_name_parsing[cm centimeter cm centimeter-expected_unit1]
```

这些映射来自两态 collection 日志，不只是源码推导；完整 P2P 均通过，没有删节点或新增测试。

两份 metadata 失败的**完整原始日志**也没有内层 traceback，只显示 hook 返回 1 和 `metadata-generation-failed`，因此当前不能确定是版本推导、readme/hook 配置还是其他打包细节。没有据此继续盲目增加依赖。

已检查改为直接源码运行的可行性，并准备 [next-cpu-source-import-input.json](next-cpu-source-import-input.json)，供 root 与固定候补合并成下一次有界 CPU 作业：

- 保留 CPU3 真实导出的全部 requirement pins 和 pytest plugin entries，只将两题 `install_editable` 从 true 改为 false。
- 保持 `PYTHONPATH=<checkout>/src:<checkout>`、模块来源核对、独立 buggy/reference、原 reference/test patch 和全部原 F2P/P2P。
- soupsieve263 从源码 `__meta__` 读取版本；当前选中 attribute 判据及测试工具未依赖 distribution 元数据。
- structlog586 只有显式访问 `__version__` 等属性时才懒调用 distribution 元数据；当前选定的 `test_stdlib.py` 与 `conftest.py` 静态检查未使用这些属性。因此可以尝试直接源码导入，但不是成功保证；不伪造 dist-info/version。

该输入没有上传或执行。若导入或完整判据仍失败，应保存真实错误而不是修改待修仓库或测试。本轮环境材料和原 11 有效题继续可复用；两个已新增有效题也无需在下一轮重复准备。
