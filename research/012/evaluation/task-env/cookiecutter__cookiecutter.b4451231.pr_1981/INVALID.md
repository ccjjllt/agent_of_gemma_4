# 无效任务说明（不进入 48 次主结果）

cookiecutter pr_1981 的轨迹唯一生产修改是注释 + docstring（零行为变更）；
buggy 代码本来就对非 dict 抛 TypeError，上游 resolved=true 是在该注释级修改下通过的，
故隐藏测试在纯 buggy 代码上同样通过，不满足“修前失败与参考修复通过的区分”。
已按同仓哈希顺位替补为 conan pr_14852（见 dev12-substitution.json）。
本目录仅保留 provenance + source 快照，不设 reference.patch，不编入 tasks-selected.jsonl。
