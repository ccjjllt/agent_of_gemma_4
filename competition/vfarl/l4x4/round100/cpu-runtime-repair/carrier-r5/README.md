# CPU r5候选：修复实际uv可执行路径，尚未提交

本目录派生自真实失败的[CPU r4](../execution-r4-20261028/README.md)，只新增`ensure_cpu_uv`及其调用；四题/current题面投影、legacy-default、原snapshot／patch／oracle与32个SOURCE_BUNDLE文件不变。具体代码见[source-diff.md](source-diff.md)。候选run SHA`4c945f17c83961b45e03e2054d6ea557db54e1399f9a2e66043eb83489b52f2a`；metadata SHA`2390aeded4eb08fe5e7fe2fbf651a067c25d88156ec1fdf2317f299020136f39`。独立新slug为`jiyujia/vfarl-current-four-cpu-runtime-repair-r5`，private、GPU/TPU关闭、models=[]。

helper先实际记录旧期望路径的lstat、size、mode、symlink、resolved path与原文件SHA，再检查可执行性和15秒内的`--version`。同一坏resolved path只尝试一次；不是native Linux ELF的wrapper不执行／不复制。若packaged uv不可用，允许复用同pinned base image PATH中实际可运行的native uv，并将其复制为期望路径上的普通文件，记录realpath／SHA／实际版本。若无可用PATH uv，只允许读取本输入`prep.runtime_wheels`已声明且SHA匹配的既有uv wheel，取唯一native uv成员恢复；不下载新binary或包。恢复后再次实际执行`--version`，失败就保留真实失败，不进入oracle。

原安装命令的包版本及CLI flags保持；额外设置`UV_OFFLINE=1`并由原子进程继承，禁止`--find-links`不足时偷偷访问index。任何cache缺项保持真实失败，不转联网补包。新helper实际执行时写`uv-runtime-repair.json`及phase日志；目前没有在Linux执行，不能预称修复成功。

复用原private material `jiyujia/vfarl-l4x4-train100-v4/1`及runtime `jiyujia/vfarl-newtasks-unsloth-r6/1`，不改GPU r4依赖。仍只执行索引9/10/23/54，排除Marshmallow；54仅当前statement版本变，parent selection7a69与四题子selection／questions哈希沿CPU r4精确绑定。验收仍为原scope、command90秒／verification300秒，timeout仅计划请求3600秒，不称已获平台保证。没有模型／optimizer，也不修改历史null、R6或课程R7。
