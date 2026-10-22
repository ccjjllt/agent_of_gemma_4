# Runpod 4×A40 接入与下一次执行入口

2026-10-23。用户确认现有服务器来自 Runpod，配置为四张 A40，并要求按 https://agents.runpod.io/ 接入。OAuth登录已成功；尚未连接服务器或开始训练。

## 本地接入事实

- 已读取 Runpod 官方接入指南。Node v24.19.0，Codex CLI 0.162.1。
- 指南命令 `npx -y skills add runpod/runpod-plugins-official --skill '*' --agent codex --yes --global` 成功；随后 `skills list --global` 确认 router、lane、journey 共 16 个技能可见。
- `codex plugin marketplace add https://github.com/runpod/runpod-plugins-official.git --json` 因 GitHub 443 连接超时失败。没有声明 marketplace 插件安装成功。
- 按指南的直接连接备用路径，`codex mcp add runpod --url https://mcp.runpod.io/` 已注册全局服务；`codex mcp get runpod` 确认 enabled、streamable_http、同一官方地址。
- OAuth 已完成，CLI 返回 `Successfully logged in.`。目前尚未以 Runpod API 读取任何 Pod。当前会话未出现 Runpod 工具；官方指南要求 Codex Desktop 完全退出再打开以加载。先前的临时授权链接不需重复使用。
- 没有创建新 Pod、修改现有 Pod、生成 API key、上传数据或启动 GPU。OAuth 链接、token 和私钥不保存在本项目。

## 接入后直接推进

1. 用实际 MCP 工具列出现有 Pod，确认用户提供的四张 A40 实例；记录 Pod ID、状态、GPU、CPU 内存、挂载/磁盘、当前按时价格和可用连接信息。仅复用现有实例，不因接入自动创建资源。
2. 进入服务器后读实际 GPU/驱动、互联、剩余磁盘、Python 依赖和缓存。4×A40 容量、NF4/NCCL 和32K均待测。不要覆盖已有作业或缓存。
3. 上传已冻结 [v3校准包](../data/gpu-bundles/smith-pilot-calibration-v3.tar.gz)（6,127,400 bytes，SHA256 `9cf66d95c05eaf4ba866d57d1caa7d4083aa0bdce88143d6336cae5a0ebbed91`）及 [主语料v2](../data/gpu-bundles/smith-main-corpus-v2.tar.gz)（15,510,952 bytes，SHA256 `6488020fb35b5e933bb480e5f583bb3df4aa8642a95b3c8e5e52a76b22c97ef7`），只在服务器核实哈希后解包。
4. 训练缓存来源是私有 Kaggle `ccjjllt/p005-cpu-nf4-cache`。本地仅 manifest/config，约18.30GB完整权重需在服务器复用或传输，不能把本地小目录当完整权重。CT tokenizer已包含在校准包。
5. 执行 [gpu-run.md](gpu-run.md) 中的16K保存/中断/恢复，再根据实测进入32K。首轮所有 GPU 执行段合计最多2小时，下载与环境准备另记；不能每次重试重新获得2小时。计费状态与GPU脚本运行时长分开记录，脚本退出并不等于Pod停止计费。
6. 用真实速度估算677题、2,383,964监督token主训练时长后，再确定连续训练预算。六工具种子另见 [target-protocol](../target-protocol/README.md)：6题、70动作、9段，不并入原生三工具同一阶段。

尚无新增GPU结果或正式得分。当前最好正式分数仍为0.15。
