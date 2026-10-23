# Runpod 4×A40 接入与下一次执行入口

2026-10-23。用户选择 Runpod 四张 A40，并要求按 https://agents.runpod.io/ 接入。OAuth与MCP读取已成功；实际为等待库存的自动部署预约，尚未创建可连接的服务器或开始训练。

## 本地接入事实

- 已读取 Runpod 官方接入指南。Node v24.19.0，Codex CLI 0.162.1。
- 指南命令 `npx -y skills add runpod/runpod-plugins-official --skill '*' --agent codex --yes --global` 成功；随后 `skills list --global` 确认 router、lane、journey 共 16 个技能可见。
- `codex plugin marketplace add https://github.com/runpod/runpod-plugins-official.git --json` 因 GitHub 443 连接超时失败。没有声明 marketplace 插件安装成功。
- 按指南的直接连接备用路径，`codex mcp add runpod --url https://mcp.runpod.io/` 已注册全局服务；`codex mcp get runpod` 确认 enabled、streamable_http、同一官方地址。
- OAuth 已完成，CLI 返回 `Successfully logged in.`。重开后本会话已加载Runpod工具，并成功执行真实API读取；先前的临时授权链接不需重复使用。
- 没有创建新 Pod、修改现有 Pod、生成 API key、上传数据或启动 GPU。OAuth 链接、token 和私钥不保存在本项目。

## 2026-10-23 00:31–00:34 北京时间：真实资源状态

`list_pods`及包含集群成员的第二次查询均返回空列表、无后续分页。随后只读查看用户Chrome中已有的 `console.runpod.io/pods` 页面，见自动部署预约 `odd_black_reindeer`，配置 **A40 x 4**，状态 **Waiting for availability**，预约标识 `cmvjqieng003akdn6uvcgz6ml`，创建时间2026-10-23 00:07、到期2026-10-24 00:37。该标识是页面的预约标识，不能冒充Pod ID。没有可用SSH地址。

MCP进一步读取 `NVIDIA A40`、`count=4`、`minCudaVersion=12.8`、`product=POD`，实际返回 `availability=NONE`，12.8/13.2均不可用。为缩短排队等待又只读比较Secure及Community同为48GB的四卡方案，返回A40、L40、L40S、RTX 6000 Ada、RTX A6000均为NONE；这只描述此次查询时和上述筛选条件下的库存，不是全平台或所有机器规格永久无货的结论。没有新建或改动任何付费资源，也未修改等位预约。

当前阻碍是四卡资源尚未分配，不是OAuth未登录。实例分配后先重新读Pod列表，再按下列顺序连接。不在本记录中声称已后台持续监控库存。

## 接入后直接推进

1. 用实际 MCP 工具列出现有 Pod，确认用户提供的四张 A40 实例；记录 Pod ID、状态、GPU、CPU 内存、挂载/磁盘、当前按时价格和可用连接信息。仅复用现有实例，不因接入自动创建资源。
2. 进入服务器后读实际 GPU/驱动、互联、剩余磁盘、Python 依赖和缓存。4×A40 容量、NF4/NCCL 和32K均待测。不要覆盖已有作业或缓存。
3. 上传已冻结 [v3校准包](../data/gpu-bundles/smith-pilot-calibration-v3.tar.gz)（6,127,400 bytes，SHA256 `9cf66d95c05eaf4ba866d57d1caa7d4083aa0bdce88143d6336cae5a0ebbed91`）及 [主语料v2](../data/gpu-bundles/smith-main-corpus-v2.tar.gz)（15,510,952 bytes，SHA256 `6488020fb35b5e933bb480e5f583bb3df4aa8642a95b3c8e5e52a76b22c97ef7`），只在服务器核实哈希后解包。
4. 训练缓存来源是私有 Kaggle `ccjjllt/p005-cpu-nf4-cache`。本地仅 manifest/config，约18.30GB完整权重需在服务器复用或传输，不能把本地小目录当完整权重。CT tokenizer已包含在校准包。
5. 执行 [gpu-run.md](gpu-run.md) 中的16K保存/中断/恢复，再根据实测进入32K。首轮所有 GPU 执行段合计最多2小时，下载与环境准备另记；不能每次重试重新获得2小时。计费状态与GPU脚本运行时长分开记录，脚本退出并不等于Pod停止计费。
6. 用真实速度估算677题、2,383,964监督token主训练时长后，再确定连续训练预算。六工具种子另见 [target-protocol](../target-protocol/README.md)：6题、70动作、9段，不并入原生三工具同一阶段。

尚无新增GPU结果或正式得分。当前最好正式分数仍为0.15。
