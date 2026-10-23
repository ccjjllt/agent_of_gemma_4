# Runpod 4×A40 接入与下一次执行入口

> 2026-10-23 11:23:32 北京时间更新：537题正式主训练已启动，68次更新、最多8小时，见 [执行记录](main-short-run-plan.md)。用户最新要求保持Pod运行、不使用定时任务；完成后由用户通知再检查结果。下方较早“正在下载/待校准”记录仅保留过程，缓存和16K校准现已完成。

2026-10-23。用户选择 Runpod 四张 A40，并要求按 https://agents.runpod.io/ 接入。当前已分配Pod、获得明确公钥授权并通过直连SSH；隔离环境、完整NF4缓存及训练包已就绪，v3已完成step 1并提交checkpoint，v4从该点恢复后又完成step 2。当前正在继续16K桶，32K及主训练尚未完成。下方早先等待库存记录仅为历史。

## 2026-10-23 09:15后：实例、SSH与环境

实际Pod `0wietnrru530l9` / `odd_black_reindeer`，EU-SE-1、RUNNING；镜像 `runpod/pytorch:1.4.1-cu1281-torch280-ubuntu2404`，API实际费率2.36美元/小时（实例运行费率，不是已发生总账单）。直连为 `ssh -i ~/.ssh/id_ed25519_ccjjllt -p 22078 root@194.68.245.193`。首次连接因缺密钥失败；用户明确回复“允许”后，重新读到账户密钥为空，添加现有ccjjllt公钥并读回确认；通过Runpod SSH代理进入当前Pod，保留authorized_keys原内容并追加同一公钥，直连随后成功。此前账户级修改的自动审批拒绝没有改变账户；明确授权后的操作已成功。

实读四张A40每张46,068MiB、初始均空闲，driver595.91.07，cgroup RAM上限200GB，无NVLink。50GiB容器盘未挂持久卷，不能称checkpoint已有独立持久存储；本轮校准结束及中断点需将实际产物回传本地。本阶段未执行Pod stop/restart/terminate；任何后续资源操作前先保存需要保留的结果。

已上传并核对v3校准包及主语料v2的SHA，解包至 `/workspace/p016/`；恢复问题修复后另冻结并使用v4，不改动v3历史。固定训练环境 `/workspace/p016/venv/bin/python` 已安装并导入成功，未修改base；准确版本、拓扑与安装证据见 [runpod-environment](runpod-environment/README.md)。模型下载入口 [download_runpod_cache.py](download_runpod_cache.py) 从本地SDK获取签名URL、经SSH stdin送至远端流式下载；账户token留在本地，15个载荷文件逐个对固定SHA/size，源manifest本身也精确核对。首次SSH断连后的补传已完成；最终日志 `research/016/data/runpod-20261023/cache-download-retry1.log`记录15个载荷文件共18,295,563,073 bytes、逐文件SHA/size匹配，源manifest SHA256为`3a9661d56f0eec51474741b01d7ddb2932fb4836d7f776051d8ece7c333750cf`。缓存路径为`/workspace/models/p005-cpu-nf4-cache/qat-nf4`。

## 本地接入事实

- 已读取 Runpod 官方接入指南。Node v24.19.0，Codex CLI 0.162.1。
- 指南命令 `npx -y skills add runpod/runpod-plugins-official --skill '*' --agent codex --yes --global` 成功；随后 `skills list --global` 确认 router、lane、journey 共 16 个技能可见。
- `codex plugin marketplace add https://github.com/runpod/runpod-plugins-official.git --json` 因 GitHub 443 连接超时失败。没有声明 marketplace 插件安装成功。
- 按指南的直接连接备用路径，`codex mcp add runpod --url https://mcp.runpod.io/` 已注册全局服务；`codex mcp get runpod` 确认 enabled、streamable_http、同一官方地址。
- OAuth 已完成，CLI 返回 `Successfully logged in.`。重开后本会话已加载Runpod工具，并成功执行真实API读取；先前的临时授权链接不需重复使用。
- OAuth接入阶段未创建Pod或启动GPU；实例后来由用户已有预约分配，本轮已在明确授权后配置SSH、上传训练包并启动校准。没有新建额外Pod或生成API key。OAuth链接、token和私钥不保存在本项目。

## 2026-10-23 00:31–00:34 北京时间：真实资源状态

`list_pods`及包含集群成员的第二次查询均返回空列表、无后续分页。随后只读查看用户Chrome中已有的 `console.runpod.io/pods` 页面，见自动部署预约 `odd_black_reindeer`，配置 **A40 x 4**，状态 **Waiting for availability**，预约标识 `cmvjqieng003akdn6uvcgz6ml`，创建时间2026-10-23 00:07、到期2026-10-24 00:37。该标识是页面的预约标识，不能冒充Pod ID。没有可用SSH地址。

MCP进一步读取 `NVIDIA A40`、`count=4`、`minCudaVersion=12.8`、`product=POD`，实际返回 `availability=NONE`，12.8/13.2均不可用。为缩短排队等待又只读比较Secure及Community同为48GB的四卡方案，返回A40、L40、L40S、RTX 6000 Ada、RTX A6000均为NONE；这只描述此次查询时和上述筛选条件下的库存，不是全平台或所有机器规格永久无货的结论。没有新建或改动任何付费资源，也未修改等位预约。

以上仅是00:31–00:34的历史快照；实例随后已分配并连接，不再把库存或OAuth当当前阻碍。没有将此记录写成后台持续监控承诺。

## 当前执行证据与继续入口

1. **资源和环境已完成。** 复用现有Pod `0wietnrru530l9`，四卡、SSH、Python栈及完整NF4缓存已确认。运行时必须额外传`NCCL_P2P_DISABLE=1`：原始v3首次NCCL collective阻塞156.585秒后停止；加此变量后完成首个更新。不能用peer-access能力查询代替真实collective结果。
2. **已完成真实step 1。** v3训练步138.326640秒、46,005 input / 8,070监督tokens，完整checkpoint 1已提交，控制退出码75。v3恢复随后耗时117.243505秒，在FSDP包装前因adapter跨rank不一致失败，没有新增更新。具体记录见 [runpod-calibration](runpod-calibration/README.md)。
3. **恢复修复已有新增更新。** 两段诊断证明裸`cuda`反序列化落到GPU0，GPU1有300个LoRA_B未正确恢复；v4显式以CPU加载adapter，再复制到当前rank模型。v4从checkpoint 1通过实际恢复，并完成step 2、提交checkpoint 2：185.115960秒、54,371 input / 7,615监督tokens，loss 0.629700261。已进入step 3 forward，包含最长15,768；后者尚不能当作最长样本更新完成。
4. **继续使用v4。** [v4构建回执](generated/smith-pilot-calibration-v4/build-manifest.json)记录包6,127,456 bytes、SHA256 `6c9cc2dc84adf09af00643d7ed7a2f44afdc1a5617548c855ddb4215da4e0b2c`，目录`/workspace/p016/smith-pilot-calibration-v4`。实际恢复命令及32K入口在 [gpu-run.md](gpu-run.md)。目前有执行段运行，先读取它的结果，不另起四卡作业。若以后恢复，选择最新完整checkpoint，不回退到000001。
5. **统一预算。** 首轮所有GPU执行段累计最多7,200秒，下载及环境准备另记。v4开始前已消耗504.195596秒，含三次启动与两段GPU诊断；v4的完整运行墙钟还需追加，不能只累计step时间。Pod运行费率与训练脚本时间分开，脚本退出不等于Pod停止计费。
6. **后续研究顺序。** 完整16K桶与恢复结果确认后，再用剩余预算单独测32K；根据两桶真实吞吐、checkpoint与加载开销估算677题、2,383,964监督tokens的主训练时长，再确定连续训练预算。主语料v2已上传，SHA256为`6488020fb35b5e933bb480e5f583bb3df4aa8642a95b3c8e5e52a76b22c97ef7`。六工具种子另见 [target-protocol](../target-protocol/README.md)：6题、70动作、9段，不并入原生三工具同一阶段。

本地已回传checkpoint 1的完整标记及训练证据；后续完整checkpoint/adapter需继续回传，容器盘不是独立持久卷。v3包及失败证据均保留。当前最好正式分数仍为0.15，校准更新不等同于修复收益。
