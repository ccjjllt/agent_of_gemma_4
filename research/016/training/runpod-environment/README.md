# P016 Runpod Python environment

2026-10-23（Asia/Shanghai），现有 Pod `0wietnrru530l9`。

隔离环境位于 `/workspace/p016/venv`，使用 `--system-site-packages` 复用镜像已有 CUDA 库；未修改全局 Python 包。所有安装均使用 `--no-cache-dir`，临时目录为 `/workspace/p016/tmp`。服务器安装原始日志位于 `/workspace/p016/setup/torch-install.log` 与 `transformers-install.log`。

固定训练栈为 torch 2.10.0+cu128、transformers 5.13.1、peft 0.21.2、bitsandbytes 0.50.2、accelerate 1.15.0、safetensors 0.8.0。为避免镜像已有 torch 2.8 的扩展与新环境冲突，环境内同时安装匹配的 torchvision 0.25.0+cu128 和 torchaudio 2.10.0+cu128。完整实际依赖见 `pip-freeze.txt`。

一次性人工命令确认全部训练依赖及 `Gemma4TextDecoderLayer` 可导入，`pip check` 无冲突，4 张 A40 可枚举、CUDA 可用，全部 GPU 对的 peer access 能力为 true。未运行训练、模型加载、性能基准或 NCCL collective；这些结果不代表训练吞吐与显存容量已验证。

设备实际显存每张 47,699,722,240 字节，compute capability 8.6；CUDA build 12.8、cuDNN 91002、NCCL 2.27.5。拓扑没有 NVLink，GPU 0/1 为 PXB，2/3 为 NODE，跨对为 SYS。cgroup 内存上限 200,000,000,000 字节。`/workspace` 与根目录共用 50 GiB 容器盘，环境完成时使用约 3.90 GB；该磁盘快照在模型下载完成前记录。

`runtime-imports.json`、`pip-check.txt`、`pip-freeze.txt` 和 `nvidia-topology.txt` 均为服务器实际命令输出。
