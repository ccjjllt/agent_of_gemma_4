# LoRA初始化差异的CPU机制证据（2026-10-27）

一次性手动CPU实验确认：shadow遗漏原NF4层的 `compute_dtype=bfloat16`，因此没有经历原生PEFT初始化中的BF16舍入。复制该属性后，410个目标层的适配器hash精确匹配旧原生运行的 `12b4e23d28fcc8ab2ea1581a9398edfd2105375e43f3fc795199f3d6b31b06dc`。

使用本机已存在的 torch 2.8.0、PEFT 0.21.2、Transformers 5.13.1 和 Python 3.11。由本地31B配置构造meta模型，只给shadow基座分配单元素共享存储；目标集合与旧原生训练清单均为410且完全相同。

| 实际变体 | trainable hash |
| --- | --- |
| 模型顺序，FP32 shadow，task_type=None | `ea31cef9bc93a0b63abe65aca89b5ebdc52d5e87cb0c59216c26b3689b1b0640` |
| 字典序，FP32 shadow，task_type=None | `8f6d4af930f3616dcac312626483d06ab56085bea377b5f070203adb55e7b19c` |
| 模型顺序，BF16 shadow，task_type=None | `12b4e23d28fcc8ab2ea1581a9398edfd2105375e43f3fc795199f3d6b31b06dc` |
| 模型顺序，FP32 shadow，task_type=CAUSAL_LM | `ea31cef9bc93a0b63abe65aca89b5ebdc52d5e87cb0c59216c26b3689b1b0640` |
| 模型顺序，FP32 shadow，compute_dtype=BF16 | `12b4e23d28fcc8ab2ea1581a9398edfd2105375e43f3fc795199f3d6b31b06dc` |

完整来源、hash定义和两条已经执行的手动Shell/Python命令保存在 [adapter-initialization-cpu.json](adapter-initialization-cpu.json)。记录文件时未重跑实验；它们是研究原件，不是持续验证脚本。没有读取任务正文或heldout内容，没有分配GPU。

这项证据仅解决初始化等价性，不代表GPU采集、RL更新、推理加速或泛化已经成功。
