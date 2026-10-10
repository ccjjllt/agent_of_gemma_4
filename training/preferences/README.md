# preferences/ —— 第二轮偏好训练占位 (交接 §4 T3, 本轮不实现)

本轮只做 SFT。到出现同一消息前缀 + 同一源码状态下的真实 chosen/rejected 对后,
再实现 DPO, 共用 SFT 的编码 (`encoding/gemma_tool_chat.py`) 与导出
(`export_peft.py`), 新增 paired loss 即可。

候选范围 (均为候选超参, 非既定值):
- 300–1000 对
- DPO beta 0.05–0.2
- lr 5e-6–2e-5
- 0.5–1 epoch

配对硬规则 (训练报告 §5.4):
- rejected = 真实无效动作 (如旧串失配后重复原调用); chosen 必须是从同状态
  执行并证实有进展的动作或短后缀, 不能拼接另一条成功轨迹的未来。
- 只有都正确时才以耗时偏好短路径; 不同环境错误、两个都失败未知优劣、
  不同源文件状态的动作不强行配对。
- 优先错误回包后的决策, 而非整条失败轨迹。
- 在线 RL 单独分支, 不阻塞首个 SFT 版本。

状态: IMPLEMENTED? No — PLACEHOLDER ONLY, NOT YET IMPLEMENTED.
