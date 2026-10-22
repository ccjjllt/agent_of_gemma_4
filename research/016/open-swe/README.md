# Open-SWE-Traces 限量候选调查

2026-10-22，CPU/公开只读网络。当前只有身份与来源证据，**没有取得轨迹正文，也没有可训练样本**；不让该候选阻塞已经有缓存的 SWE-smith 主线。

固定数据版本为 `nvidia/Open-SWE-Traces@1db674173119d71df45c22007af99e214600711a`。复用 P016 70 个历史 train 仓家族与 23 个非 train 排除家族，只投影 v1.2 两个 Qwen3.8-27B 分片的 instance_id、repo、license、language、trajectory_id、resolved。两个分片各 2,500 行，没有读取 messages/参考补丁列。

元数据中符合 Python、resolved=1、历史 train 家族与独立问题限制的候选为 OpenHands 31 条、mini-swe-agent 13 条；按仓轮询冻结为 [selection.json](selection.json)。这是指定教师的两个首分片，不代表全库分布。

首次按 trajectory_id 精确 `/filter` 获取正文，三次读取超时，未得到正文。随后核对数据服务转换分片与原文件 LFS SHA256 完全相同，只读前序分片页脚行数，得到精确全局行号：OpenHands 分片起点 12,541，mini-swe-agent 分片起点 53,553。映射和来源在 [row-offsets.json](row-offsets.json)。使用该行号的 `/rows` 请求仍返回 HTTP 500，仍未取得任何正文；没有扩大到全库下载，也没有把失败包装成数据产出。

`acquire.py` 是可复用的数据获取入口，只有冻结身份之后才允许获取正文；保存正文前核对身份、响应是否截断，前后核对固定版本。原 API 元数据、分片身份投影与页脚计数缓存在 ignored `research/016/data/open-swe/`。未来服务可用时可续取当前固定 44 条，不必重做调查或换题；本轮先完成 SWE-smith 的实际训练供给。
