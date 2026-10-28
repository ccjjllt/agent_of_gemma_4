# 当前SFT工具roundtrip发现的CRLF切片缺陷

2026-10-28。只读取当前fixed100监督与其实际messages，对current NativeArguments/parse_call作文本调用，没有执行示范中的任何工具／shell命令，没有模型、oracle或GPU运行，没有新测试文件。

旧supervision-r1 SHA `191bd117025f91094f3fa00eb1d029131974b5b9857d859093bc53206746e360` 含706条工具completion和99条终止completion。逐tool_call_id对照原source_episode JSON字符串解码后的arguments，58条工具completion不符；原件的completion、行SHA、source_episode及SHA、原工具和参数已冻结在 [actual-mismatches.json](actual-mismatches.json)，SHA `62e35250733c46175d1c8b9deb7617504897739e024c910d540ca287718ca8df`。不是parser误拒合法语法，因此没有放宽parser。

根因在format_replays.action_examples：render_episode在内存中按保留CRLF的字符串记录action_spans；后来Path.read_text默认通用换行，把CRLF转换LF，仍按旧字符跨度切片。真实moto5200全文UTF8解码为12222字元，默认read_text为12170，52字元之差都来自实际prompt的52个CRLF。原span6220:6261在原字元上正是完整get_status调用；归一化后切到tool_response中部，且该坏片段正是已写入SFT的completion。详见 [crlf-span-mechanism.json](crlf-span-mechanism.json)，SHA `f00b05228db85e1693bd24ea9a9da12faaa45c74670d95593c3481982f1c214a`。

终止文本也受影响，虽然parseNone仍会通过。moto5200真实回答以Submitted the production fix开头，旧completion却从robe now satisfies开始，见 [crlf-terminal-span.json](crlf-terminal-span.json)。所以此前单纯的token IDs／mask／prefix自一致并不足以证明completion就是原工具动作；不能把747个parse结果相等直接称作747条完整监督正确。

本轮只最小修复 [format_replays.py](/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4/competition/vfarl/format_replays.py) 的action_examples读取：先核rendered原始bytes SHA等于episode记录，再decode UTF8，保留原字元。没有改变span定义、模板、用户prompt CRLF、源messages或旧数据。bool/null及未知工具执行边界不涉及此修复。

实际仅调用一次真实moto5200的action_examples，8条返回到内存，没有写新SFT文件：全部completion等于原始raw span，7条工具与actual messages精确roundtrip，终止原文完整。前后tool／terminal例子及各source SHA见 [formatter-moto5200-after-fix.json](formatter-moto5200-after-fix.json)，SHA `1fd86c7e8547c1f9d219fb354892dc924fdb71a19f2ad69a364c051cda2196f9`。修复后源码SHA `5a9ce116d2c7835c854e17b3109812bec5050037dc865cff94bcd8b78bca04eb`，git diff --check通过。

受影响8个episode要全部重导66行（58工具＋8终止），不能只换58个工具片段；其余episode的原始行由root保留。精确source manifest、episode文件与行号、record line SHA、声明／actual-run decls、canonical、actual messages、rendered文件及SHA在 [affected-episode-sources.json](affected-episode-sources.json)，SHA `cc8d624c35ded569898814f7f6c1f08e6db49ee949851242991f83d1ed3f7abf`。Hydra的旧全局episode在replay/artifacts/gemma-sft/sft-episodes.jsonl第107行，原canonical仍匹配且严格准入，没有缺来源需猜补。

| ID | 全部行 | 工具行 | 终止行 | 对应episode文件行 |
| --- | ---: | ---: | ---: | ---: |
| facebookresearch__hydra.0f03eb60.pr_1915 | 8 | 7 | 1 | 107 |
| getmoto__moto-5200 | 8 | 7 | 1 | 1 |
| getmoto__moto-5348 | 9 | 8 | 1 | 1 |
| getmoto__moto-6159 | 8 | 7 | 1 | 2 |
| getmoto__moto-6492 | 8 | 7 | 1 | 3 |
| getmoto__moto-6609 | 8 | 7 | 1 | 2 |
| getmoto__moto-6755 | 9 | 8 | 1 | 4 |
| getmoto__moto-7111 | 8 | 7 | 1 | 6 |

root负责另版本重导和合成，旧supervision-r1保留为已知错误历史材料。本轮没有批量重写SFT，也没有将重新导出称为新真实teacher重放、RL奖励或模型收益。
