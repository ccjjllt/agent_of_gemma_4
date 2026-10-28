# 固定100新版题面的12条真实示范增量

2026-10-28。仅处理r2固定train索引25、26、29、31、32、36、39、41、43、44、46、49；42已有独立增强teacher，本轮未重复。十二题全部按原reviewed recipe实际执行新的官方六工具轨迹，原recipe没有修改；没有把旧prompt或旧messages重命名冒充新执行。

每题先按原canonical实际run的source-inventory逐文件恢复bytes与mode，原tree SHA完整匹配；没有复制archive内原清单之外的post-setup conftest.py/pytest.ini。原预算和harness数值类型原样传给未改动run_official.replay，实际初始prompt SHA逐字等于question-revision-r2。原required节点和目标文件均未放宽。每题独立deepcopy原pilot96／supplemental catalog单项，核对selection的task/review pins，再新增原文／curator文本及SHA的版本披露；全局catalog、旧题与旧teacher不变。

十二题全部获得真实baseline=False、repair=True，未修改的strict normalizer准入无缺项。每题独立canonical、official-prompts与gemma-sft保存在对应三位索引目录；完整来源与哈希见[teacher-increment-manifest.json](teacher-increment-manifest.json)。

|索引|原完整oracle用例|新增决策|监督tokens|最大context|
|---|---:|---:|---:|---:|
|25|124|8|900|3445|
|26|124|8|864|3374|
|29|124|8|946|3546|
|31|124|8|547|3260|
|32|1061|8|1138|3983|
|36|1061|8|530|3031|
|39|945|8|541|2742|
|41|204|8|940|3443|
|43|365|8|901|3498|
|44|280|8|627|3362|
|46|178|8|783|3374|
|49|178|8|565|2962|

合计12题、96条决策、9282个监督tokens，最大context 3983。推荐新增量为[gemma-sft/sft-actions.jsonl](gemma-sft/sft-actions.jsonl)，SHA `80cfcf9b78d5c678040f5a62bd70456add932521ddb25488896cd7e6a691ffab`；它只按索引顺序拼接已独立准入和导出的逐题数据，不改动作或观察。CT tokenizer/template两SHA保持既有pin，实际CPU格式运行版本保存在逐题manifest。thinking0；不含生成思考、模型行为logprob、Gemma rollout、GPU或训练更新。

Glom46/49的teacher-r1真实失败保留：目标生产修复已执行，但原CLI oracle因找不到glom命令各有1失败／177通过，未准入canonical/SFT。旧shared-deps已不含boltons/face，本轮只从既有CPU runtime archive读取其纯模块与metadata，逐文件成员和SHA保存在glom-dependency-source.json；再依据原setup.py的 `glom = glom.cli:console_main` 用现有sandbox_executable接口补正常console入口，在新的teacher-r2里重新实际完整重放，二者才取得178／178。没有改recipe、生产实现或oracle来绕过这个环境问题。初始化阶段两次依赖枚举失败也简短保留，未当作任务奖励。

这是明确指定并实际执行的human-reviewed off-policy示范材料。macOS CPU成功不等于Kaggle/Linux环境等价、模型独立修复收益或RL收益；没有网络下载、外发提交、预留题读取或新增测试文件。
