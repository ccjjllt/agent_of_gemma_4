# 固定100题的实际依赖环境修正

2026-10-27。只修复运行环境，不换题、改原题面或改变原官方完整验收要求。

2026-10-28最新：[R7已验证并私有导出运行时](execution-r7-20261028/README.md)，四题全部有效baseline0／repair1；原历史null保留。产物为`jiyujia/vfarl-current-four-cpu-runtime-export-r7/1`的`vfarl-cpu-stage.bin`，实际925,108,005字节、SHA256 `f13237d48dd0adecb3556334106836002290efa2b02987061f17b606771be44c`，基底仍为0a8…；只准备四题所需的三个stock环境，其他题不宣称获得新验收。用户已明确允许这项私有持久化导出。没有模型权重或训练，当前课程／默认运行时尚未自动变更。

此前[CPU5四题记录](execution-r5-20261028/README.md)中，4题全部`usable=false`，baseline／repair共8次均`verification_complete=false`、`reward=null`。UV虽恢复，离线包cache仍不完整，旧prevalidate沿用原runtime而缺sure／colorama／bs4。这不是有效0/1，其864,775,947字节旧导出包不能作为已修复版本。用户随后确认继续新一轮额度，才产生下述R6／R7真正修复结果。

维护中run.py已补上真实环境消费修复，SHA `8759790a7644572957f8f1ca20f62d158f0f72feaec8b7221299fff81505d608`：TaskEnvironment在创建沙箱前拒绝明确失败的准备记录，prevalidate保留原环境错误，不运行旧python／site-packages，不生成baseline或repair奖励。未声明Linux准备状态的现有本地环境照常使用；原`_verify`函数未改。四个CPU5失败声明的实际拒绝、一次prevalidate消费和可用Tenacity公开API控制见[手动证据](runtime-consumption-repair-20261028/result.json)。这些是入口行为证据，还不等于四题Linux依赖已修好。

后续[R6实际完成](execution-r6-20261028/README.md)：UTC02:55单次提交`jiyujia/vfarl-current-four-cpu-runtime-repair-r6/1`，源SHA `16daf7dbf89cccefacb4c514618b0a548f869c8a52aa69a5381fb6cd4c3b04d2`。它冻结上述新run.py，仅为四题构建三个stock profile；准备期从PyPI获取原来锁定、已有历史Linux安装证据的包版本，准备结束后恢复UV_OFFLINE=1。四题真实chroot全部baseline0／repair1、8次verification_complete=true、LLM calls=0。任务／source／原patch／公开prompt均一致，54是当前新版题面。没有改依赖版本或原验收、没有GPU、模型及runtime大包导出。当前采样仍挂旧0a8运行时，所以课程暂停不能仅凭本次CPU成功解除；实际51项输出中没有可复用profile产物，下一步须将同一成功环境来源打包并接入消费。

R7沿用上述成功实现并启用`--prepare-only-selected-tasks --export-runtime`，在四题再次0／1后用既有stage archive格式保存基底和依赖。四题runtime配置hash均与R6相同。新增选项默认关闭，只允许显式CPU prevalidate任务；不引入新的overlay格式或消费框架。实际发布文件的1字节Range读回确认总大小925,108,005字节，本地没有下载或独立重哈希整包；未来runner仍须按上述实际producer SHA验证后解包。现有CPU证据复用依据task版本／isolation／bytecode及实际依赖profile判别，不能称已有完整runtime配置hash等价校验；路径或准备元数据不同也不能直接当作环境语义缺陷。

R3的四条评分无效轨迹已有明确环境原因：`getmoto__moto-5348`、`getmoto__moto-6159`的完整验收缺少`sure`；DVC `pr_7628`缺少`colorama`；markdownify `pr_161`缺少`bs4`。原`original-result.json`和`original-official-verification.json`均保留，历史奖励继续为null，不能补0。

根因不是这些任务没有历史CPU通过记录。`next-round/runtime-rebuild-v4-evidence/vfarl-preparation.json`里的`profile-00`仍把18个stock任务绑定到仅有pytest、iniconfig、packaging、pluggy的Python 3.12环境；训练载体复用该原始运行时与跨批CPU证据时，没有调用CPU专用的`prepare_cpu_repairs`。正确的moto/DVC/markdownify依赖清单原本已经存在于`kaggle/stock-environment-pins.json`，却没有用于这些实际采样环境。

维护中的runner现在根据固定任务的显式依赖清单选择运行时，从实际site-packages的dist-info读取包版本并拒绝错误stock回退。完全等价的现有profile可以复用，例如pytest-only环境仍可服务dominate；不会仅凭profile名字将有效环境排除。只暂停实际环境不可用的任务，其他真实可用任务继续，历史CPU结果和null奖励不变。实际绑定及未就绪原因随原`carrier.json`保存。

## 历史五题CPU载体r3

当时冻结的载体为`carrier-r3/`，kernel计划为`jiyujia/vfarl-train100-cpu-runtime-repair-r3`，源文件SHA256为`2cadd176d90ea8703b5ea0b9d9c54f4b4b3041ecade0c348b32fdfb231aebbae`。r1／r2／r3均为历史未提交构建；实际后续只执行四个current task的r4及r5，不将下面旧五题计划当成执行结果。

仅挂载既有私有材料`jiyujia/vfarl-l4x4-train100-v4/1`和`jiyujia/vfarl-newtasks-unsloth-r6/1`。原任务材料SHA256为`4b3792470dd1dacc73e1899ba38a3d96f2054bfad916d4a9b51b7e4db4333504`；输入运行时SHA256为`0a8e277010c63701b1a163869bd6998732c18307770d7297a5cea524391879a0`。GPU关闭，不读取模型权重或采样，不执行optimizer。

对四道unknown任务和固定100内的marshmallow `nqk1hyjp`共五题执行冷环境原始版本与已审修复的完整官方验收，要求分别取得有效0/1证据。第五题用于核对下文已证实的非stock漂移。为导出可供固定100题复用的运行时，准备或复用固定100内实际引用的依赖profile，并展开该冻结材料中必要的源码元数据；不会对其余题执行验收或读取预留4题正文。`--export-runtime`导出`vfarl-cpu-stage.bin`及哈希，不只导出这五题的局部环境。

构建命令：

```bash
python3 competition/vfarl/l4x4/kaggle/build_runner.py \
  --material-build competition/vfarl/l4x4/kaggle/artifacts/material-v4/material-build.json \
  --mode prevalidate \
  --out competition/vfarl/l4x4/round100/cpu-runtime-repair/carrier-r3 \
  --runtime-kernel jiyujia/vfarl-newtasks-unsloth-r6/1 \
  --runtime-sha256 0a8e277010c63701b1a163869bd6998732c18307770d7297a5cea524391879a0 \
  --only-task getmoto__moto-5348 \
  --only-task getmoto__moto-6159 \
  --only-task iterative__dvc.1d6ea681.pr_7628 \
  --only-task matthewwithanm__python-markdownify.6258f5c3.pr_161 \
  --only-task marshmallow-code__marshmallow.9716fc62.lm_rewrite__nqk1hyjp \
  --kernel-owner jiyujia \
  --dataset-source jiyujia/vfarl-l4x4-train100-v4/1 \
  --kernel-id jiyujia/vfarl-train100-cpu-runtime-repair-r3 \
  --title 'VFARL fixed100 CPU runtime repair R3' \
  --export-runtime
```

上述输出目录已经存在，重建必须另选新目录并保留真实来源。实际提交须通过显式jiyujia专属凭据的launcher；机器默认CLI属于另一账号，不能使用未指定账号的裸CLI提交。此历史r3不是当前执行候选。

本地已用一次性Python命令核对载体语法、R3只含上述五题的执行清单和GPU关闭；从真实raw preparation重建pytest-only profile的包元数据，确认dominate仍可绑定，而四道unknown均被拒绝错误回退；补入声明依赖元数据后可绑定到stock-moto、stock-dvc、stock-markdownify。这只核对绑定逻辑，不等于真实Linux导入、完整oracle或修复收益。没有新增测试文件。实际CPU执行和导出结果以随后回收的原始日志为准。

2026-10-28早期：远端CPU提交曾因具体payload外发授权不够明确被自动审批拒绝。用户随后明确同意私有CPU及短GPU验证；按显式jiyujia凭据执行的r4在验收前缺UV失败，r5实际结果见开头。原拒绝已不是当前阻断条件，旧载体和失败记录保留。

## 非stock漂移的确证

继续审阅发现raw运行时的`profile-15`虽然`success=true`，仍缺少已声明的`tzdata==2025.2`和marshmallow源码`3.23.1`的dist-info。旧R2 CPU完整验收的原始输出已因缺少`America/Chicago`的ZoneInfo同时阻断baseline和参考修复；tail2 CPU在补时区资源并以该源码的flit backend生成准确dist-info后，baseline为1 failed、1230 passed，参考修复为1231 passed。这些是历史不同环境的真实记录，不能把tail2的usable套到后来的raw profile-15。

来源为`next-round/runtime-rebuild-v4-evidence/vfarl-preparation.json`、`kaggle/environment-profiles.json`、`kaggle/prevalidate-r2-evidence-v1/vfarl-prevalidate/prevalidation.jsonl`以及`kaggle/tail2-cpu-evidence-v1/vfarl-prevalidate/{prevalidation.jsonl,runtime-preparation-repair.json}`。tail2生成的METADATA哈希为`3ec6bd44c48fe4aeaedbb769b3be149e5c94f56eb565c9fd0cb02dc1dfac8d7b`，WHEEL为`1b68144734c4b66791f27add5d425f3620775585718a03d0f9b110ba3a4d88db`。

R3冻结前将非stock绑定补齐为实际声明包版本、额外Linux资源，以及生成的源码metadata实例／版本／pyproject／backend来源和实际文件哈希。一次性本地元数据核对确认：raw profile-15被拒；只补版本而无源码来源仍被拒；完整来源与文件哈希相符时可绑定；修改METADATA文件后被拒。该核对只验证真实问题的绑定判别，不等于五题的新Linux完整验收已经完成。

随后按raw preparation全部17个profile的真实包版本记录构造临时dist-info做一次固定100范围绑定回放：83题可绑定、17题不可绑定；其他81道非stock任务均保留绑定，唯一不可绑定的非stock是上述marshmallow。dominate和mistune两道实际依赖完全等价的任务均可复用profile-00。此为元数据判别回放，没有运行这些任务的Linux导入或完整验收。
