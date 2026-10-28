# 显式新任务Python缓存配置

2026-10-28。实际受控恢复遇到“正确源码已写但新进程仍执行旧值”后，另做了真实归档Moto/official SubprocessManager复现。源码7→6保持同秒mtime及3211字节大小；不同新PID的实际secret_arn结果依次为7、7、6、6、7。第二次源码已为6，legacy仍读旧缓存；第三、四次采用新配置读到6；第五次显式python-E忽略环境而再次读到7。源码SHA、mtime_ns、PID、cache路径／存在性及旧pyc头和SHA保存在[actual-manager-moto.json](actual-manager-moto.json)。旧source-adjacent pyc在全程保持原SHA，没有删除任何依赖、任务源码或模型缓存。单次实际耗时legacy约0.18–0.20s，新配置约0.53–0.57s，说明禁用任务Python字节码复用也有编译成本；这是少量本机观察，不能外推真实训练吞吐。

`isolated-prefix-no-write-v1`是显式选择的新profile，默认`legacy-default`不替换manager.exec。新profile只为该manager的官方bash命令正文添加export：PYTHONDONTWRITEBYTECODE=1，PYTHONPYCACHEPREFIX=$TMPDIR/.vfarl-empty-bytecode-v1。它在official sanitizer后生效，绕开旧source-adjacent缓存查找，并禁止常规新进程写cache；不靠sleep、改mtime、删除cache或仅设host变量。实际parent已有dontwrite=1时legacy任务Python仍为False，也验证了host flag不被自动继承。

调用入口为run_official的`--source-bytecode-policy isolated-prefix-no-write-v1`，或未来新asset/runtime.json显式字段`source_bytecode_policy`；TaskEnvironment在phaseA与所有独立official验证使用同一个已绑定manager。新prevalidation记录policy，cpu_evidence_applicability对新policy拒绝旧默认记录；缺field只解释为历史legacy。[policy-applicability.json](policy-applicability.json)是基于真实资产hash的纯metadata函数观测，不是新的CPU usable。现有100题selection、冻结runtime、carrier、历史CPU结果和SFT均没有因此换名或自动启用。

静态审阅确认manager.exec_async经self.exec覆盖，Snapshot绑定实际wrapper源SHA并保存完整任务tmp；build/build_runner的l4x4/*.py收录新模块。chroot/jail仍将TMPDIR映射到同任务root/tmp，host launcher的-I不作用于之后普通task Python。见[独立静态复核](static-review.json)；这些是代码依据，未执行Linux/chroot/bwrap。直接文件CLI曾因新增包导入缺repo root失败，已仅在__main__直接调用时补包根；official-host-only PYTHONPATH的--help随后成功。首个证据命令错误使用dataclass结果的model_dump，清理后改用asdict再执行，失败信息保留在JSON。

此profile不处理已经import的活module对象、不保证Python-E/-I、主动覆盖环境或直接导入无源码pyc的语义；新prefix为空是新sandbox的初始状态，任务主动往该prefix制造pyc仍是另一种动作。共享依赖源/资源继续复用，但普通任务Python的既有字节码加速会被绕开。GPU模型进程环境不改。没有新测试文件、任务奖励、模型采样、训练或GPU作业，未宣称R3旧失败已证实由同一pyc头导致。
