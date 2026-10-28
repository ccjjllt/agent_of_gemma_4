# 索引84：原SVG oracle的实际输出语义盲点

2026-10-28。仅固定100 train索引84、`lincolnloop__python-qrcode.456b01d4.func_basic__rpcljfd7`。本目录全部产物是 **oracle弱点探针，禁止训练消费**；没有teacher/messages/SFT/canonical增量，也没有更改奖励、selection或原oracle。受控错误生产patch只在报告指定的/private/tmp目录，未保存进训练材料。

原source任务与当前题面相同：`_svg`返回None而不是proper SVG element，导致保存/使用失败；期望正常SVG生成。生产源码的公开类契约分别是SvgFragmentImage创建SVG document fragment，SvgImage创建standalone SVG document，SvgFillImage填充白底。真实源/任务/快照/patch及运行时hash、完整题面与源docstring保存在[oracle-observations.json](oracle-observations.json)。

同一未改动官方verify_task、原6个F2P与75个P2P、8个完整目标文件，四次新的CPU结果：

|生产变体|原81用例|官方resolved|实际SVG文档观察|
|---|---:|---|---|
|原mutant|6失败／75通过|False|返回None，保存失败|
|只返回super()._svg|81通过|True|XML无SVG namespace，白底缺失|
|受控返回非SVG根元素|81通过|True|真实序列化展开根为not-svg|
|原已审reference|81通过|True|展开根为{http://www.w3.org/2000/svg}svg，白底rect正常|

第三行是题目内的直接反例：返回值虽然可被XML序列化，文档根确实不是SVG，违背proper SVG element及standalone SVG document契约。没有伪造stdout、mock verifier或改测试。第二行单列为namespace/背景的相邻公开契约边界；不单独把这类partial修复与OSA等同。

六个F2P函数只调用make_image/save或CLI；capsys参数没有读取/断言输出。已追save实现，只将任意Element序列化。完整81例的真实成功证明其他P2P和fixture也未拒绝非SVG文档；不只是静态统计缺assert。每次原始官方观察与JUnit均保留。受控witness使用同一临时生产文件并记录实际import路径及文件SHA，再解析实际保存字节；正确reference作正控制。

建议在新可信奖励协议形成前，暂停84未来RL奖励消费，保留原真实结果与正确SFT。此建议没有在本目录修改eligibility，也没有反向把历史奖励改写为null/0。macOS CPU反例不是Kaggle/Linux等价或模型修复收益；没有GPU、训练、新测试文件或网络下载。
