# 索引16：boto外层契约的只读来源补充

范围仅固定100的 `getmoto__moto.694ce1f4.pr_4924`。读取旧真实 run 的 `source-snapshot.tgz` 中明确的 ELBV2／mock入口源码成员，没有执行工具／API／oracle、创建环境或新 teacher。原审阅 JSON 保持不变。

源归档 SHA：`17816b2e68a0321955ef765899a458ecb1212abe35b1149218bf6bc84caf014f`。

调用链在该 snapshot 是直接的：`moto/elbv2/responses.py:554–564` 的 describe_listeners 先读取 LoadBalancerArn 和 ListenerArns.member；两者都空时，直接 `raise ListenerOrBalancerMissingError()`，尚未调用 backend。`models.py:1612–1628` 的 backend 方法负责已有 ARN 查询，不负责这条空参数验证。异常类在 `exceptions.py:151–156` 却要求 arn，于是旧 teacher 直接构造异常时观察到的 TypeError 正是外层空参数路径要触发的错误。只改构造器去掉 arn 的现有 patch 与该调用点一致；旧行为证据有效。

若后续补外层观察，应使用这份源代码真实导出的 **`moto.mock_aws`**。`moto/__init__.py` 显示版本 `5.1.1.dev`，不是其他固定题常见的 `mock_elbv2` API；`moto/core/decorator.py:25–37` 提供 mock_aws context。后续有限 inline 方案可在实际 in-process MockAWS context 中，以显式测试凭据创建 boto3 elbv2 client 并调用 describe_listeners()，分别观察原 TypeError 与修后 ClientError 的 ValidationError/code/message。该方案没有在这里执行，不声称外层已通过；不得借其他 Moto snapshot 的 decorator 名称。

源码成员 SHA：

| 成员 | SHA256 |
| --- | --- |
| moto/elbv2/responses.py | `8e794bcf122bb42fe9a5ac2a5f422875156483172735def700bb01d2574608fa` |
| moto/elbv2/models.py | `276370bc8839bdd122b3c740185e717ff06821cd311f3d1fcf3c56c6d0349cc1` |
| moto/elbv2/exceptions.py | `9c6702c0f2e7a96f3da032ecac86e916f2580afd3b128a68a98f17643bf50bba` |
| moto/__init__.py | `600780105550cce1c3a11d1c71298fa92ca2df62bd273edfa6b66f6b5772722b` |
| moto/core/decorator.py | `88bd153a65fd3641f077acea735870069bc74540d201b6d45673b0e3a6fe8249` |

这不是新缺陷或新修复收益，只把可选外层 replay 所需的真实 API 和同 snapshot 调用契约说明清楚。
