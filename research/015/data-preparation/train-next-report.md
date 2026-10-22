# 后续同协议 train 采样材料

2026-10-22。仅使用 `train-identity-freeze.json` 原32身份，排除已采样的xmltodict81、boltons338/298；没有扩大身份池、读取dev/select/reserved内容、启动GPU或自行推远程。最终8题/8仓以原冻结身份顺序排列。选择依据是来源与CPU环境可复现性，没有使用31B胜负或参考模型表现选题。

唯一冻结入口为 `research/015/data/train-next-material/`，manifest SHA256为 `c0b1590da817f077a9c2dfe48ea67066b63b8c8f7e56dad7edb8cf00d7799171`。244个成员共71,656,251 bytes。该目录已交evaluation负责人封装独立后续CPU作业，此后不在原地改动。当前15题合并CPU载体、主worker与已运行服务均未改动。

| 原冻结位置 | 独立问题 | 来源F2P | 来源P2P |
|---|---|---:|---:|
| 1 | marshmallow-code/marshmallow2715 | 10 | 25 |
| 2 | graphql-python/graphene1478 | 1 | 27 |
| 6 | john-kurkowski/tldextract258 | 1 | 3 |
| 11 | theskumar/python-dotenv260 | 1 | 49 |
| 16 | marshmallow-code/apispec804 | 4 | 244 |
| 18 | dask/dask9212 | 2 | 87 |
| 26 | sloria/environs379 | 4 | 100 |
| 28 | oauthlib/oauthlib771 | 1 | 35 |

位置从0起。每题均在最终每题依赖集合下实际执行：buggy的来源F2P全失败、P2P全通过；reference的来源F2P/P2P全通过，没有skip冒充pass。完整输出在ignored材料的cpu-logs，精确依赖集合手动结果在 `train-next-exact-pins-results.json`；来源与实际环境准备失败在 `train-next-preparation.json`。这些是可采样环境与判据证据，并非模型新增修复或训练收益。

本机实际为macOS arm64、CPython3.12.13，复用既有pytest8.3.4包，未新装测试框架。每题最终requirements均附可离线安装的CPython3.12 Linux x86_64轮子；本地Dask使用同版本NumPy2.0.2/Pandas2.2.3的Mac二进制。simplejson/PyYAML本地验证使用相同wheel内的纯Python实现，未在Mac冒用Linux扩展。尚未声称实际Linux执行通过，必须由根安排的新CPU作业确认后再部署采样。来源环境常是Python3.9，实际包版本及变化逐项保留；dotenv来源sh1.14.3没有312兼容wheel，实际采用2.2.2，全部原P2P已执行。

## 输入边界

`agent-tasks.json` 仅iid/repo/base_commit/原始issue，逐字复制来源。`snapshots/<iid>.tar.gz` 根为repo/，含原始真实Git对象，HEAD与原base_commit一致、工作树干净。每题原有tests保留原版本；来源新增criterion、参考修复、teacher消息均未进入Agent快照。原仓代码许可证随完整源码保留，不把数据集许可冒充各仓代码许可。

`host-criteria.json` 单独保存来源节点、criterion_patch与reference_patch路径。判据只在fresh judge上施用，不是生成期输入。无现成独立问题切片、人工恢复标记或teacher未来消息混入采样问题。

## tldextract：必须保留标准构建步骤

本地实际浅Git HEAD为 `86c82c3a9e3e9a2980a39ce8b77e1ad0bdebcc12`。固定setuptools75.8.2、setuptools_scm8.3.1、packaging23.0；在临时来源工作树执行原有 `setup.py --version`，退出0，并由原标准SCM集成生成 `tldextract/_version.py`。浅clone无release tag时真实生成版本为 `0.1.dev1+g86c82c3`，并非冒称3.2 release；未通过手写_version.py伪造环境。Agent原始快照在该生成步骤之前已封存，不能把本地生成结果塞进原base。

后续独立Linux CPU与Agent/judge都必须读取该题environment.install_editable=true，按对应wheelhouse执行如下标准流程（替换实际路径）：

```sh
<venv>/bin/python -m pip install --no-index --no-deps --find-links=<wheelhouse> -r <requirements.txt>
<venv>/bin/python -m pip install --no-index --no-deps --no-build-isolation --find-links=<wheelhouse> -e <repo>
```

setuptools75.8.2、setuptools_scm8.3.1、packaging23.0、wheel0.45.1均在该题离线wheelhouse。本地没有声称已执行该完整pip editable流程，实际执行的是原setup.py生成步骤与随后完整原F2P/P2P；新CPU入口必须补做标准editable安装后核验源码origin。源布局应使用 `PYTHONPATH=<repo>/src:<repo>`。

## Dask：保留原执行语义

原source_test_cmd已明确 `-W ignore::DeprecationWarning`；原requirements含pytest-rerunfailures15.0。未带原flag时NumPy2的np.compat告警会被源pytest配置当错误；未加载原插件时flaky标记无法注册。已按原语义显式载入 `pytest_rerunfailures` 并保留原警告flag，没有泛化抑制其他告警、修改标记或重写判据。

environment中保存pytest_plugins、pytest_args；每题f2p_command/p2p_command也完整写明原节点和flag，兼容当前支持custom command的worker。Linux新CPU入口必须执行相同命令。Dask的全部87原P2P在本地两态均通过，没有optional skip补数。

## 真实失败与准备修正

- marshmallow1249的来源P2P含被截断的参数节点`[My`，buggy/reference精确collection均退出4；尽管其2个F2P能复现并被reference修复，本批仍未用它凑数，转同仓冻结备选2715。
- Trio1575的旧traceback补丁不兼容Python3.12的compact参数；freezegun245直接引用已移除time.clock；tenacity201直接使用已移除asyncio.coroutine。未补生产兼容shim或伪造通过。
- Safety343的F2P可复现，但4个来源P2P依赖pyup.io网络/用户缓存，离线两态均失败，因此不收入本次离线采样池。
- DVC3097、旧Tornado2058、SQLFluff4151分别有mockssh、旧SSL backport、tblib/旧CLI插件工具链等CPU准备缺口；记录为本批环境未准备，不据此断言原问题无效或模型失败。
- 首个简化codeload提取器拒绝Tornado/SQLFluff原仓合法symlink，后改为精确Git fetch，未把提取器限制误报成来源404。
- Environs379初次reference复制了原路径的pytest字节码，find_dotenv从栈中读到旧co_filename，造成3个路径假失败。清理临时reference的__pycache__后，相同原P2P100项全部通过；未改源码或测试。后续CPU入口在同一绝对工作树依次运行两态并清cache，避免复制路径问题。

首次拟把所有仓首候选一起下载被自动审批认为超过“最多8题”而拒绝，实际未执行。随后明确限制每批最多8个活跃身份，已有真实准备失败退出后才按冻结序补位。无新建测试脚本、套件或门控项目；上述验证全部为一次性手动命令及来源上游判据。
