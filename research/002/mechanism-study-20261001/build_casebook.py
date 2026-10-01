#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a reviewed, bounded mechanism casebook from collect.py evidence."""
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent
DATA=json.loads((OUT/'evidence.json').read_text())


def ref(label, attempt, steps, run_contains=None):
    rows=[r for r in DATA['attempts'] if r['attempt_id']==attempt and (not run_contains or run_contains in r['run_id'])]
    assert len(rows)==1,(attempt,run_contains,len(rows))
    row=rows[0]
    selected=[e for e in row['events'] if e['step'] in steps]
    assert {e['step'] for e in selected}==set(steps),(attempt,steps)
    return {'label':label,'attempt_key':row['key'],'raw_resolved':row['resolved'],'trace_path':row['trace_path'],'trace_sha256':row['trace_sha256'],'patch_path':row['patch_path'],'final_production_files':row.get('final_production_files'),'selected_tool_events':selected}


CASES=[
 {
  'id':'M01','task_id':'fastapi_14873','title':'同一字段初始化顺序，早已读到却迟迟未改',
  'issue_contract':'APIRouter.on_startup/on_shutdown 的用户值被父类初始化覆盖；验证构造后字段是否仍含处理器，兼顾 startup/shutdown。',
  'comparison':'0C 同一 C8192 组 r1 成功与 r2 失败；另看 T0 r1 失败。不是对新策略的随机对照。',
  'evidence':[
   ref('C8192 r1 成功','06-fastapi_14873-r1-c8192',[11,27,30,31,32,35,36]),
   ref('C8192 r2 失败','17-fastapi_14873-r2-c8192',[11,27,33,42,43,44]),
   ref('T0 r1 失败','05-fastapi_14873-r1-t0',[11,13,30,41,46,47,48]),
  ],
  'observations':['四次0C运行都在25.8~27.4秒成功读取routing.py 951~1000附近；文件召回已完成。','成功C8192在150.9秒取得handler未执行反例；174.2秒替换失败后176.0秒回读，196.5秒成功修改，242.7秒见证通过，243.7秒提交。','失败C8192到235.7秒已打印startup/shutdown均False，但接着在290.8/292.8/294.9秒从routing.py首部步进重读，没有成功生产编辑；最终patch仅repro.py。','失败T0的早期脚本只打印进入TestClient，没有断言handler真的执行；之后另一个脚本报错，仍反复读取相同区域。'],
  'mechanism_hypothesis':'缺少把已确认局部状态覆盖关系转换为一次可逆修改的行动承诺。检索更多文件不是已证实的必要步骤。',
  'candidate_action':'记录覆盖前后两个赋值点；复现可直接检查router字段或handler计数。局部因果链完整后先移动赋值/修最小块，再重跑同一见证。',
  'limit':'文件命中不是已理解证明；初始化顺序语义来自离线题面/gold核对，不能把它作为未来任务答案注入。',
 },
 {
  'id':'M02','task_id':'fastapi_9425','title':'全绿的错误反例与正确的None/字符串None区分',
  'issue_contract':'题意指返回注解字符串"None"；必须区分值None、字符串"None"以及求值后的NoneType。',
  'evidence':[
   ref('0C C8192 r1 失败','07-fastapi_9425-r1-c8192',[7,25,37,39,44]),
   ref('0C T0 r2 成功','19-fastapi_9425-r2-t0',[7,14,44,45,50]),
   ref('P003 A0 修改错误层','24-fastapi_9425-a0',[16,26,37,44,45,47],'panel'),
   ref('P003 B 修改转换层成功','23-fastapi_9425-b',[5,16,43,46,47,48],'panel'),
  ],
  'observations':['失败C8192的134.7秒复现和245.3秒测试都PASS，但展示的是->None路径；最终只留下4个根目录临时/测试脚本，无生产变更。','成功T0在53.3秒直接比较None与字符串None的get_typed_annotation结果，显示后者变成NoneType；113.6秒在utils转换层修正，114.5秒同一见证通过。','P003 A0在171.4秒修改routing.py的response_model分支，测试继续打印Response model: None却官方失败；B在utils.py归一化字符串None后官方成功。'],
  'mechanism_hypothesis':'“复现命令成功”与“已覆盖问题触发条件”被混同，导致错误地点修改或在不相关成功情形上停滞。',
  'candidate_action':'复现前写出类型/边界对照表，用能区分竞争假设的最小断言；修改数据转换的不变量，避免仅放宽下游报错条件。',
  'limit':'各组/重复不同，不能从此证明capacity或prompt造成差异。成功补丁与gold文本不同但裁判通过，最小修复不要求抄gold。',
 },
 {
  'id':'M03','task_id':'fastapi_11194','title':'UploadFile全绿掩盖bytes转换失败，首个有效反例太晚',
  'issue_contract':'Form/File顺序只是一个轴；UploadFile对象与bytes读取分支是另一个必要轴。',
  'evidence':[
   ref('smoke A0 成功','smoke-fastapi_11194-a0',[4,6,21,34,35]),
   ref('探索B成功','02-fastapi_11194-b',[4,6,31,43,44],'explore'),
   ref('面板B失败','02-fastapi_11194-b',[4,8,13,15,17,44],'panel'),
   ref('面板A0失败','01-fastapi_11194-a0',[4,7,14,18,20],'panel'),
  ],
  'observations':['成功smoke先得到200，换成bytes后51.0秒看见422；277.6秒将使用first_field_info改为当前field.field_info，278.6秒顺序互换见证均200。','探索B同样从20.4秒的200转到45.3秒的bytes 422；289.5秒完成局部修改，290.2秒提交并通过。','面板B在14.1、60.9、163.5、177.7、194.3秒连续验证UploadFile类情形全绿，最终直到约300.5秒才成功读到utils.py，没有生产编辑。','面板A0到116.8秒已读到gold锚点，但有效bytes反例晚至204.1秒；仍未形成修改。'],
  'mechanism_hypothesis':'泛化复现脚本数量增加没有覆盖关键类型分支；之后已定位时还缺从循环不变量到局部变更的闭环。',
  'candidate_action':'第一轮复现按问题涉及的输入类型和顺序做最小二维见证；修复每个field应按自己的field_info处理这一局部不变量。',
  'limit':'289秒的成功说明不能把早编辑或固定240秒硬停当普遍必要条件。',
 },
 {
  'id':'M04','task_id':'rich_2725','title':'相同语义修复，一行选择索引优于反复大块重排',
  'issue_contract':'头/中/尾的渲染字符映射必须与当前行角色一致，用不同哨兵字符检查位置。',
  'evidence':[
   ref('探索B成功','07-rich_2725-b',[7,12,21,22,24],'explore'),
   ref('探索A0失败','08-rich_2725-a0',[16,21,29,30,33,34,36,39,40],'explore'),
   ref('面板B失败','34-rich_2725-b',[9,13,16,21,23],'panel'),
  ],
  'observations':['成功B用88字符旧串/88字符新串的一行修改，将box_segments的last与middle索引交换；step21/206.2秒返回ok，官方通过。','失败A0早在142.7秒读到目标列表，227.4~295.0秒七次尝试464~721字符多行块替换都old_string not found；另在89.1秒尝试修改box.py也失败。','面板B在39.6秒读table.py、61.1秒见锚点，却在127.6/159.8/174.1秒连续修复复现脚本异常，没有生产编辑。'],
  'mechanism_hypothesis':'语义已接近正确时，大编辑面与长多行参数造成执行脆弱；另一些运行停留在复现工具自身的复杂度。',
  'candidate_action':'在语义等价且局部可验证时选择更小的变更；一次替换失败后强制回读精确片段并缩小替换，而非换转义重发整块。',
  'limit':'一行修改是此例的可行解，不意味着所有任务都应选择最少字符或最小diff。',
 },
 {
  'id':'M05','task_id':'rich_3905','title':'修复后58秒已有见证，仍探索到283秒；双向见证可形成收尾条件',
  'issue_contract':'disabled=True不应输出空行；正常enabled行为应保留。',
  'evidence':[
   ref('探索A0成功但未主动收尾','14-rich_3905-a0',[7,9,10,11,16,24,25,26,27],'explore'),
   ref('探索B成功且很快收尾','15-rich_3905-b',[7,14,15,16,17],'explore'),
  ],
  'observations':['A0在35.0秒观察到多余换行、57.0秒成功改stop、57.9秒复现变为空输出；随后16个工具事件持续到282.8秒，没有额外成功生产修改，也没有submit。','B在65.9秒修改，66.7秒同时输出Disabled为空、Enabled有正常换行，67.8秒提交；两组最终均resolved。'],
  'mechanism_hypothesis':'缺少基于证据的完成判据，让已经成立的修复继续消耗预算；比最后20秒的时间提醒更有针对性的出口是反例+邻近正例+差异检查。',
  'candidate_action':'已建立前后反例并通过一个受影响邻近正例后，做一次diff/语法/文件检查并提交；若出现新失败再重新开放调查。',
  'limit':'潜在可节约时间是此历史轨迹的反事实，不是已实测新策略提速；两组补丁细节不同，不能据此证明早停对所有任务无害。',
 },
 {
  'id':'M06','task_id':'rich_3518','title':'明确的一行属性继承可用最小端到端修复路径完成',
  'issue_contract':'add_row隐式新增Column时应继承Table.highlight；显式已有列与隐式新增列成对检查。',
  'evidence':[
   ref('0C较短成功路径','24-rich_3518-r2-c8192',[4,7,9,10,11,14]),
   ref('P003 A0从四次参数失败中恢复','41-rich_3518-a0',[7,12,13,14,15,16,18,20],'panel'),
  ],
  'observations':['0C成功路径：7.8秒读文件、36.4秒目标行、60.6秒见证失败、79.4秒单行加入highlight=self.highlight、80.2秒两列均True、104.5秒提交。','P003 A0在101.0~116.5秒四次缺old_string参数错误，122.9秒才成功；之后复现通过、157.8秒提交。','该题六次结果均raw成功；一条T0成功trace缺失，因此这里不是全部六条路径的全局最短证明。'],
  'mechanism_hypothesis':'无需扩大检索：题目已给出属性继承契约，主要成本来自建立见证与将精确行修改执行成功。',
  'candidate_action':'把这一类问题压缩为“定位构造点→父子对象属性见证→单行继承→双列断言→提交”的策略模板，模板只包含推理步骤而不含任务答案。',
  'limit':'目前五条可读轨迹中更短的观察案例；不能把相同模板的未来成功率或耗时写成已测事实。',
 },
 {
  'id':'M07','task_id':'fastapi_5077','title':'打印Success不等于类型解析成功，issue已经给出函数与关系',
  'issue_contract':'包装函数保留字符串注解但globals属于另一模块；解析结果应是原函数模块中的实际类型，不只是调用不抛异常。',
  'evidence':[
   ref('面板A0失败','21-fastapi_5077-a0',[4,7,9,12,17,23],'panel'),
   ref('面板B失败','22-fastapi_5077-b',[3,5,8,16,19,23,27,33],'panel'),
  ],
  'observations':['两组18.7/25.3秒已成功读目标utils.py，25.9/35.0秒命中gold旧行。题面本身还明确指出get_typed_signature、__wrapped__与__globals__关系。','A0最初将原函数和wrapper放同一脚本，129.3秒打印Success；后来跨模块脚本仍仅以不抛异常打印Success。','B在116.0与224.1秒同时打印get_typed_signature success和Parameter a: ForwardRef(\'MyModel\')，没有把未解析的ForwardRef断言为失败；297.8秒才出现Test failed，无生产编辑。'],
  'mechanism_hypothesis':'测试判据弱于题目要求，掩盖了已经观察到的语义差异；这是反例契约与断言设计问题，不是缺少文件名。',
  'candidate_action':'将issue内的对照接口get_type_hints与目标函数输出进行类型身份对比；满足定位证据后修globals来源，保留输入/返回注解两处的一致性检查。',
  'limit':'两次皆失败，没有同题成功对照；建议来自明确工具结果与离线目标核对，仍需新的冻结实验验证。',
 },
 {
  'id':'M08','task_id':'rich_3130','title':'复现没有建立表格路径，却在同一输出上循环十次',
  'issue_contract':'Markdown表格单元格接收多个有不同样式的文本片段时应追加，不能每片覆盖。',
  'evidence':[
   ref('P003 A0循环至末期编辑失败','36-rich_3130-a0',[4,6,23,25,27,29,31,33,35,37,39,41,43,44,46],'panel'),
   ref('P003 B迟修改但有生产变更','35-rich_3130-b',[4,6,43,45,47],'panel'),
  ],
  'observations':['A0在15.7秒已读到目标锚点；最终复现文本仅一行竖线包住Bold/Italic，没有表头分隔行，输出保留原始竖线，未建立TableDataElement被触发的反例。','248.3~284.6秒十次运行给出相同Result，之间重复写同一个脚本；291.5/299.5秒两次生产编辑都old_string not found。','B在254.4秒能完成修改并提交，但本题在后续0B中gold也失败，所以不能把最终resolved=False全部归因这个候选改动。'],
  'mechanism_hypothesis':'重复确认同一个无区分力输出没有新信息；双动作“重写→运行”循环逃过简单相邻重复检测。',
  'candidate_action':'对见证记录输入哈希、目标执行路径证据、断言和输出摘要；相同输入/输出/无源码变化重复两轮后，下一步必须改触发条件或执行局部修改，不能再重跑同一脚本。',
  'limit':'本题保留为过程诊断，不作为修复成功率比较的有效任务；判断依据不是模型thinking全文。',
 },
 {
  'id':'M09','task_id':'requests_6592','title':'提交是补丁事务的最后一步，提交后清理没有改变已冻结产物',
  'issue_contract':'生产status code更新与最终提交文件集合是两个独立契约；清理应在submit之前完成。',
  'evidence':[ref('smoke A0提交后才删除临时文件','smoke-requests_6592-a0',[22,25,26,27,28])],
  'observations':['148.1秒完成生产编辑；155.6秒运行自建tests/test_status_codes.py；157.7秒submit返回4个changed files。','169.4秒才删除reproduce_issue.py、check_codes.py、自建测试；最终冻结patch仍包含这些临时/测试文件。'],
  'mechanism_hypothesis':'把submit当作进度标记而非不可变产物快照，导致卫生检查顺序失效。',
  'candidate_action':'最终diff检查、临时文件清理/排除与语法检查先完成，再submit一次；收到成功响应后停止工具动作。',
  'limit':'requests_6592有已证验证环境故障，本例不支持任何模型修复能力归因，只用于提交事务设计。',
 },
]

STRATEGIES=[
 {'id':'S1','name':'有区分力的反例合同','supported_cases':['M01','M02','M03','M07','M08'],'mechanism':'把题意转成一个故障触发轴和一个邻近对照；测试须断言目标语义，不以exit=0、打印Success或不抛异常替代。','candidate_protocol':['先记录issue中的具体差异（如字符串/值、bytes/UploadFile、wrapper/original globals）和预期观察。','第一版脚本只含一个目标断言和一个对照断言；若全绿，检查是否漏掉触发轴，而非追加同类场景。','保存已运行脚本输入哈希、断言、实际输出与仍未知项；两轮同输入/同输出且无源变更时禁止第三次原样运行。'],'what_changes':'执行策略与证据记录；不需扩大图检索、预算或训练。实现时不得使用本casebook里的任务专属答案。','test':'与冻结T0/T1底座配对；主要观察有效反例形成时间、题意分支覆盖、同证据重复次数及最终eligible修复。策略产生的断言质量需盲审，不靠脚本数量衡量。','risk':'错误合同会约束错误方向；允许当观察反驳合同后显式修订，不能机械禁止合理重复。'},
 {'id':'S2','name':'证据到最小编辑的局部事务','supported_cases':['M01','M02','M04','M06','M07'],'mechanism':'在具体因果位置和反例已确定后完成一次可逆变更，避免继续无界读取；编辑失败时改变操作而非重发等价参数。','candidate_protocol':['将当前源片段、预期不变量、最小语义变更和同一验证断言放进一个短记录。','优先一次局部语义变更；一次旧串不匹配后必须回读精确片段并缩小编辑范围，禁止原样连续重试。','成功编辑后立即运行同一见证，失败则依据新差异修订；禁止通过删除下游检查来替代上游语义修复。'],'what_changes':'候选可提交skill和使用纪律；不是强制固定时间开始编辑，也不是每题只允许一行。','test':'比较成功读取相关文件→首次成功生产修改的转化率、无效编辑重试数、正确函数/语义位置盲审及resolved；所有指标按task聚合，避免单题循环占权重。','risk':'file touch不足以触发编辑；需要明确局部证据，防止为提高编辑率制造无根据patch。'},
 {'id':'S3','name':'证据闭环后的条件收尾','supported_cases':['M05','M06','M09'],'mechanism':'用故障见证修复、邻近对照未退化和最终diff通过来决定结束，不依赖最后20秒的软提醒。','candidate_protocol':['已观察目标反例fail→pass，并通过一项受影响邻近行为后，进入一次最终diff/语法/文件检查。','清理或排除临时文件，检查允许的生产变更集合；随后submit一次并停止。','若检查产生新的失败或不确定性，可重新调查，但需记录新证据，不能仅因仍有时间继续浏览。'],'what_changes':'条件式终止协议；不缩减硬预算、不把主动提交率当主指标。','test':'先看同样resolved条件下从证据闭环到提交的冗余时间，并以eligible配对resolved/回退为主指标；不声称节省时间自动换成更多成功。','risk':'局部见证不保证完整回归安全；需保留与改动影响面匹配的验证，不能只跑自建快乐路径。'},
]

output={'date':'2026-10-01','purpose':'Mechanism hypotheses and implementation handoff only; no candidate executed.','provenance':'evidence.json generated by collect.py; source hashes and exact tool step/time retained. No long model thought text exported.','gold_boundary':DATA['gold_boundary'],'summary_funnel':DATA['funnel'],'historical_oracle':DATA['historical_oracle'],'cases':CASES,'candidate_strategies':STRATEGIES,'recommended_order':'S2 first (planned T2 read-handle to exact minimal edit), then S1 (planned T3 discriminating counterexample); S3 separately. No change to frozen T0/T1. Context stratification is a separate study. Do not simultaneously modify capacity, thinking, graph retrieval, hardware or LoRA and attribute gains to these policies.'}
(OUT/'casebook.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'cases':len(CASES),'strategies':len(STRATEGIES),'casebook':str(OUT/'casebook.json')},ensure_ascii=False))
