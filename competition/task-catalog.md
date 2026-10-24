# 129 个公共开发任务的逐题机制目录

2026-10-24；来自当日重新下载的官方公开 tasks.jsonl。主机制用于统计，边界见 [研究报告](README.md)；机器可读版本见 [task-catalog.json](task-catalog.json)。以下为题面、参考实现和验收补丁的静态审阅，不代表已经运行或修复成功。

129题全部已用于本轮公开数据分析，今后不能把其中任何旧本地保留集宣称为从未查看的独立样本。官方隐藏评测未读取。

## M1 局部规则与边界条件（23题）

既有路径中的判定、默认值、常量或声明式输出规则有误。 先明确输入边界和期望输出，做局部或横向一致的规则修复。

| ID | 实际工作 | 定位 | 参考补丁规模 |
| --- | --- | --- | --- |
| fastapi_15588 | 对 SSE event/id 添加 CR、LF 单行校验，id 继续禁止 NUL；通过 Pydantic AfterValidator 组合验证。 | fastapi/sse.py:_check_single_line/_check_id_valid/ServerSentEvent | 1文件 +18/−5 |
| fastapi_15280 | 在 FastAPI 新增 vibe(path,prompt) 方法，方法立即抛 FastAPIError("Are you kidding me? Happy April Fool's")；附一个示例。 | fastapi/applications.py:FastAPI.vibe | 2文件 +71/−1 |
| requests_7427 | no_proxy 按完整域名或点分域边界匹配，同时区分端口，避免后缀碰撞。 | should_bypass_proxies | 1文件 +4/−2 |
| requests_7205 | netrc 返回非空元组但所有字段为空时忽略凭证，要求 any(_netrc)。 | get_netrc_auth | 1文件 +1/−1 |
| rich_4075 | Console.print在没有objects时按end选择NewLine或空字符串，避免end为!等非默认值时仍插入换行。 | Console.print: if not objects | 1文件 +12/−3 |
| fastapi_14258 | 在 include_router 入口断言 self is not router，并提供清晰错误消息。 | fastapi/routing.py:APIRouter.include_router | 1文件 +4/−0 |
| fastapi_14492 | 把可执行 metadata 示例 license_info.identifier 从 MIT 改为 Apache-2.0，使名称与 SPDX 标识对应。 | docs_src/metadata/tutorial001_1_py39.py:FastAPI(license_info=...) | 1文件 +1/−1 |
| fastapi_14479 | 为 analyze_param 的 Query 类型支持断言添加包含参数名的消息。 | fastapi/dependencies/utils.py:analyze_param | 1文件 +1/−1 |
| fastapi_14361 | 在定义/字段映射重命名流程中增加 "$ref" in schema 条件，避免访问未存在的引用键。 | fastapi/_compat/v2.py:_remap_definitions_and_field_mappings | 1文件 +1/−1 |
| rich_3944 | get_character_cell_size对大于宽度表最后范围的码点默认返回1而非0，恢复Nerd Font私用区字符的占宽。 | get_character_cell_size: codepoint > table[-1][1] | 1文件 +1/−1 |
| rich_3942 | 统一改动标题等级配色/对齐、取消h1 Panel边框，水平分隔符改为短横线并补空行，调整表格box、边缘/折叠padding和表头样式。 | DEFAULT_STYLES markdown.* / Heading.__rich_console__ / HorizontalRule.__rich_console__ / TableElement.__rich_console__ | 2文件 +43/−27 |
| fastapi_13537 | 用全部声明字段别名集合判断 extra，而不是看已提取非 None 值的字典；使空字符串按 missing 处理得到 None 后不再被第二遍回填。 | fastapi/dependencies/utils.py:_extract_form_body | 1文件 +2/−1 |
| fastapi_13786 | 为 APIKey/HTTP/OAuth/OpenId 基类统一 make_not_authenticated_error 和可覆盖 challenge 构造；认证缺失/方案错误返回 401，包含相应 WWW-Authenticate，auto_error=False 继续返回 None。 | fastapi/security/api_key.py:APIKeyBase; fastapi/security/http.py:HTTPBase/HTTPBasic/HTTPBearer/HTTPDigest; fastapi/security/oauth2.py; fastapi/security/open_id_connect_url.py | 11文件 +176/−83 |
| fastapi_11194 | _extract_form_body 循环内使用当前 field.field_info 判断 File 与 bytes/sequence 读取，而不是所有字段沿用 body_fields[0] 类型。 | fastapi/dependencies/utils.py:_extract_form_body | 1文件 +3/−4 |
| requests_6592 | 在425状态码映射中补 too_early 别名，同时保留原别名。 | status_codes._codes[425] | 1文件 +1/−1 |
| rich_3535 | 修正_is_single_cell_widths正则范围缺失的连字符与ASCII终点，增加box drawing等常见单宽范围；Segment.split_cells增加cut非负断言。 | rich.cells._is_single_cell_widths / Segment.split_cells | 2文件 +7/−1 |
| rich_3454 | ReprHighlighter URL正则字符集合增加@，避免视频站等/@name地址只高亮前缀。 | ReprHighlighter.highlights: (?P<url>...) | 1文件 +1/−1 |
| rich_3469 | Markdown.__rich_console__在node_type为inline时抑制前置_new_line_segment，防止空表格cell触发额外空行。 | Markdown.__rich_console__: if new_line and node_type != inline | 1文件 +1/−1 |
| rich_3278 | re_ansi增加ESC后接0到?的识别分支，把保存/恢复光标等private序列消耗掉，避免漏进渲染文本执行终端控制。 | rich.ansi.re_ansi | 1文件 +1/−0 |
| rich_2725 | Table._render的box部件三元组顺序从head/foot/mid改为head/mid/foot，使行角色与边框字符相对应。 | Table._render: head_left / mid_left / foot_left | 1文件 +5/−5 |
| rich_3043 | 将<html>起始标签移到<head>之前，恢复DOCTYPE→html→head/body的合法结构。 | CONSOLE_HTML_FORMAT: html/head order | 1文件 +1/−1 |
| rich_3067 | ReprHighlighter URL匹配字符集合加入~，避免/~folder等合法URL只匹配前缀。 | ReprHighlighter.highlights: (?P<url>...) | 1文件 +1/−1 |
| rich_3064 | TableElement.__rich_console__将必须存在header/header.row/body的assert改为可选分支，解析出部分表格时渲染已有部分。 | TableElement.__rich_console__: self.header / self.body | 1文件 +8/−9 |

## M2 数据表示与转换（24题）

数据跨字符串/字节、别名/字段、结构/Schema或富文本表示时丢失或误解信息。 写清转换前后语义，检查往返、基数、字节数及样式/位置信息。

| ID | 实际工作 | 定位 | 参考补丁规模 |
| --- | --- | --- | --- |
| fastapi_15589 | Header 模型解析时把规范化后的别名和原始 validation alias 均记为已处理，避免原始下划线 key 从 extra 字段通道再次进入模型。 | fastapi/dependencies/utils.py:request_params_to_args | 1文件 +4/−0 |
| fastapi_14962 | 为兼容 ModelField 添加 serialize_json；默认响应类且有 response model 时走 TypeAdapter.dump_json，直接构建 JSON Response；显式响应类保留原序列化路径。 | fastapi/_compat/v2.py:ModelField.serialize_json; fastapi/routing.py:serialize_response/get_request_handler | 3文件 +49/−6 |
| requests_7315 | 移除对请求路径开头双斜线的折叠，保留S3预签名路径原始字节语义。 | HTTPAdapter.request_url | 1文件 +0/−2 |
| requests_7309 | Content-Type 参数必须包含等号才写入映射，忽略 malformed charset 使编码回退正确。 | _parse_content_type_header / get_encoding_from_headers | 1文件 +6/−9 |
| rich_4079 | TableDataElement.on_text区分原始str与已带样式的Text；只对str施加当前上下文样式，Text保持原token高亮样式再拼接。 | TableDataElement.on_text | 1文件 +4/−3 |
| rich_4076 | AnsiDecoder.decode以换行后的零宽边界切分而非splitlines，保留末尾空行；仅去掉每段终止LF后交给decode_line，使Text.from_ansi往返保留换行数量。 | AnsiDecoder.decode | 1文件 +3/−3 |
| fastapi_14786 | Authorization 按首个空格拆 scheme 后，对 param.strip()，清除多空格分隔和尾部空白。 | fastapi/security/utils.py:get_authorization_scheme_param | 1文件 +1/−1 |
| fastapi_14349 | 递归 schema 重写遇到 $ref 键时，仅对字符串值提取/改写引用名；保留作为用户属性描述的字典值。 | fastapi/_compat/v2.py:_replace_refs | 1文件 +6/−6 |
| fastapi_14487 | 自定义 RequestValidationError handler 逐条提取 loc/msg 组成文本，替换直接 str(exc)，避免把完整错误对象中的输入内容泄露给客户端。 | docs_src/handling_errors/tutorial004.py:validation_exception_handler | 1文件 +5/−2 |
| fastapi_13207 | 即使 separate_input_output_schemas=False，检测到 computed_fields 时仍保留 validation/serialization 区分，以输出包含计算字段的 Schema。 | fastapi/_compat/v2.py:get_schema_from_model_field/get_definitions | 1文件 +12/−2 |
| fastapi_14360 | request_params_to_args 用 field.alias 而非 field.name 存取待校验数据，并只记录实际处理的别名，保留错误额外字段信息。 | fastapi/dependencies/utils.py:request_params_to_args | 1文件 +1/−2 |
| fastapi_14356 | 逐 key 获取 getlist 并按单值解包/多值保留 list；普通 Mapping 用 get，避免 items() 把重复 key 压成末值。 | fastapi/dependencies/utils.py:request_params_to_args | 1文件 +9/−2 |
| rich_3938 | 新增Segment.split_lines_terminator返回每行及是否含终止LF；Console.print对行内segment施加style，再追加无style的换行，避免背景跨行泄漏。 | Segment.split_lines_terminator / Console.print styled branch | 2文件 +40/−5 |
| fastapi_14791 | validation_error_definition 增加 input 与 ctx 属性，使 OpenAPI 描述匹配 Pydantic v2 返回错误结构。 | fastapi/openapi/utils.py:validation_error_definition | 1文件 +2/−0 |
| fastapi_14371 | ModelField 暴露 validation_alias/serialization_alias；抽取 get_validation_alias，统一 Query/Header/Form/Body 数据提取、校验 loc 和 OpenAPI 命名，按 mode 选择 Schema title。 | fastapi/_compat/v2.py:ModelField/get_schema_from_model_field; fastapi/dependencies/utils.py:get_validation_alias/request_params_to_args/_extract_form_body/request_body_to_args; fastapi/openapi/utils.py:_get_openapi_operation_parameters | 4文件 +46/−16 |
| fastapi_14303 | Form extra 字段逐 key 用 getlist，单值解包、多值保留 list，修复 items() 丢失重复值。 | fastapi/dependencies/utils.py:_extract_form_body | 1文件 +6/−2 |
| fastapi_14953 | 自定义 GenerateJsonSchema.bytes_schema，输出 contentMediaType、按配置补 base64 contentEncoding、保留字节验证约束；UploadFile schema 同步；加 JSON bytes 示例/截图脚本。 | fastapi/_compat/v2.py:GenerateJsonSchema.bytes_schema; fastapi/datastructures.py:UploadFile.__get_pydantic_json_schema__ | 4文件 +102/−2 |
| fastapi_14616 | 检测 field_info.metadata 中 Json 标记；这类字段不调用 MultiDict.getlist，按 scalar 原始 JSON 字符串交给 Pydantic 解析。 | fastapi/dependencies/utils.py:_is_json_field/_get_multidict_value | 1文件 +10/−2 |
| requests_6644 | 为避免urllib3把 //v:h 重解释为主机端口，将多余开头斜线折叠为单斜线。 | HTTPAdapter.request_url | 1文件 +3/−0 |
| requests_6589 | super_len 对str先做UTF-8编码，再计算字节长度，匹配发送Content-Length。 | utils.super_len | 1文件 +3/−0 |
| rich_3471 | Text.append_tokens在存文本和计算span终点之前调用strip_control_codes，避免构造文本时自动清理与原始token长度不一致。 | Text.append_tokens: content = strip_control_codes(content) | 1文件 +1/−0 |
| rich_3130 | TableDataElement.content始终是带justify的Text；每次on_text把当前样式施加到片段后append_text，保留全部文本与每段style/link，而不是拿plain重建并覆盖。 | TableDataElement.__init__ / on_text | 1文件 +5/−7 |
| rich_3063 | markup.escape补齐尾部未成对反斜杠，让转义内容再嵌入[red]...[/red]后不吃掉闭合标签，plain值及span位置可往返恢复。 | rich.markup.escape: markup.endswith(backslash) | 1文件 +3/−1 |
| rich_3061 | 新增Span.extend/Text.extend_style；逐行按cell位置计算tab空格，延展tab所在全部样式并重拼富文本；Text.tab_size缺省改为None使Console缺省可继承。 | Span.extend / Text.extend_style / Text.expand_tabs / Text.__rich_console__ | 1文件 +67/−24 |

## M3 参数、配置与元数据传播（16题）

已有语义没有穿过所有入口、包含关系、递归层或覆盖优先级。 沿生产者到消费者追踪每一条传递边，补齐继承、合并与覆盖规则。

| ID | 实际工作 | 定位 | 参考补丁规模 |
| --- | --- | --- | --- |
| fastapi_14978 | 增加默认启用的 strict_content_type，沿 FastAPI、APIRouter、APIRoute、include_router 和 get_request_handler 传播并解决覆盖优先级；控制无 Content-Type 请求是否解析 JSON。 | fastapi/applications.py:FastAPI.__init__; fastapi/routing.py:APIRouter.__init__/add_api_route/include_router/APIRoute/get_request_handler | 3文件 +81/−1 |
| fastapi_14482 | ModelField 增加 config 并传给 TypeAdapter；get_model_fields 对普通字段传递父模型 model_config，同时避开 BaseModel/dict/dataclass 自带配置的类型。 | fastapi/_compat/v2.py:ModelField.__post_init__/get_model_fields | 1文件 +25/−7 |
| fastapi_14306 | 提取并缓存端点文件/行号/名称，在 HTTP、WebSocket、请求/响应错误链传递 EndpointContext；异常 __str__ 增路由/源码上下文。 | fastapi/exceptions.py:EndpointContext/ValidationException; fastapi/routing.py:_extract_endpoint_context/serialize_response/get_request_handler/get_websocket_app | 2文件 +121/−16 |
| fastapi_14459 | 依赖扁平化保存 own/parent scopes、call 和依赖树；识别实际 SecurityBase 依赖并以其有效 scopes 生成 OpenAPI 安全要求，代替旧 SecurityRequirement 列表。 | fastapi/dependencies/models.py:_uses_scopes/_security_dependencies; fastapi/dependencies/utils.py:get_flat_dependant; fastapi/openapi/utils.py:get_openapi_security_definitions | 3文件 +49/−22 |
| fastapi_14301 | get_parameterless_sub_dependant 调用 get_dependant 时补 scope=depends.scope，使 route/app-level Depends(scope="function") 生效。 | fastapi/dependencies/utils.py:get_parameterless_sub_dependant | 1文件 +4/−1 |
| fastapi_14246 | 把 validation/serialization 字段分别展开其嵌套模型，再以相同 mode 构建 ModelField，避免所有嵌套模型默认变成 validation。 | fastapi/_compat/v2.py:get_definitions | 1文件 +24/−4 |
| fastapi_13713 | FastAPI 构造器接收并保存 openapi_external_docs，openapi() 传给 get_openapi，再输出 externalDocs 字段。 | fastapi/applications.py:FastAPI.__init__/openapi; fastapi/openapi/utils.py:get_openapi | 2文件 +31/−0 |
| rich_3718 | 将Panel主style与border_style合成，再以stylize_before传给标题和副标题，继承背景同时保留标题显式前景色优先级。 | Panel.__rich_console__: border_style / title_text.stylize_before | 1文件 +3/−4 |
| rich_3882 | PromptBase.on_validate_error打印内部校验信息时传markup=True，使内部带标记错误不受外部Console(markup=False)影响。 | PromptBase.on_validate_error | 1文件 +1/−1 |
| rich_3777 | Console.__init__在force_interactive未显式设置时解释TTY_INTERACTIVE=0/1，再统一计算is_interactive；诊断输出也暴露该环境变量。 | Console.__init__: force_interactive / diagnose.report | 2文件 +12/−1 |
| rich_3296 | Syntax外层Padding的style从纯主题背景改为_get_base_style，使用户background_color覆盖同步作用于空白padding。 | Syntax.__rich_console__: Padding(style=self._get_base_style()) | 1文件 +1/−3 |
| rich_3675 | Console.is_terminal依显式force参数、IDLE/Jupyter、TTY_COMPATIBLE=0/1、FORCE_COLOR非空、file.isatty的顺序决定能力；NO_COLOR由存在性改为非空语义，诊断加入新变量。 | Console.is_terminal / Console.__init__: no_color / diagnose.report | 2文件 +28/−13 |
| rich_3676 | 给Stack和_SyntaxError增加notes字段；从Exception.__notes__提取并传入，Traceback渲染每条带[NOTE]样式的注释。 | Stack.notes / _SyntaxError.notes / Traceback.extract / Traceback.__rich_console__ | 2文件 +12/−1 |
| rich_3518 | Table.add_row自动创建Column时显式传highlight=self.highlight，令动态列与Table级配置一致。 | Table.add_row: Column(_index=index, highlight=self.highlight) | 1文件 +1/−1 |
| rich_3052 | case_sensitive从ask重载及入口传入实例；check_choice按需lower匹配；process_response返回choices中的原始规范值而非用户大小写版本。 | PromptBase.ask / __init__ / check_choice / process_response | 1文件 +29/−5 |
| rich_3105 | 导出HTML模板在code标签加font-family:inherit，让嵌套code沿用pre自定义等宽字体，而非被浏览器默认code字体规则覆盖。 | CONSOLE_HTML_FORMAT: <code style=font-family:inherit> | 1文件 +1/−1 |

## M4 状态、缓存与资源生命周期（15题）

正确行为依赖操作历史、对象共享、缓存键或资源退出时点。 画出状态与所有权，核对初始化/复制/缓存/清理顺序和分支隔离。

| ID | 实际工作 | 定位 | 参考补丁规模 |
| --- | --- | --- | --- |
| fastapi_15023 | docs_src 中 4 个流式端点改为在 with read_image() 内迭代文件，另增加同步 yield from 端点。 | docs_src/stream_data/tutorial002_py310.py:stream_image/stream_image_no_async/stream_image_no_async_yield_from | 1文件 +18/−8 |
| fastapi_14986 | OpenAPI 响应按请求复制 schema 后附 root_path，不把不同请求的 root_path 累积到共享 servers；Swagger UI 内 JSON 转义 <、>、&。 | fastapi/applications.py:FastAPI.setup.openapi; fastapi/openapi/docs.py:_html_safe_json/get_swagger_ui_html | 2文件 +25/−9 |
| requests_7328 | 在把当前响应加入hist之前复制history，消除响应自引用和历史膨胀。 | SessionRedirectMixin.resolve_redirects | 1文件 +1/−2 |
| fastapi_14873 | 把 APIRouter.on_startup/on_shutdown 初始化移到 super().__init__ 之后，防止 Starlette 父类覆盖用户传入的列表。 | fastapi/routing.py:APIRouter.__init__ | 1文件 +11/−10 |
| fastapi_14463 | 把 additional_response.copy() 改为 copy.deepcopy()，防止深层 content/schema 与共享 app-level responses 被就地 merge 影响其他路由。 | fastapi/openapi/utils.py:get_openapi_path | 1文件 +2/−1 |
| fastapi_14099 | 新 AsyncExitStack 中间件保留上传文件；自定义 request_response/websocket_session 把 yield 依赖退出栈包住完整 response ASGI 执行，补异常被吞后的错误处理。 | fastapi/applications.py:build_middleware_stack; fastapi/middleware/asyncexitstack.py; fastapi/routing.py:request_response/websocket_session/get_request_handler | 5文件 +346/−131 |
| fastapi_5624 | 把就地 extend 改为列表相加生成分支副本，避免一条子依赖添加的 scopes 污染兄弟依赖。 | fastapi/dependencies/utils.py:get_dependant | 1文件 +3/−1 |
| fastapi_14419 | 拆分 own/parent OAuth scopes，递归计算依赖及子依赖是否真正使用 scopes；只有使用时才把规范化 scopes 加入 cache_key，保留独立生命周期 scope。 | fastapi/dependencies/models.py:oauth_scopes/cache_key/_uses_scopes; fastapi/dependencies/utils.py:get_dependant/solve_dependencies | 2文件 +42/−20 |
| fastapi_14262 | Depends 新增 scope，Dependant 计算默认 scope 并加入缓存键；使用 function/request 两层 AsyncExitStack，按 scope 选栈并拒绝 request 依赖较短的 function 依赖。 | fastapi/param_functions.py:Depends; fastapi/dependencies/models.py:computed_scope/cache_key; fastapi/dependencies/utils.py:get_dependant/solve_dependencies; fastapi/routing.py:request_response/websocket_session | 10文件 +196/−70 |
| rich_3934 | LiveRender公开last_render_height读取最近_shape；Live.stop只在确实渲染过非零行高时补终止换行。 | Live.stop / LiveRender.last_render_height | 2文件 +17/−3 |
| rich_3905 | Progress.stop与start对称遵守disable标志，不调用live.stop也不在非交互控制台输出空行。 | Progress.stop | 1文件 +4/−3 |
| httpx_3672 | 同步/异步parser、pool、server统一complete→reset；keepalive时读尽body再重置，退出时关闭活动连接并处理KeyboardInterrupt。 | HTTPParser.reset/is_keepalive / Connection._reset / Server.handle_requests / NetworkServer.__exit__ | 7文件 +57/−22 |
| rich_3468 | 把写出提取为_write_buffer，在_check_buffer捕获BrokenPipeError后调用可覆写on_broken_pipe；默认设quiet、把stdout重定向devnull并退出1，防止退出刷新再报错。 | Console._check_buffer / _write_buffer / on_broken_pipe | 1文件 +36/−6 |
| rich_3470 | Console._write_buffer仅在record开启且_buffer_index为0时记录，隔离嵌套capture状态与长期记录缓冲，避免捕获内容被重复累计或污染record。 | Console._write_buffer: self.record and not self._buffer_index | 1文件 +1/−1 |
| rich_2943 | Style.clear_meta_and_links生成已清理的新Style时将_hash设None，不能复制依赖旧meta/link的已缓存hash。 | Style.clear_meta_and_links: style._hash = None | 1文件 +1/−1 |

## M5 算法与结构不变量（12题）

需要保持扫描终止、图遍历覆盖/去环、坐标映射或预算守恒。 明确不变量，处理空输入、边界、别名、双宽/零宽及根节点等反例。

| ID | 实际工作 | 定位 | 参考补丁规模 |
| --- | --- | --- | --- |
| rich_4006 | split_graphemes为前导ZWJ/VS16/零宽控制字符建立零宽span，所有分支推进index；处理尾ZWJ避免越界，并令spans无缝覆盖原字符串。 | split_graphemes: while index < codepoint_count | 1文件 +33/−12 |
| fastapi_14455 | 以 scheme_name 为 key 累积 scopes，稳定去重后再生成 OpenAPI operation_security，避免同 scheme 不同 scopes 产生重复要求条目。 | fastapi/openapi/utils.py:get_openapi_security_definitions | 1文件 +11/−2 |
| fastapi_14266 | 把 SecurityBase 检测与 SecurityRequirement 登记从父节点遍历子依赖分支移到 get_dependant 当前节点入口，修复 app 顶层安全依赖遗漏。 | fastapi/dependencies/utils.py:get_dependant | 1文件 +8/−10 |
| rich_3953 | 字符宽度查找显式把C0/C1控制字符置零，并重写二分边界；_cell_len以索引循环替代next(iterator)，尾ZWJ可安全跳过而不抛StopIteration。 | get_character_cell_size / _cell_len | 1文件 +16/−10 |
| rich_3935 | Table._get_padding_width纳入collapse_padding与pad_edge的首尾列规则，修正各列测量额外宽度，令三个固定宽列间距一致且边缘无多余padding。 | Table._get_padding_width | 1文件 +13/−3 |
| fastapi_15763 | include_router 无 prefix 时遍历 route/context，按有效 APIRoute 路径或包装后的 Starlette 路径判断空路径，不再使用原始叶子路径。 | fastapi/routing.py:APIRouter.include_router/_iter_routes_with_context | 1文件 +10/−3 |
| rich_3180 | chop_cells改成从左到右按cell宽度折行；divide_line分清当前余宽不足和整行也容不下的长词，输出codepoint断点，防止CJK在双宽字符中间被裁掉。 | chop_cells / divide_line | 2文件 +89/−38 |
| rich_3782 | PaddingProperty将赋值规范化成四元组；measure和render共享_padding，_get_syntax在代码宽度预算中扣除左右padding并钳制非负，word_wrap在真正内容区折行。 | PaddingProperty / Syntax.__rich_measure__ / __rich_console__ / _get_syntax | 1文件 +24/−5 |
| rich_3772 | Traceback.extract传递共享visited异常集合，避免异常组重复递归；直接cause须不等于自身；组内遍历时不再无条件沿cause/context回走。 | Traceback.extract: _visited_exceptions / grouped_exceptions / __cause__ | 1文件 +29/−16 |
| rich_3521 | Segment._split_cells从比例估算位置起，比较前缀cell宽与cut，按误差向前或向后移动；切穿双宽字符时两侧补空格，保证左右宽度之和。 | Segment._split_cells: out_by | 1文件 +16/−19 |
| rich_3506 | Segment._split_cells的初始codepoint估算从len(text)-1改为len(text)后减1并下限置0，避免估算越过可单向搜索的目标位置。 | Segment._split_cells: pos = int((cut / cell_length) * len(text)) - 1 | 1文件 +15/−2 |
| rich_3480 | Text.append和append_text对源_spans取快照再extend，防止source与target为同一对象时迭代自己不断追加的列表而无限循环。 | Text.append / Text.append_text: text._spans.copy() | 1文件 +2/−2 |

## M6 类型反射与动态调用解析（17题）

类型声明、装饰/partial包装与实际被执行对象或求值命名空间不一致。 分层解包类型和callable，核对globals、注解、实例/类、同步/异步/生成器。

| ID | 实际工作 | 定位 | 参考补丁规模 |
| --- | --- | --- | --- |
| fastapi_13920 | analyze_param 在分析前用 is_typealiastype 检测别名，读取 __value__ 解开 TypeAliasType。 | fastapi/dependencies/utils.py:analyze_param | 1文件 +4/−0 |
| fastapi_14794 | 在 Response/Request 等特殊注入类型分支前检查 depends is None，使显式 Depends 的返回值按普通依赖解析。 | fastapi/dependencies/utils.py:analyze_param | 1文件 +3/−2 |
| fastapi_12942 | field_annotation_is_complex 遇到 Annotated 时继续分析第一个类型参数，正确把 tagged discriminated union 识别为 body。 | fastapi/_compat.py:field_annotation_is_complex | 1文件 +3/−0 |
| fastapi_14512 | 增加 FieldInfo 的 asdict 兼容实现，拆开 annotation、metadata、attributes 后重新创建 Field；避免旧 FieldInfo 元数据再次包裹导致 discriminator/Tag 错位。 | fastapi/_compat/v2.py:asdict/ModelField.__post_init__ | 1文件 +48/−4 |
| fastapi_11355 | Python >=3.10 时 inspect.signature(call, eval_str=True)，同时用于参数和返回注解提取。 | fastapi/dependencies/utils.py:get_typed_signature/get_typed_return_annotation | 1文件 +9/−2 |
| fastapi_14485 | 抽取 _get_signature；eval_str=True 遇 NameError 时回退不求值签名，使仅类型检查可见且运行期不用的注解不阻止依赖解析。 | fastapi/dependencies/utils.py:_get_signature/get_typed_signature/get_typed_return_annotation | 1文件 +13/−6 |
| fastapi_14458 | 在 generator/async-generator/coroutine 检测中，解包对象若是 class 则提前返回 False，避免把实例的 async __call__ 当作类构造行为。 | fastapi/dependencies/models.py:Dependant.is_gen_callable/is_async_gen_callable/is_coroutine_callable | 1文件 +6/−1 |
| fastapi_14448 | 新增 _impartial/_unwrapped_call，并在原函数、wrapper、partial、实例 __call__ 多层检测同步/异步/生成器；优先处理 async generator。 | fastapi/dependencies/models.py:_impartial/_unwrapped_call/Dependant.*callable; fastapi/dependencies/utils.py:_solve_generator | 2文件 +82/−24 |
| fastapi_9753 | 依赖的 _unwrapped_call 在 inspect.unwrap 后遇 functools.partial 时取 .func，再进行调用类型识别。 | fastapi/dependencies/models.py:Dependant._unwrapped_call | 1文件 +5/−2 |
| fastapi_14430 | serialize_sequence_value 同时识别 typing.Union 和 types.UnionType，选取非 None 分支序列类型。 | fastapi/_compat/v2.py:serialize_sequence_value | 1文件 +2/−2 |
| fastapi_5077 | inspect.unwrap 后使用原函数 __globals__，同时修复参数和返回类型的 ForwardRef 求值。 | fastapi/dependencies/utils.py:get_typed_signature/get_typed_return_annotation | 1文件 +4/−2 |
| fastapi_9555 | 新增缓存的 _unwrapped_call=inspect.unwrap(call)，以原函数及其 __call__ 判定 coroutine/generator/async generator。 | fastapi/dependencies/models.py:Dependant._unwrapped_call/is_*_callable | 1文件 +14/−8 |
| fastapi_9425 | ForwardRef 求值若得到 NoneType，返回 None，使无响应体状态码可使用 -> "None"。 | fastapi/dependencies/utils.py:get_typed_annotation | 1文件 +2/−0 |
| rich_3894 | _get_signature对__qualname__验证str类型，失败依次回退到__name__及调用方name，防止slots/lru_cache描述符进入字符串处理。 | Inspect._get_signature: qualname | 1文件 +4/−0 |
| fastapi_14297 | serialize_sequence_value 对 Union 去掉 NoneType 后选取真实序列 origin，再转换收集到的文件字节值。 | fastapi/_compat/v2.py:serialize_sequence_value | 1文件 +7/−0 |
| rich_3472 | pretty遍历dataclass字段时除field.repr外再检查hasattr，跳过init=False且没有设置实例属性的字段，防止getattr异常。 | rich.pretty.traverse: iter_attrs / fields(obj) | 1文件 +3/−1 |
| rich_3006 | auto_rich_repr把param.default == param.empty改为is，避免用户自定义__eq__把真实默认实例误报成inspect.Parameter.empty。 | rich.repr.auto: auto_rich_repr / param.default is param.empty | 1文件 +1/−1 |

## M7 外部协议与依赖兼容（14题）

Python对象协议、继承分派或外部库/版本契约与当前实现不匹配。 识别真正能力和版本边界，在兼容矩阵中保持两侧契约。

| ID | 实际工作 | 定位 | 参考补丁规模 |
| --- | --- | --- | --- |
| fastapi_14964 | 用 typing_extensions.deprecated 标注两种响应类，指定 FastAPIDeprecationWarning、消息和 stacklevel，保持各自 render 可用并更新类文档。 | fastapi/responses.py:UJSONResponse/ORJSONResponse | 1文件 +44/−8 |
| requests_7505 | 把动态代理的 read 能力检测集中到 has_read，并覆盖参数编码、multipart 和 body 三条路径。 | _types.has_read / RequestEncodingMixin._encode_params/_encode_files / PreparedRequest.prepare_body | 2文件 +9/−7 |
| requests_7502 | 文件上传接受通过 __getattr__ 暴露 read 的代理对象，在 Protocol isinstance 之外补 hasattr。 | RequestEncodingMixin._encode_files | 1文件 +3/−1 |
| requests_7433 | 流式 body 检测为 Iterable 或拥有 __iter__，恢复代理文件在307跳转后的流式重放。 | PreparedRequest.prepare_body | 1文件 +3/−3 |
| rich_4077 | 为FileProxy补齐文件对象的isatty方法，结果直接委托底层流，保持包装前后终端探测协议一致。 | FileProxy.isatty | 1文件 +3/−0 |
| fastapi_14851 | FastAPI 自行实现启动/关闭 handler、默认 lifespan、同步 generator 的异步包装；不再向 Starlette 传递被移除的旧事件参数。 | fastapi/routing.py:_AsyncLiftContextManager/_wrap_gen_lifespan_context/_DefaultLifespan/APIRouter.__init__/_startup/_shutdown/add_event_handler | 1文件 +153/−4 |
| fastapi_14609 | 删除 v1/main/may_v1/model_field 兼容层和临时参数模块，收敛到 v2 接口，更新字段/编码/Schema/路由调用，遇到 v1 模型显式抛 PydanticV1NotSupportedError。 | fastapi/_compat/*; fastapi/dependencies/utils.py; fastapi/encoders.py; fastapi/routing.py; fastapi/utils.py | 20文件 +192/−1855 |
| fastapi_14372 | Depends/Security 改为 frozen dataclass；分析注解需要补 dependency 时使用 dataclasses.replace，避免修改冻结实例。 | fastapi/params.py:Depends/Security; fastapi/dependencies/utils.py:analyze_param | 2文件 +4/−3 |
| fastapi_14186 | 新增 may_v1 条件兼容层：Python >=3.14 以 dummy 类型及 v2 API 替代 v1 导入；其余路径按需延迟导入 v1，更新类型判断、编码器和 schema 聚合。 | fastapi/_compat/may_v1.py; fastapi/_compat/main.py; fastapi/encoders.py; fastapi/dependencies/utils.py; fastapi/utils.py | 10文件 +266/−98 |
| fastapi_14077 | 不再导入被弃用的 HTTP_422_UNPROCESSABLE_ENTITY 名称，生产异常处理用整数 422，OpenAPI 用字符串 "422"，示例同步改动。 | fastapi/exception_handlers.py:request_validation_exception_handler; fastapi/openapi/utils.py:get_openapi_path | 3文件 +5/−6 |
| fastapi_14605 | 新增 UserWarning 子类 FastAPIDeprecationWarning，替换参数、Pydantic v1、操作 ID 等各入口的默认被忽略 DeprecationWarning。 | fastapi/exceptions.py:FastAPIDeprecationWarning; fastapi/params.py; fastapi/temp_pydantic_v1_params.py; fastapi/dependencies/utils.py; fastapi/routing.py; fastapi/openapi/utils.py; fastapi/utils.py | 7文件 +28/−16 |
| fastapi_14583 | 在端点参数、response_model、additional responses 三处检测 v1 并发出带上下文与 stacklevel 的 DeprecationWarning。 | fastapi/dependencies/utils.py:get_dependant; fastapi/routing.py:APIRoute.__init__ | 2文件 +24/−0 |
| requests_6757 | 检测urllib3主版本，只有2.x才将str按UTF-8字节计算Content-Length，1.x沿用原语义。 | compat.is_urllib3_1 / utils.super_len | 2文件 +16/−1 |
| requests_6629 | 显式委派CompatJSONDecodeError.__reduce__，绕过IOError优先的MRO，保存反序列化所需全部构造参数。 | JSONDecodeError.__reduce__ | 1文件 +10/−0 |

## M8 新能力与架构实现（8题）

须建立新的执行/表示子系统、公共能力组合或改变既有架构的组织方式。 先拆出数据模型和执行路径，再贯通接口、实现、交付和回归约束。

| ID | 实际工作 | 定位 | 参考补丁规模 |
| --- | --- | --- | --- |
| fastapi_15661 | 新增 Typer 发布 CLI：按 major/minor/patch 增版本、校验唯一版本赋值、重写版本和发布说明、提取指定版本正文。 | scripts/prepare_release.py:parse_version/bump_version/update_release_notes/get_release_notes_body/prepare | 1文件 +216/−0 |
| fastapi_15030 | 新增 EventSourceResponse/ServerSentEvent/线格式编码器；连接路由生成器、Pydantic 逐项验证和 JSON 编码、同步线程池、异步生产者/心跳、响应头及 OpenAPI itemSchema。 | fastapi/sse.py; fastapi/routing.py:get_request_handler/APIRoute.__init__; fastapi/openapi/utils.py:get_openapi_path | 9文件 +525/−35 |
| rich_4070 | 重排Console、RichHandler等9个模块的依赖加载时机；类型专用导入移入TYPE_CHECKING，运行时依赖移入使用分支，替换Path/inspect.isclass，移除死导入及_svg_hash。 | Console._collect_renderables / _caller_frame_info / RichHandler.emit / Theme.from_file | 9文件 +71/−57 |
| rich_3930 | 以CellTable/CellSpan及21份Unicode版本表替代单个CELL_WIDTHS；新增版本解析/环境选择/懒加载，按ZWJ/VS16组合字素；让长度、切分、裁剪、折行共同使用字素span表示。 | CellTable / split_graphemes / split_text / set_cell_size / chop_cells / rich._unicode_data.load | 26文件 +12170/−544 |
| fastapi_15800 | 新增 app/router.frontend 公共 API、静态文件/HTML fallback、媒体类型导航识别、最长前缀匹配、低优先路由树与缓存；正常 API/405/斜杠重定向优先，处理路径/符号链接和 fallback 校验。 | fastapi/applications.py:FastAPI.frontend; fastapi/routing.py:_FrontendStaticFiles/_FrontendRoute/_FrontendRouteGroup/APIRouter.frontend/app/_match_low_priority | 8文件 +649/−5 |
| fastapi_15785 | 新增 frozen RouteContext 代理与 iter_route_contexts，统一直接路由和包含树的有效路径/元数据视图；OpenAPI 路由与 webhook 接受该视图用于过滤等操作。 | fastapi/routing.py:RouteContext/iter_route_contexts; fastapi/openapi/utils.py:_get_api_route_for_openapi/get_fields_from_routes/get_openapi | 2文件 +66/−15 |
| fastapi_15745 | include_router 从克隆展开改为保留原始对象的 _IncludedRouter 树；_RouterIncludeContext/_EffectiveRouteContext 分离对象身份和有效元数据；重建匹配/handle/url_path_for、递归循环检测与路由版本/OpenAPI 缓存失效。 | fastapi/routing.py:_populate_api_route_state/_RouterIncludeContext/_EffectiveRouteContext/_IncludedRouter/APIRoute.handle/APIRouter.include_router; fastapi/applications.py:openapi; fastapi/openapi/utils.py:_iter_routes_with_context | 4文件 +939/−291 |
| rich_3486 | Python 3.11+用f_lasti/co_positions抽取起止行列并存入Frame.last_instruction；Traceback渲染将范围传入Syntax，新增style_before控制叠加优先级及默认错误范围样式。 | Frame.last_instruction / Traceback.extract / Syntax.stylize_range / Syntax._apply_stylized_ranges | 3文件 +74/−24 |

## 逐题的解释限制

以下说明保留题面缺失、版本依赖、参考补丁超出可见验收等情况；不是另设“问题题”类别，也不等同已实跑失败。

**fastapi_15661**：标题标为 CI，实际是新增可执行 CLI；题面主体未自足列出版本文件格式、CLI签名和发布说明约束，但 AI Disclaimer 给了 asyncer PR 608 参考链接。该外链未在本轮完整审读，不能断言不存在实现线索。

**fastapi_15588**：同时涉及协议字段的词法边界；并非需要实现完整 SSE。

**fastapi_15589**：表面是输入拒绝规则，根因是字段别名转换与 extra 参数回填的两个阶段不一致。

**fastapi_15030**：大范围能力实现，兼含表示转换、并发与流生命周期；题面只有标题和发布说明，未写心跳、raw_data、keepalive、混合类型及 OpenAPI 等细节；5 个 docs_src 示例与核心实现同时变更。

**fastapi_14962**：优化表示管线而非普通格式修补；题面未阐明默认类/显式类分界，参考测试要求默认路径不调用 json.dumps。

**fastapi_15280**：严重题面不足：标题说支持 app.vibe，实际评测只要求特殊异常，不能据标题推断实现 LLM 能力；新接口属于局部异常契约，不能算完整能力工程。

**fastapi_14964**：题面还要求从 fastapi[all] extras 移除依赖，但公开 patch 仅触及 responses.py；不能把题面提到的依赖元数据修改量算进本题生产补丁。

**fastapi_15023**：标题标为 docs，实际是可执行示例的资源生命周期修复和一个路由扩展；仅 docs_src，不触及 fastapi/ 包；当前官方评分过滤允许此路径。

**fastapi_14986**：一个任务包含两个独立机制：共享状态隔离为主，HTML 嵌入编码为次；仅按 OpenAPI 主题会把它们错误合并。

**fastapi_14978**：核心难点是所有创建/包含路径及 DefaultPlaceholder 的优先级，不止改一个 if；题面只有标题，详细继承契约主要在代码文档/公开测试。

**requests_7505**：不能只修 _encode_files；动态 __getattr__ 不保证 runtime_checkable Protocol 检查命中。

**requests_7502**：协议识别问题，补丁虽短但需理解 NamedTemporaryFile 动态代理。

**requests_7433**：补丁局部，验收横跨 stream detection 与 redirect/rewind。

**requests_7427**：环境变量是触发条件；应保留合法子域名行为。

**requests_7315**：与 requests_6644 方向相反，必须按当前base_commit的契约判断，不能复用固定修法。

**requests_7328**：关键在对象引用和操作顺序，不是简单列表切片语法。

**requests_7309**：既是语法解析也有局部边界；主机制按字符串到字典转换归类。

**requests_7205**：Python3.11改变上游返回值；主变更是值的有效性规则，版本依赖作为次标签。

**rich_4070**：跨9个生产文件；目标是冷启动性能，但公开test_patch仅调整getpass monkeypatch和Console导入，没有计时oracle；不能以功能测试通过推断达到题面1.5x/1.75x。

**rich_4077**：题面仅短标题及issue链接，具体期望由公开回归用例补足。

**rich_4079**：题面仅短标题及issue链接；公开用例带inline_code_theme/lexer，以ANSI快照验证样式。

**rich_4076**：题面仅短标题及issue链接；公开用例覆盖空串、首换行、多尾换行。

**rich_4075**：题面另提Drop Python 3.8；提供的生产patch实际行为变更仅空print分支，余为格式。

**rich_4006**：题面仅短标题及issue链接；关键是终止性、索引覆盖和宽度守恒，不能只归为Unicode显示。

**fastapi_14258**：这是直接自引用检查，尚非后续路由重构中的间接循环图检测。

**fastapi_14851**：依赖版本 API 适配为主，生命周期调度为次；新增 153 行，需兼顾同步/异步 handler 与旧调用方式。

**fastapi_14873**：生命周期字段初始化顺序问题；是 14851 后续回归，不能把两题当独立分布样本。

**fastapi_13920**：依赖 typing_inspection 的类型对象识别；修复类型别名包裹的 Depends 等元数据。

**fastapi_14794**：不只 Response：公开测试同时涉及 Request/BackgroundTasks；核心是注解派发优先级。

**fastapi_14786**：单点词法规范化，可从题面直接定位。

**fastapi_14492**：标题为 docs，但评测对象是 docs_src 中生成 OpenAPI 元数据的 Python 示例常量；仅一行；当前官方评分过滤允许此路径。

**fastapi_14479**：生产变更只改变诊断消息，未扩大支持类型；精确错误文本属于可评分行为。

**fastapi_14463**：根因是浅复制后的别名共享与状态污染，不是 Union 类型推断本身。

**fastapi_12942**：Annotated 嵌套 + Union 的递归类型语义；会影响 Query/body 自动选择。

**fastapi_14099**：大范围生命周期重排（+346/-131），牵涉中间件次序、contextvars、StreamingResponse、WebSocket、异常传播；题面简短但正文说明了主要退出时点。

**fastapi_14609**：大范围迁移（20 文件，+192/-1855），不是单个异常分支；含删除两个 docs_src 示例，仍须保持 v2 行为。

**fastapi_14372**：对象 hash/不可变协议适配；只改 frozen 会令旧赋值路径失败，必须配套改写。

**fastapi_14349**：兼含递归结构不变量：同名键在 JSON Schema 的不同结构位置具有不同含义。

**fastapi_14512**：反射元数据组合为主，依赖版本适配为次；为旧 Pydantic（尚无 FieldInfo.asdict）自带兼容实现；题面仅标题和关联链接。

**fastapi_14487**：标题为 docs，实际改可执行异常响应序列化；仅 docs_src；当前官方评分过滤允许此路径。

**fastapi_11355**：版本条件控制语言反射行为；后续 14485 修复 TYPE_CHECKING 的 NameError 回归。

**fastapi_14485**：是 11355 的交互边界修复；只捕获 NameError，不能把所有求值错误吞掉。

**fastapi_14482**：核心是 arbitrary_types_allowed 等配置跨模型到字段适配器的传播与例外，兼含类型元数据展开。

**fastapi_14306**：跨层上下文传播为主；兼含缓存和安全边界：只增强日志字符串，不在 HTTP 响应泄露路径；+121/-16。

**fastapi_14458**：反射对象层级边界；是 14448 后续回归。

**fastapi_14459**：跨树层级传播为主，兼含安全依赖对象识别；父层声明 scopes 而叶安全 scheme 自身无 scopes 是关键边界。

**fastapi_14448**：大组合空间（+82/-24），任何一层 async 都影响调度，不能只解包到最底层；与 9753/9555/14458 属同一连续演进。

**fastapi_14455**：有序聚合/去重算法；需维持不同 scheme 分组及 scope 顺序。

**fastapi_9753**：生产只 5 行新增/2 行删除，公开测试组合较多；尚未覆盖后续 partial/wraps 多层混合解包。

**fastapi_14430**：是 14297 之后补 Python 3.10 | 语法的遗漏；不能把两题视作两种机制。

**fastapi_5077**：定位点是注解来源命名空间；与仅判断 async 的 wraps 任务机制不同但同属 Python 反射。

**fastapi_9555**：解决 wrapper 隐藏 async/generator 的运行方式；题面提缓存 reawaitable 只是使用背景，不要求实现 coroutine 缓存。

**fastapi_14361**：只有一条额外保护条件；与用户属性同名 $ref 的 14349 边界不同。

**fastapi_13207**：核心是同一模型的输入/输出表示不等价；题面演示不同配置输出差异，但实现要同步两处 mode 选择。

**fastapi_14360**：输入 wire key、模型 field name、validation alias 是不同命名空间；后续 14371/15589 再扩展此流程。

**fastapi_14356**：容器表示转换和基数保真；Cookie 重名语义由来源容器限制。

**fastapi_5624**：核心是递归分支共享可变对象污染，兼含传播；题面仅链接 defect。

**fastapi_14419**：缓存等价关系为主，递归可达性为次；公开测试要求同一无 scopes DB session 只创建一次，而有不同 scopes 的用户依赖仍执行两次。

**fastapi_14301**：丢失一条参数转发边造成生命周期差异；无需重新设计退出栈。

**fastapi_14262**：大范围生命周期能力（10 文件，+196/-70）；标题对 request 的退出时点表述与正文相冲突，应按正文/参考实现：function 在发送前退出，request 在发送后退出。

**fastapi_14266**：递归遍历的根/子节点不对称；移动逻辑到节点自身以覆盖根，不只是添加一次顶层特判。

**fastapi_14246**：跨递归模型图的 mode 传播；比单个 schema 字段值修改更广。

**fastapi_14186**：大范围版本矩阵适配（10 文件，+266/-98）；避免仅安装 v2 仍因导入 v1 触发警告，需保留旧 Python 的混用迁移能力。

**fastapi_9425**：语言级 None/NoneType/字符串注解表示差异；不是 HTTP 响应体清空流程错误。

**fastapi_13713**：现有 OpenAPI 生成管道增加元数据传递：构造器、实例保存、get_openapi 新参数及 externalDocs 输出全部需连接；不是只接通一个原本已有的 get_openapi 参数。

**fastapi_14077**：标题是依赖升级，但公开 patch 未改 pyproject 等版本范围，只修正兼容两代 Starlette 的生产代码和示例。

**rich_3718**：题面说明历史多次回归；公开用例分别验证border颜色与panel背景组合。

**rich_3934**：题面仅短标题及issue链接；条件依赖先前渲染历史，不能视为静态空输入分支。

**rich_3953**：题面仅短标题和两个issue链接；多处变更同时涉及字符规则与扫描终止边界。

**rich_3944**：题面仅fix fonts和issue链接；参考patch是一行默认值变化，并不需重建字素模型。

**rich_3942**：规格不足：题面只有Update to markdown styles，未给出具体颜色、对齐、box或空行目标；参考patch和更新快照提供了额外事后规格。；大ANSI/演示快照按结构审阅，没有逐字节穷举。

**rich_3938**：题面仅短标题及issue链接；须同时保持控制segment和有无终止换行信息。

**rich_3935**：题面仅短标题及issue链接；公开用例Table.grid(padding=(0,1))输出aaa aaa aaa；改动本质为测量/渲染宽度预算一致。

**rich_3930**：大范围：26个生产文件，patch约308 KB，主要为21份版本宽度与narrow_to_wide数据表；已审阅加载/算法/用例和表结构，未逐行核对每个Unicode范围。；官方当前保护规则未禁止这些新增.py；交付时新增文件须被最终git diff收集。；题面明确能力方向但未逐条指定版本表，表的来源为题面公开wcwidth数据。

**httpx_3672**：横跨7个文件且有同步/异步镜像；公开验证patch只改tests/test_parsers.py的complete调用，不能把该测试通过等同全部连接管理行为验证。

**fastapi_14791**：数据契约描述与运行时对象对齐；不是新增异常返回字段。

**fastapi_15800**：大范围能力实现（+649/-5），题面只有签名和模板，未给 fallback/优先级/导航/路径边界等设计；6 个 docs_src 示例之外核心新增很多结构。

**fastapi_15785**：是 15745 树状路由重构后的公共可遍历接口补充；含对象代理协议，不能归为只加函数名的局部接口。

**fastapi_14371**：4 文件多入口表示约定同步；题面称仅支持字符串 validation_alias，不能擅自扩成 AliasPath/AliasChoices 全实现。

**fastapi_14605**：7 文件横向一致变更，改动量小但入口分散；核心是 Python warnings 过滤协议与用户可见性。

**fastapi_15763**：15745 树状包含后出现的边界；相同叶子 path="" 可因包含前缀而有效，需使用组合后路径。

**fastapi_14583**：这是迁移提示阶段，仍要保持旧模型正常工作；后续 14605 改警告类别，14609 才移除支持。

**fastapi_15745**：最大结构重构题（+939/-291），需要运行时路由、子类 hook、动态加路由、重复包含前缀、OpenAPI、反向 URL 等共同保持等价；题面明确破坏 router.routes 平面列表假设，不是仅改 deepcopy 或 identity 判断。

**fastapi_14297**：先修 typing.Optional，后续 14430 才补 | Union；与单纯 list 数据序列化不同，触发原因在注解类型。

**fastapi_14303**：与 14356 同一机制，来源容器不同；需要保持单值旧行为。

**fastapi_13537**：状态区别是已声明/已处理与值是否非 None，不能拿输出字典是否有 key 代替；局部缺省值边界。

**fastapi_14953**：协议表示对齐为主，Pydantic 上游能力适配为次；题面内容充分，兼顾 validation/serialization bytes 配置；scripts/docs_src 并非评分保护路径。

**fastapi_13786**：4 个核心文件/多个安全类横向契约一致修改；题面自注部分过时，描述的 not_authenticated_status_code 构造参数不在最终参考实现，实际扩展点是子类覆盖错误方法；标题数字替换不足以完成。

**fastapi_14616**：逻辑序列与传输层单个 JSON 字符串不同；不只 Form，公开测试也含 Query/Header/Cookie。

**fastapi_11194**：维护参数顺序不影响语义的不变量；局部误用首项元数据，不需重写 Form 解析架构。

**requests_6757**：标题写测试但参考补丁修改生产兼容层；与 requests_6589 构成版本条件链。

**requests_6629**：对象序列化协议与多继承；可同时标记表示转换，但修复关键是协议分派。

**requests_6644**：和后来的 requests_7315 正好相反，不能脱离历史上下文判断规范化是否正确。

**requests_6589**：后来 requests_6757 限定此行为仅适用于urllib3 2.x。

**requests_6592**：数据表变更也是有效任务；不需算法或系统设计。

**rich_3180**：两生产文件；重要不变量是字符不丢失、断点不切穿双宽字、折行后继续累计下一词位置。

**rich_3486**：3个生产文件；新增版本相关能力；公开用例只验证last_instruction存在及基本形状，没有覆盖完整渲染快照。

**rich_3777**：公共用例有一段构造非交互Console但缺相邻assert；不能据此扩大声称所有优先级都被oracle直接覆盖。

**rich_3468**：Unix进程/管道语义；公开回归通过python -m rich | head -1检查退出状态，Windows跳过。

**rich_3782**：题面仅短标题及issue链接；公开用例是在24列宽减去两侧各3列后折行。

**rich_3772**：题面仅两个issue链接；公开新增用例覆盖raise e from e，参考patch范围还包含ExceptionGroup去环，覆盖范围需区分。

**rich_3675**：题面只明确新TTY_COMPATIBLE变量；参考patch同时调整NO_COLOR/FORCE_COLOR空值语义，这些属于附带兼容行为。；公开test_patch大部分是类型标注/格式调整，实质新增oracle为TTY_COMPATIBLE组合用例。

**rich_3676**：题面仅Notes / Add support for Exception.__notes__；公开用例验证提取列表，未直接断言注释渲染。

**rich_3535**：任务强调性能，但公开用例只验证字符集合分类，未计时；主机制是fast path识别谓词修正。

**rich_3521**：题面仅短标题及外仓issue链接；公开用例覆盖每个cell位置与混合宽输入，不能仅靠string长度估算。

**rich_3506**：与rich_3521同一函数但不同历史基线；这一题主要修正估算，后续题才改成完整双向搜索。

**rich_3480**：题面仅短标题及issue链接；关键是别名导致的遍历集合自增长，并非普通字符串追加。

**rich_3472**：题面仅短标题及issue链接；元数据声明存在不等于实例属性存在。

**rich_3471**：题面仅短标题及issue链接；关键是清理后的内容与offset共同变化。

**rich_3470**：公开用例替换旧预期：capture中的内容从此不属于export_text；属于语义变更，不能用旧混合record/capture假设。

**rich_3469**：题面仅短标题及issue链接；公开用例比较含空cell与完整cell的行数。

**rich_3052**：单文件多入口和重载同步；不能只改成员检查，否则新参数不可达或返回值不规范。

**rich_2943**：涉及派生值缓存与对象内容一致性；不是单纯样式传递。

**rich_2725**：公开用例用各不相同的字符区分头/体/脚并覆盖show_header/show_footer组合。

**rich_3105**：题面为Fix #3104且只引用issue，具体CSS规范需外链或公开参考变更；test_patch为HTML字符串快照，没有浏览器实际字体渲染。

**rich_3130**：一生产文件；同时修正token.attrs.get的style键。与4079同类信息保真但基线不同：本题解决覆盖/丢片段，4079解决对既有Text再次stylize导致的样式覆盖。

**rich_3043**：题面提到pre-commit更新，但提供生产patch只有HTML模板；不应把题面附带工程说明当必要修复目标。

**rich_3064**：题面仅短标题及issue链接；公开用例只含header和未完成separator，强调可缺失结构。

**rich_3063**：题面仅短标题及issue链接；虽然实现是局部分支，核心契约是escape→markup parse的往返信息保真。

**rich_3061**：一生产文件但含多函数联动；核心为文本长度变化时富文本span坐标和双宽cell坐标同步更新。


## 验收条件的交叉复核

完整记录见 [oracle-caveats.json](oracle-caveats.json)：2处公开required与互斥skip的静态冲突（rich_3486、fastapi_14186），20项依赖/平台/fixture条件候选，另3项候选已排除。它们仍保留原机制主分类，不从129题目录中擅自剔除，也不标成已观测的线上失败。精确基线URL、关键源摘录和完整下载指纹保存在记录中。
