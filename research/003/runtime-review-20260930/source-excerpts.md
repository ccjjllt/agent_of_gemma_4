# Runtime source excerpts

Copied 2026-09-30 from previously extracted official wheels; current-wheel-comparison.json verifies all compared Python files are byte-identical to newly downloaded public wheelhouse packages. Original line numbers preserved. README is the 2026-09-28 archive, not newly downloaded.

## swegemma/harness/agent_runner.py

SHA-256: `dff7d2505340633c3fe50ba7073f2609fb2b41becd69c356d9af25cc4c1f98dd`

Original lines 319–346:
```text
 319         tools = context.create_tools()
 320 
 321         # Compile agent submission
 322         script_timeout = (
 323             config.harness.command_timeout_seconds
 324             if config.harness.command_timeout_seconds is not None
 325             else 300
 326         )
 327         code_executor = AdkSandboxCodeExecutor(
 328             sandbox=context.sandbox,
 329             timeout_seconds=script_timeout,
 330             start_time=context.task_start_time,
 331             max_time_minutes=config.budget.time_minutes,
 332             budget_check_fn=context.check_budget,
 333         )
 334         agent: BaseAgent = compile_submission(
 335             submission_dir=config.submission_dir,
 336             tool_registry=tools,
 337             model_registry=config.models,
 338             code_executor=code_executor,
 339             script_timeout=script_timeout,
 340             limits=config.limits,
 341             generation_constraints=config.generation_constraints,
 342             adapter_manifest=config.adapter_manifest,
 343             adapter_resolver_fn=lambda base, info: resolve_swegemma_adapter(
 344                 base, info, config.models
 345             ),
 346         )
```

Original lines 403–441:
```text
 403         app_kwargs: dict[str, Any] = {
 404             'name': 'swegemma_eval',
 405             'root_agent': agent,
 406             'plugins': [display_plugin, retry_plugin],
 407         }
 408         if config.context_cache_config is not None:
 409             app_kwargs['context_cache_config'] = config.context_cache_config
 410         if config.events_compaction_config is not None:
 411             app_kwargs['events_compaction_config'] = config.events_compaction_config
 412 
 413         app = App(**app_kwargs)
 414         runner = Runner(app=app, session_service=session_service)
 415 
 416         initial_prompt = build_agent_prompt(
 417             task=task,
 418             config=config,
 419             workspace_tree=workspace_tree,
 420             enable_sandbox_testing=enable_sandbox_testing,
 421         )
 422 
 423         user_message = genai_types.Content(
 424             role='user',
 425             parts=[genai_types.Part(text=initial_prompt)],
 426         )
 427 
 428         # Execute agent within time and turn budgets
 429         max_llm_calls = (
 430             config.budget.turns if config.budget.turns is not None else 500
 431         )
 432 
 433         timeout_sec = (
 434             config.budget.time_minutes * 60
 435             if config.budget.time_minutes is not None
 436             else None
 437         )
 438         disp_context = display_mgr if display_mgr else contextlib.nullcontext()
 439         timeout_context = (
 440             asyncio.timeout(timeout_sec)
 441             if timeout_sec is not None
```

Original lines 493–528:
```text
 493                 context.set_phase('Agent Loop')
 494                 context.set_activity('Starting agent loop...')
 495                 context.start_agent_session()
 496                 code_executor.start_time = context.agent_start_time
 497 
 498                 async with timeout_context:
 499                     max_nudges = 3
 500                     consecutive_nudges = 0
 501                     current_message = user_message
 502 
 503                     while True:
 504                         remaining_turns = max_llm_calls - context.llm_calls_used
 505                         if remaining_turns <= 0:
 506                             agent_error = (
 507                                 f'Agent exceeded turns budget ({config.budget.turns} turns)'
 508                                 if config.budget.turns is not None
 509                                 else 'Agent exceeded maximum allowed LLM turns'
 510                             )
 511                             logger.warning(
 512                                 'Task %s: LLM turn budget exhausted.', task.instance_id
 513                             )
 514                             break
 515 
 516                         run_config = RunConfig(max_llm_calls=remaining_turns)
 517                         last_event_finish_reason = None
 518                         last_assistant_text = ''
 519                         turn_has_tool_call = False
 520                         prev_calls = context.llm_calls_used
 521                         async for _event in runner.run_async(
 522                             user_id='eval_user',
 523                             session_id=session.id,
 524                             new_message=current_message,
 525                             run_config=run_config,
 526                         ):
 527                             # Process event in case mock runner bypassed ADK plugins
 528                             context.handle_adk_event(_event, agent_name=agent.name)
```

Original lines 635–673:
```text
 635                         if context.patch_submitted:
 636                             context.set_activity('Patch submitted')
 637                             break
 638 
 639                         if (
 640                             timeout_sec is not None
 641                             and context.elapsed_seconds > timeout_sec
 642                         ):
 643                             if not agent_error:
 644                                 agent_error = f'Agent exceeded session timeout ({config.budget.time_minutes} min)'
 645                             break
 646 
 647                         if (
 648                             context.budget.tool_calls is not None
 649                             and context.tool_calls_used >= context.budget.tool_calls
 650                         ):
 651                             if not agent_error:
 652                                 agent_error = (
 653                                     f'Agent exceeded tool call budget ({context.budget.tool_calls} calls)'
 654                                 )
 655                             logger.warning(
 656                                 'Task %s: Tool call budget (%d) exhausted; terminating agent loop.',
 657                                 task.instance_id,
 658                                 context.budget.tool_calls,
 659                             )
 660                             break
 661 
 662                         if context.llm_calls_used >= max_llm_calls:
 663                             if not agent_error:
 664                                 agent_error = (
 665                                     f'Agent exceeded turns budget ({config.budget.turns} turns)'
 666                                     if config.budget.turns is not None
 667                                     else 'Agent exceeded maximum allowed LLM turns'
 668                                 )
 669                             logger.warning(
 670                                 'Task %s: LLM turn budget exhausted.', task.instance_id
 671                             )
 672                             break
 673 
```

Original lines 755–796:
```text
 755         if context is not None and hasattr(context, 'stop_agent_session'):
 756             context.stop_agent_session()
 757 
 758         agent_patch = (context.submitted_patch if context is not None else '') or ''
 759 
 760         if not (context is not None and context.patch_submitted):
 761             # Fallback: check if the agent made unsubmitted modifications on disk
 762             try:
 763                 await sandbox_exec(docker, sandbox_id, 'cd /workspace && git add -N .')
 764                 diff_res = await sandbox_exec(
 765                     docker,
 766                     sandbox_id,
 767                     'cd /workspace && (git diff --binary _swegemma_baseline 2>/dev/null || git diff --binary HEAD)',
 768                 )
 769                 unsubmitted_diff = diff_res.stdout or ''
 770                 if unsubmitted_diff.strip():
 771                     logger.info(
 772                         'Task %s: Captured unsubmitted working tree modifications as fallback patch (%d bytes)',
 773                         task.instance_id,
 774                         len(unsubmitted_diff),
 775                     )
 776                     agent_patch = unsubmitted_diff
 777                 else:
 778                     if not agent_error:
 779                         agent_error = (
 780                             'Agent completed execution without calling submit_patch.'
 781                         )
 782                     logger.warning(
 783                         'Task %s completed without explicit submit_patch call and no working tree modifications.',
 784                         task.instance_id,
 785                     )
 786             except Exception as fallback_err:
 787                 logger.warning(
 788                     'Failed to check fallback working tree diff for task %s: %s',
 789                     task.instance_id,
 790                     fallback_err,
 791                 )
 792                 if not agent_error:
 793                     agent_error = (
 794                         'Agent completed execution without calling submit_patch.'
 795                     )
 796 
```

## swegemma/context.py

SHA-256: `4d7426bd6121f868af2f0a8786c78f1364ee285917ce325e02dc41672ebab50c`

Original lines 183–198:
```text
 183     def start_agent_session(self) -> None:
 184         """Mark the start of the active agent problem-solving session."""
 185         with self._lock:
 186             self.agent_start_time = time.perf_counter()
 187             self.agent_end_time = None
 188 
 189     def stop_agent_session(self) -> None:
 190         """Mark the end of the active agent session to freeze the agent elapsed timer."""
 191         with self._lock:
 192             if self.agent_start_time is not None and self.agent_end_time is None:
 193                 self.agent_end_time = time.perf_counter()
 194 
 195     def record_llm_call(self) -> None:
 196         """Record that an LLM call was executed."""
 197         with self._lock:
 198             self.llm_calls_used += 1
```

Original lines 459–545:
```text
 459     def remaining_time_seconds(self) -> float | None:
 460         """Remaining agent session time in seconds, or None if session time is unconstrained."""
 461         if self.budget.time_minutes is None:
 462             return None
 463         if self.agent_start_time is None:
 464             return self.budget.time_minutes * 60.0
 465         return max(0.0, (self.budget.time_minutes * 60.0) - self.agent_elapsed_seconds)
 466 
 467     def check_budget(self, *, count_tool_call: bool = True) -> str | None:
 468         """Check consumable budget limits and increment call count. Returns error JSON if exceeded."""
 469         with self._lock:
 470             if (
 471                 count_tool_call
 472                 and self.budget.tool_calls is not None
 473                 and self.tool_calls_used >= self.budget.tool_calls
 474             ):
 475                 return json.dumps(
 476                     {
 477                         'status': 'error',
 478                         'error_type': 'BudgetExceeded',
 479                         'error_message': f'Tool call budget exhausted ({self.budget.tool_calls} calls)',
 480                     }
 481                 )
 482 
 483             if (
 484                 count_tool_call
 485                 and self.budget.turns is not None
 486                 and self.llm_calls_used >= self.budget.turns
 487             ):
 488                 return json.dumps(
 489                     {
 490                         'status': 'error',
 491                         'error_type': 'BudgetExceeded',
 492                         'error_message': f'Turn budget exhausted ({self.budget.turns} turns)',
 493                     }
 494                 )
 495 
 496             if (
 497                 count_tool_call
 498                 and self.budget.cost_usd is not None
 499                 and self.total_cost_usd >= self.budget.cost_usd
 500             ):
 501                 return json.dumps(
 502                     {
 503                         'status': 'error',
 504                         'error_type': 'BudgetExceeded',
 505                         'error_message': f'Cost budget exhausted (${self.budget.cost_usd:.4f} USD)',
 506                     }
 507                 )
 508 
 509             if (
 510                 count_tool_call
 511                 and self.budget.total_tokens is not None
 512                 and self.total_tokens >= self.budget.total_tokens
 513             ):
 514                 return json.dumps(
 515                     {
 516                         'status': 'error',
 517                         'error_type': 'BudgetExceeded',
 518                         'error_message': f'Token budget exhausted ({self.budget.total_tokens} tokens)',
 519                     }
 520                 )
 521 
 522             if self.budget.time_minutes is not None:
 523                 if self.budget.time_minutes <= 0:
 524                     return json.dumps(
 525                         {
 526                             'status': 'error',
 527                             'error_type': 'TimeoutExceeded',
 528                             'error_message': f'Session time budget exhausted ({self.budget.time_minutes} minutes)',
 529                         }
 530                     )
 531                 if self.agent_start_time is not None:
 532                     elapsed_minutes = self.agent_elapsed_seconds / 60.0
 533                     if elapsed_minutes > self.budget.time_minutes:
 534                         return json.dumps(
 535                             {
 536                                 'status': 'error',
 537                                 'error_type': 'TimeoutExceeded',
 538                                 'error_message': f'Session time budget exhausted ({self.budget.time_minutes} minutes)',
 539                             }
 540                         )
 541 
 542             if count_tool_call:
 543                 self.tool_calls_used += 1
 544             return None
 545 
```

## swegemma/tools/execution.py

SHA-256: `71b7d4f4d62550ee74722d8551bd3337f64792023fd1a8ff3f64695e4d407374`

Original lines 15–47:
```text
  15 @budget_gated
  16 def run_command(ctx: Any, command: str) -> str:
  17     """Execute a shell command inside the repository sandbox.
  18 
  19     You can run any shell command (Python scripts, tests, git commands, etc.).
  20     The command runs in the repository workspace. No network access.
  21 
  22     Args:
  23         ctx: Context instance.
  24         command: Shell command to execute (e.g., "pytest tests/", "python3 script.py").
  25     """
  26     # Determine effective command timeout
  27     cmd_timeout = ctx.harness.command_timeout_seconds
  28     if ctx.budget.time_minutes is not None:
  29         remaining = ctx.remaining_time_seconds
  30         if remaining is not None:
  31             # Provide a minimum floor of 5s on remaining time so near-deadline commands don't fail immediately on 1s
  32             effective_remaining = max(5, int(remaining))
  33             if cmd_timeout is not None:
  34                 effective_timeout = min(cmd_timeout, effective_remaining)
  35             else:
  36                 effective_timeout = effective_remaining
  37         else:
  38             effective_timeout = cmd_timeout
  39     else:
  40         effective_timeout = cmd_timeout
  41 
  42     result = ctx.docker.exec(ctx.container_id, command, timeout=effective_timeout)
  43     if result.timed_out:
  44         return error_response(
  45             error_type='TimeoutExceeded',
  46             error_message=f'Command timed out after {effective_timeout} seconds',
  47         )
```

Original lines 71–98:
```text
  71 @budget_gated(count_tool_call=False)
  72 def submit_patch(ctx: Any) -> str:
  73     """Capture the current working tree modifications as the agent's submission.
  74 
  75     This generates a unified git diff against the baseline commit. The agent
  76     can submit multiple times; only the final submission is evaluated.
  77 
  78     Args:
  79         ctx: Context instance.
  80     """
  81     ctx.docker.exec(ctx.container_id, 'cd /workspace && git add -N .')
  82     res = ctx.docker.exec(
  83         ctx.container_id,
  84         'cd /workspace && (git diff --binary _swegemma_baseline 2>/dev/null || git diff --binary HEAD)',
  85     )
  86     if res.exit_code != 0:
  87         return error_response(
  88             error_type='GitDiffError',
  89             error_message=f'git diff failed with exit code {res.exit_code}: {res.stderr}',
  90         )
  91 
  92     diff = res.stdout
  93     ctx.submitted_patch = diff
  94     ctx.patch_submitted = True
  95     return ok_response(
  96         patch_size=len(diff),
  97         files_changed=diff.count('diff --git'),
  98     )
```

## swegemma/harness/verification.py

SHA-256: `5602f33acb49062de8e99073199745679d86445297be9dd075ccec60c41b60dd`

Original lines 482–505:
```text
 482         # Run pytest on verification test files (excluding conftest.py from positional test args)
 483         pytest_targets = [
 484             p
 485             for p in target_test_files
 486             if Path(p).name != 'conftest.py' and p.endswith('.py')
 487         ] or target_test_files
 488         test_files_quoted = [shlex.quote(p) for p in pytest_targets]
 489         test_files_str = ' '.join(test_files_quoted) if test_files_quoted else '.'
 490         junit_xml_path = f'/tmp/_swegemma_junit_{uuid.uuid4().hex[:12]}.xml'
 491         await sandbox_exec(docker, eval_id, f'rm -f {junit_xml_path}')
 492         pytest_cmd = (
 493             f'cd /workspace && PYTHONSAFEPATH=1 PYTHONNOUSERSITE=1 python3 -s -m pytest {test_files_str} '
 494             f'--junitxml={junit_xml_path} '
 495             '-p no:anyio -o timeout=0 '
 496             '-o python_classes="Test* *Test" -q'
 497         )
 498         test_res = await sandbox_exec(
 499             docker,
 500             eval_id,
 501             pytest_cmd,
 502             timeout=config.harness.command_timeout_seconds,
 503         )
 504 
 505         resolved = test_res.exit_code == 0
```

## adk_submission/resolvers/callbacks.py

SHA-256: `332ce0732aa3eb1b555630c57683f51e59d5caee3f519371d376210e9ae33e29`

Original lines 65–95:
```text
  65 def resolve_callback_fields(
  66     config: Any,
  67     ctx: CompilationContext,
  68     fields: list[tuple[str, str, CallbackType]],
  69 ) -> dict[str, Any]:
  70     """Resolve a specified list of YAML callback fields into ADK constructor keyword arguments.
  71 
  72     Args:
  73         config: The sandboxed agent configuration object.
  74         ctx: The compilation context containing the callback registry.
  75         fields: A list of tuples containing (yaml_field_name, adk_kwarg_name, CallbackType).
  76 
  77     Returns:
  78         A dictionary mapping ADK callback keyword arguments to lists of resolved callback callables.
  79 
  80     Raises:
  81         CallbackNotFoundError: If a referenced callback name is not found in the callback registry.
  82     """
  83     kwargs: dict[str, Any] = {}
  84     if ctx.callback_registry is None:
  85         return kwargs
  86 
  87     for yaml_field, adk_kwarg, cb_type in fields:
  88         names: list[str] | None = getattr(config, yaml_field, None)
  89         if names:
  90             resolved = [
  91                 ctx.callback_registry.get(name, cb_type) for name in names
  92             ]
  93             kwargs[adk_kwarg] = resolved
  94 
  95     return kwargs
```

## adk_submission/resolvers/generation.py

SHA-256: `e54d0995488e57011d6daeae327bd7ed0555c230703dfd29feb80da4bc7e5434`

Original lines 61–126:
```text
  61 def apply_thinking_config_to_model(model: Any, gen_config: Any) -> Any:
  62     """Attach thinking_config from GenerateContentConfig to LiteLlm._additional_args.
  63 
  64     Because ADK's LiteLlm._get_completion_inputs() does not forward Google GenAI
  65     thinking_config to OpenAI-compatible endpoints (like vLLM), this bridges
  66     thinking_level and include_thoughts into extra_body.chat_template_kwargs.enable_thinking
  67     and reasoning_effort on a cloned model instance.
  68     """
  69     if model is None or gen_config is None:
  70         return model
  71 
  72     thinking_cfg = getattr(gen_config, "thinking_config", None)
  73     if thinking_cfg is None and isinstance(gen_config, dict):
  74         thinking_cfg = gen_config.get("thinking_config")
  75     if thinking_cfg is None:
  76         return model
  77 
  78     if not hasattr(model, "_additional_args") or not isinstance(model._additional_args, dict):
  79         return model
  80 
  81     if isinstance(thinking_cfg, dict):
  82         include_thoughts = thinking_cfg.get("include_thoughts")
  83         raw_level = thinking_cfg.get("thinking_level")
  84         thinking_budget = thinking_cfg.get("thinking_budget")
  85     else:
  86         include_thoughts = getattr(thinking_cfg, "include_thoughts", None)
  87         raw_level = getattr(thinking_cfg, "thinking_level", None)
  88         thinking_budget = getattr(thinking_cfg, "thinking_budget", None)
  89 
  90     level_str: str | None = None
  91     if raw_level is not None:
  92         level_str = str(getattr(raw_level, "value", raw_level)).lower()
  93 
  94     if (
  95         include_thoughts is False
  96         or level_str == "none"
  97         or (thinking_budget is not None and int(thinking_budget) <= 0)
  98     ):
  99         enable_thinking = False
 100     elif (
 101         include_thoughts is True
 102         or level_str in {"minimal", "low", "medium", "high"}
 103         or (thinking_budget is not None and int(thinking_budget) > 0)
 104     ):
 105         enable_thinking = True
 106     else:
 107         return model
 108 
 109     import copy
 110 
 111     cloned = copy.copy(model)
 112     additional_args = copy.deepcopy(model._additional_args)
 113 
 114     extra_body = dict(additional_args.get("extra_body") or {})
 115     chat_template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
 116     chat_template_kwargs["enable_thinking"] = enable_thinking
 117     extra_body["chat_template_kwargs"] = chat_template_kwargs
 118     additional_args["extra_body"] = extra_body
 119 
 120     if enable_thinking and level_str in {"low", "medium", "high"}:
 121         additional_args["reasoning_effort"] = level_str
 122     elif not enable_thinking:
 123         additional_args.pop("reasoning_effort", None)
 124 
 125     cloned._additional_args = additional_args
 126     return cloned
```

## adk_submission/schema.py

SHA-256: `f58d205915065e3e4ffb83eff8d71aa094509105a3e048b15f5fa2f48b727c82`

Original lines 155–175:
```text
 155 class ThinkingConfig(BaseModel):
 156     """Thinking configuration for supported LLMs (e.g., reasoning models).
 157 
 158     Attributes:
 159         thinking_budget (Optional[int]): The token budget allocated for the model's internal
 160             reasoning/thinking process. Must be greater than or equal to 1. Defaults to None.
 161         include_thoughts (Optional[bool]): Whether to include the raw reasoning thoughts in the
 162             final model output. Defaults to None.
 163         thinking_level (Optional[str]): Presets for model thinking level (e.g., 'MINIMAL', 'LOW', 'MEDIUM', 'HIGH'). Defaults to None.
 164     """
 165 
 166     model_config = ConfigDict(extra="forbid", frozen=True)
 167 
 168     thinking_budget: int | None = Field(default=None, ge=1)
 169     include_thoughts: bool | None = None
 170     thinking_level: str | None = None
 171 
 172     @field_validator("thinking_budget", mode="before")
 173     @classmethod
 174     def _validate_thinking_budget(cls, v: Any) -> Any:
 175         return _reject_bool_nan_inf(v, "thinking_budget")
```

Original lines 278–298:
```text
 278     adapter: str | None = None
 279     instruction: str = Field(
 280         description="Agent instruction text — inline or resolved from !include."
 281     )
 282     output_key: str | None = None
 283     include_contents: Literal["default", "none"] = "default"
 284     disallow_transfer_to_parent: bool | None = None
 285     disallow_transfer_to_peers: bool | None = None
 286     tools: list[str | AgentToolEntry] | None = None
 287     skills: list[str] | None = Field(
 288         default=None,
 289         description="List of relative paths to skill directories within the submission archive.",
 290     )
 291     sub_agents: list[SubAgentRef] | None = None
 292     generate_content_config: GenerateContentConfig | None = None
 293 
 294     # LLM-specific callbacks
 295     before_model_callbacks: list[str] | None = None
 296     after_model_callbacks: list[str] | None = None
 297     before_tool_callbacks: list[str] | None = None
 298     after_tool_callbacks: list[str] | None = None
```

## google/adk/models/lite_llm.py

SHA-256: `2f28c6afa110006e0fdec46c9ac7c6b614d74c81e0d0bf6fda2b0d6353111098`

Original lines 800–825:
```text
 800   tool_messages: list[Message] = []
 801   non_tool_parts: list[types.Part] = []
 802   for part in content.parts:
 803     if part.function_response:
 804       response = part.function_response.response
 805       response_content = (
 806           response
 807           if isinstance(response, str)
 808           else _safe_json_serialize(response)
 809       )
 810       # gemma4 requires role='tool_responses' for recognizing function_response parts as responses
 811       # from the tool call, instead of OpenAI-compatible 'tool' role used by other models.
 812       # Earlier Gemma versions before version 4 do not support tool use,
 813       # so this check is intentionally scoped to only look for "gemma4" in the model name.
 814       tool_role = "tool_responses" if "gemma4" in model.lower() else "tool"
 815       tool_messages.append(
 816           ChatCompletionToolMessage(
 817               role=tool_role,
 818               tool_call_id=part.function_response.id,
 819               content=response_content,
 820           )
 821       )
 822     else:
 823       non_tool_parts.append(part)
 824 
 825   if tool_messages and not non_tool_parts:
```

Original lines 879–940:
```text
 879       elif part.thought:
 880         reasoning_parts.append(part)
 881       else:
 882         content_parts.append(part)
 883 
 884     final_content = (
 885         await _get_content(content_parts, provider=provider, model=model)
 886         if content_parts
 887         else None
 888     )
 889     if final_content and isinstance(final_content, list):
 890       # when the content is a single text object, we can use it directly.
 891       # this is needed for ollama_chat provider which fails if content is a list
 892       final_content = (
 893           final_content[0].get("text", "")
 894           if final_content[0].get("type", None) == "text"
 895           else final_content
 896       )
 897 
 898     # For Anthropic models, rebuild thinking_blocks with signatures so that
 899     # thinking is preserved across tool call boundaries. Without this,
 900     # Anthropic silently drops thinking after the first turn.
 901     if model and _is_anthropic_model(model) and reasoning_parts:
 902       thinking_blocks = []
 903       for part in reasoning_parts:
 904         if part.text and part.thought_signature:
 905           sig = part.thought_signature
 906           if isinstance(sig, bytes):
 907             sig = sig.decode("utf-8")
 908           thinking_blocks.append({
 909               "type": "thinking",
 910               "thinking": part.text,
 911               "signature": sig,
 912           })
 913       if thinking_blocks:
 914         msg = ChatCompletionAssistantMessage(
 915             role=role,
 916             content=final_content,
 917             tool_calls=tool_calls or None,
 918         )
 919         msg["thinking_blocks"] = thinking_blocks  # type: ignore[typeddict-unknown-key]
 920         return msg
 921 
 922     reasoning_texts = []
 923     for part in reasoning_parts:
 924       if part.text:
 925         reasoning_texts.append(part.text)
 926       elif (
 927           part.inline_data
 928           and part.inline_data.data
 929           and part.inline_data.mime_type
 930           and part.inline_data.mime_type.startswith("text/")
 931       ):
 932         reasoning_texts.append(_decode_inline_text_data(part.inline_data.data))
 933 
 934     reasoning_content = _NEW_LINE.join(text for text in reasoning_texts if text)
 935     return ChatCompletionAssistantMessage(
 936         role=role,
 937         content=final_content,
 938         tool_calls=tool_calls or None,
 939         reasoning_content=reasoning_content or None,
 940     )
```

Original lines 1943–1970:
```text
1943   # 4. Extract generation parameters
1944   generation_params: dict | None = None
1945   if llm_request.config:
1946     config_dict = llm_request.config.model_dump(exclude_none=True)
1947     # Generate LiteLlm parameters here,
1948     # Following https://docs.litellm.ai/docs/completion/input.
1949     generation_params = {}
1950     param_mapping = {
1951         "max_output_tokens": "max_completion_tokens",
1952         "stop_sequences": "stop",
1953     }
1954     for key in (
1955         "temperature",
1956         "max_output_tokens",
1957         "top_p",
1958         "top_k",
1959         "stop_sequences",
1960         "presence_penalty",
1961         "frequency_penalty",
1962     ):
1963       if key in config_dict:
1964         mapped_key = param_mapping.get(key, key)
1965         generation_params[mapped_key] = config_dict[key]
1966 
1967     if not generation_params:
1968       generation_params = None
1969 
1970   return messages, tools, response_format, generation_params
```

## google/adk/flows/llm_flows/functions.py

SHA-256: `551bf06d5e6bc0f257eef899deb431493d0b6c45e415f7731f5ed7bac4e95e06`

Original lines 1110–1132:
```text
1110     tool: BaseTool,
1111     function_result: dict[str, object],
1112     tool_context: ToolContext,
1113     invocation_context: InvocationContext,
1114 ) -> Event:
1115   # Specs requires the result to be a dict.
1116   if not isinstance(function_result, dict):
1117     function_result = {'result': function_result}
1118 
1119   function_response_parts = None
1120   if isinstance(tool, ComputerUseTool):
1121     function_response_parts = _try_decode_computer_use_image(
1122         tool, function_result
1123     )
1124 
1125   part_function_response = types.Part.from_function_response(
1126       name=tool.name,
1127       response=function_result,
1128       parts=function_response_parts,
1129   )
1130   part_function_response.function_response.id = tool_context.function_call_id
1131 
1132   content = types.Content(
```

## swegemma/tools/base.py

SHA-256: `b430d48b6b32b9f886f8e4f07196a5e47a10d5307b827b19816a8d14f81b4ee2`

Original lines 11–31:
```text
  11 def ok_response(**kwargs: Any) -> str:
  12     """Format a successful tool JSON response."""
  13     data = {'status': 'ok'}
  14     data.update(kwargs)
  15     return json.dumps(data)
  16 
  17 
  18 def error_response(
  19     error_type: str,
  20     error_message: str,
  21     details: dict[str, Any] | None = None,
  22 ) -> str:
  23     """Format an error tool JSON response."""
  24     data: dict[str, Any] = {
  25         'status': 'error',
  26         'error_type': error_type,
  27         'error_message': error_message,
  28     }
  29     if details is not None:
  30         data['details'] = details
  31     return json.dumps(data)
```

## adk_submission/resolvers/tools.py

SHA-256: `ae44a5ca815a05edfd763b92283d0444c77755afbd827fc35bcb87eb2633259e`

Original lines 184–221:
```text
 184         else:
 185             skill_path = validate_sandboxed_path(
 186                 path=skill_ref,
 187                 base_dir=ctx.root_dir,
 188                 allow_relative=True,
 189                 must_exist=True,
 190                 allow_symlinks=False,
 191                 error_prefix="Skill",
 192             )
 193             skill_size = 0
 194             for child in sorted(skill_path.rglob("*")):
 195                 if child.is_symlink():
 196                     raise PathTraversalError(
 197                         f"Skill path escapes submission directory: {child}"
 198                     )
 199                 validate_sandboxed_path(
 200                     path=child,
 201                     base_dir=ctx.root_dir,
 202                     must_exist=True,
 203                     allow_symlinks=False,
 204                     error_prefix="Skill",
 205                 )
 206                 if child.is_file():
 207                     skill_size += child.stat().st_size
 208             if skill_size > ctx.limits.max_skill_size_bytes:
 209                 raise LimitExceededError(
 210                     f"Skill '{skill_ref}' size {skill_size:,} bytes "
 211                     f"exceeds limit of {ctx.limits.max_skill_size_bytes:,} bytes"
 212                 )
 213             try:
 214                 from google.adk.skills import load_skill_from_dir
 215 
 216                 skill = load_skill_from_dir(skill_path)
 217                 loaded_skills.append(skill)
 218             except Exception as e:
 219                 raise SubmissionValidationError(
 220                     f"Failed to load skill '{skill_ref}': {e}"
 221                 ) from e
```

Original lines 253–264:
```text
 253     skill_toolset_cls = _get_skill_toolset_cls(ctx)
 254     try:
 255         return skill_toolset_cls(
 256             skills=loaded_skills,
 257             code_executor=ctx.code_executor,
 258             script_timeout=ctx.script_timeout,
 259             additional_tools=list(resolved_tools),
 260         )
 261     except ValueError as e:
 262         raise SubmissionValidationError(
 263             f"Agent '{config.name}': {e}"
 264         ) from e
```

## swegemma/graph/embedding_utils.py

SHA-256: `804c5f2224515ed3e41913a41d91664ac0f01abcd080085c1fb175f868964144`

Original lines 201–216:
```text
 201         repo_cache = _REPO_CACHES[cache_key]
 202         if node_str in repo_cache:
 203             return repo_cache[node_str]
 204 
 205         # Key lookup with repo suffix fallback
 206         if repo_str and not node_str.startswith(f'{repo_str};'):
 207             alt_key = f'{repo_str};{node_str}'
 208             if alt_key in repo_cache:
 209                 return repo_cache[alt_key]
 210 
 211         # Suffix matching fallback (e.g. FastAPI -> fastapi.applications.FastAPI)
 212         for k in repo_cache:
 213             if k == node_str or k.endswith(f'.{node_str}') or k.endswith(f';{node_str}'):
 214                 return repo_cache[k]
 215 
 216         return None
```

## swegemma/graph/graph_utils.py

SHA-256: `b41439734ee0838acc7ae867b4718e9175bfea56830c4050e7789b4a0b713dcb`

Original lines 326–365:
```text
 326     single_input = isinstance(node, (str, bytes))
 327     node_list = [node] if single_input else list(node)
 328 
 329     results = {}
 330     for target in node_list:
 331         target_str = (
 332             target.decode('utf-8', errors='replace')
 333             if isinstance(target, bytes)
 334             else str(target)
 335         )
 336         resolved = resolve_node_name(target_str, graph)
 337         if not resolved:
 338             results[target] = []
 339             continue
 340 
 341         neighbors: set[str] = set()
 342 
 343         # Outgoing edges
 344         if direction in ('both', 'out'):
 345             for _, neighbor, data in graph.out_edges(resolved, data=True):
 346                 if (
 347                     edge_type is None
 348                     or data.get('type') == edge_type
 349                     or data.get('relation') == edge_type
 350                 ):
 351                     neighbors.add(str(neighbor))
 352 
 353         # Incoming edges
 354         if direction in ('both', 'in'):
 355             for neighbor, _, data in graph.in_edges(resolved, data=True):
 356                 if (
 357                     edge_type is None
 358                     or data.get('type') == edge_type
 359                     or data.get('relation') == edge_type
 360                 ):
 361                     neighbors.add(str(neighbor))
 362 
 363         results[target] = sorted(neighbors)[:max_neighbors]
 364 
 365     return results[node] if single_input else results
```

## swegemma/graph/retrieval_utils.py

SHA-256: `82016367c25a2f705c40a1cc2d4c6fecccdb27a6e554d24935a687357c66b5e3`

Original lines 377–421:
```text
 377     if query_emb is None and node_name:
 378         query_emb = embedding_utils.embed(
 379             node_name,
 380             target_repo,
 381             embeddings_dir=embeddings_dir,
 382             base_commit=base_commit,
 383         )
 384 
 385     if query_emb is None:
 386         return []
 387 
 388     query_emb = np.asarray(query_emb, dtype=np.float32).flatten()
 389     if emb_matrix.shape[1] > 0 and query_emb.shape[0] != emb_matrix.shape[1]:
 390         adjusted = np.zeros((emb_matrix.shape[1],), dtype=np.float32)
 391         dim = min(query_emb.shape[0], emb_matrix.shape[1])
 392         adjusted[:dim] = query_emb[:dim]
 393         query_emb = adjusted
 394 
 395     q_norm = np.linalg.norm(query_emb)
 396     if q_norm == 0:
 397         q_norm = 1e-10
 398 
 399     matrix_norms = np.linalg.norm(emb_matrix, axis=1)
 400     matrix_norms[matrix_norms == 0] = 1e-10
 401 
 402     dots = np.dot(emb_matrix, query_emb)
 403     sims = dots / (matrix_norms * q_norm)
 404 
 405     sorted_indices = np.argsort(sims)[::-1]
 406 
 407     results = []
 408     for idx in sorted_indices:
 409         cand_name = names[idx]
 410         if cand_name == node_name:
 411             continue
 412         results.append(
 413             {
 414                 'node_name': cand_name,
 415                 'code': texts[idx],
 416                 'similarity': float(sims[idx]),
 417                 'embedding': emb_matrix[idx],
 418             }
 419         )
 420         if len(results) == k:
 421             break
```

## swegemma/sandbox/subprocess.py

SHA-256: `a26e9265d93f0387dd0a820ea451d1c530884bebca41583738f3ab2cc31df1f2`

Original lines 28–45:
```text
  28 
  29 
  30 class SubprocessManager(BaseSandbox):
  31     """Manages isolated local working directories and virtual environments directly conforming to BaseSandbox."""
  32 
  33     def __init__(
  34         self,
  35         timeout_seconds: int = 300,
  36         base_dir: Path | None = None,
  37         system_site_packages: bool = True,
  38     ):
  39         self.timeout_seconds = timeout_seconds
  40         self.base_dir = base_dir
  41         self.system_site_packages = system_site_packages
  42         self._sandboxes: dict[str, dict[str, Path]] = {}
  43         self._default_sandbox_id: str | None = None
  44         self._initial_files: dict[str, str | bytes] = {}
  45         self._lock = threading.RLock()
```

Original lines 317–395:
```text
 317         else:
 318             sandbox_root = Path(
 319                 tempfile.mkdtemp(prefix=f'swegemma_sandbox_{sandbox_id}_')
 320             )
 321 
 322         workspace_dir = sandbox_root / 'workspace'
 323         tmp_dir = sandbox_root / 'tmp'
 324         wheels_dir = sandbox_root / 'wheels'
 325         venv_dir = sandbox_root / 'venv'
 326 
 327         workspace_dir.mkdir(parents=True, exist_ok=True)
 328         tmp_dir.mkdir(parents=True, exist_ok=True)
 329         wheels_dir.mkdir(parents=True, exist_ok=True)
 330 
 331         venv_has_pip = False
 332         for use_symlinks in (True, False):
 333             try:
 334                 venv.create(
 335                     venv_dir,
 336                     system_site_packages=self.system_site_packages,
 337                     with_pip=True,
 338                     symlinks=use_symlinks,
 339                     clear=True,
 340                 )
 341                 venv_has_pip = True
 342                 break
 343             except Exception as e:
 344                 if use_symlinks:
 345                     logger.debug(
 346                         'venv with_pip+symlinks failed for sandbox %s: %s',
 347                         sandbox_id,
 348                         e,
 349                     )
 350                     continue
 351                 logger.warning(
 352                     'ensurepip unavailable in sandbox %s, creating venv without pip: %s',
 353                     sandbox_id,
 354                     e,
 355                 )
 356 
 357         if not venv_has_pip:
 358             for use_symlinks in (True, False):
 359                 try:
 360                     venv.create(
 361                         venv_dir,
 362                         system_site_packages=self.system_site_packages,
 363                         with_pip=False,
 364                         symlinks=use_symlinks,
 365                         clear=True,
 366                     )
 367                     break
 368                 except Exception:
 369                     if not use_symlinks:
 370                         raise
 371 
 372             venv_python = venv_dir / 'bin' / 'python3'
 373             if venv_python.exists():
 374                 try:
 375                     subprocess.run(
 376                         [str(venv_python), '-m', 'pip', '--version'],
 377                         capture_output=True,
 378                         timeout=30,
 379                         check=True,
 380                     )
 381                     venv_has_pip = True
 382                 except Exception:
 383                     logger.warning(
 384                         'pip not available in sandbox %s venv; '
 385                         'task package installation may be skipped.',
 386                         sandbox_id,
 387                     )
 388 
 389         if self.system_site_packages:
 390             import sys
 391 
 392             host_site_packages = []
 393             for p in sys.path:
 394                 if 'site-packages' in p and Path(p).exists() and Path(p).is_dir():
 395                     host_site_packages.append(p)
```

Original lines 480–497:
```text
 480 
 481         translated_cmd = translated_cmd.replace(sentinel, root_dir_str)
 482 
 483         venv_bin = venv_dir / 'bin'
 484         host_bin = str(Path(sys.executable).parent)
 485         extra_env = {
 486             'VIRTUAL_ENV': str(venv_dir),
 487             'PATH': f'{venv_bin}:{host_bin}:{os.environ.get("PATH", "/usr/bin:/bin")}',
 488             'PYTHONPATH': str(workspace_dir),
 489         }
 490         env = build_sanitized_env(
 491             work_dir=workspace_dir,
 492             tmp_dir=tmp_dir,
 493             extra_env=extra_env,
 494         )
 495         return execute_subprocess_command(
 496             command=['/bin/bash', '-c', translated_cmd],
 497             cwd=workspace_dir,
```

## /private/tmp/gemma4-research/HARNESS_README.md

SHA-256: `3d6e57a13234cb4e783ba24eaab486459af76e0923ea4c6607cd41cc8961bbbb`

Original lines 97–98:
```text
  97 - **Root Config Discovery**: Exactly one root config (`agent.yaml`, `agent.yml`, `root_agent.yaml`, or `root_agent.yml`) must exist in the submission root. Having zero root files raises `MissingRootConfigError`; having more than one raises `MultipleRootConfigsError`.
  98 - **Optional Evaluation Budget Config (`eval_config.yaml`)**: Submissions may include an optional `eval_config.yaml` at the root under an `evaluation:` key (`timeout_seconds`, `max_tool_calls`, `max_time_minutes`, `max_turns`) to customize per-task operational budgets during Stage 1 inference (`load_submission_eval_config` in `scripts/inference.py`).
```

Original lines 104–128:
```text
 104 ### 2.3. Supported Agent Classes (`agent_class`)
 105 `SandboxedAgentConfig` supports four discriminated agent types (defaulting to `LlmAgent` if `agent_class` is omitted):
 106 
 107 1. **`LlmAgent` (default)**:
 108    - `name` (`str`, required): Agent identifier string.
 109    - `model` (`str`, required unless a default model is set): Model alias from the registry (`gemma-4-31b-it-qat-w4a16-ct` for competition submissions).
 110    - `adapter` (`str | None`): Optional adapter name matching a subdirectory or weight file under `adapters/`.
 111    - `description` (`str`, up to `1,000,000` chars): Used by parent agents when delegating to this agent.
 112    - `instruction` (`str`, up to `1,000,000` chars): System prompt for the agent. Supports ADK session state templating (`{problem_description}`, `{hints}`, or custom `output_key` variables).
 113    - `tools` (`list`): List of tool names (`str`), `AgentTool` references (`agent_tool: {config_path: ..., skip_summarization: true}`), or inline agent dictionaries.
 114    - `skills` (`list[str] | None`): List of relative paths to skill directories within the submission archive.
 115    - `sub_agents` (`list`): List of `{config_path: ...}` references for multi-agent transfer.
 116    - `output_key` (`str | None`): Saves the agent's final text output into `session.state[output_key]`.
 117    - `include_contents` (`"default" | "none"`): Controls whether conversation history is passed to the agent.
 118    - `disallow_transfer_to_parent` / `disallow_transfer_to_peers` (`bool | None`).
 119    - `generate_content_config` (`dict | None`): Sampling, seed, and thinking parameters (see constraints below).
 120 2. **`SequentialAgent`**:
 121    - Executes `sub_agents` in sequential order. Accepts `name`, `description`, `sub_agents`, and agent callbacks.
 122 3. **`ParallelAgent`**:
 123    - Executes `sub_agents` concurrently in isolated branches. Accepts `name`, `description`, `sub_agents`, and agent callbacks.
 124 4. **`LoopAgent`**:
 125    - Repeatedly executes `sub_agents` up to `max_iterations` (`int`, default `500`, bounded to `1..500`).
 126 
 127 ### 2.4. Enforced Structural Limits & Generation Constraints
 128 Submission constraints in `swegemma` (`build_submission_limits()` in `swegemma/config.py`) are intentionally minimal: **aside from security sandboxing (no symlinks, no path traversal, declarative YAML only), the primary hard constraint is a total unpacked submission size of `< 3 GiB` (`3,221,225,472` bytes), including all `adapters/` weights**, plus token ceilings required by the 4x L4 GPU context window (`32,768` tokens).
```

Original lines 247–257:
```text
 247 - **Docker Backend (`--sandbox docker`, default)**:
 248   - Image: `swebench-sandbox:latest`
 249   - Network: **`network_mode="none"`** (completely air-gapped; no internet or PyPI access)
 250   - Memory Limit: **`4g`** (`4 GiB`; processes exceeding this are killed with exit code `137` / `SIGKILL`)
 251   - CPU Quota: **2 vCPUs** (`cpu_period=100_000`, `cpu_quota=200_000`)
 252   - Working Directory: `/workspace` (`TEST_TMPDIR=/tmp`)
 253   - Warm Pooling (`reuse_containers=True`): When a phase finishes, `/workspace` is completely wiped (`find /workspace -mindepth 1 -maxdepth 1 -exec rm -rf {} +`), temporary patches/stubs in `/tmp` and `/var/tmp` are removed, and the container is returned to an idle pool.
 254 - **Subprocess Backend (`--sandbox subprocess`, used in environments without a Docker daemon)**:
 255   - Creates an isolated temporary directory `swegemma_sandbox_<id>_` with `workspace/`, `tmp/`, `wheels/`, and an isolated virtual environment `venv/`.
 256   - Rewrites shell references to `/workspace`, `/tmp`, `/wheels`, and `/usr/local/bin` to the sandbox directory paths and executes commands in a new process group (`start_new_session=True`), terminating the full process group (`os.killpg(..., signal.SIGKILL)`) on command timeout.
 257 
```

Original lines 528–563:
```text
 528 Participants can customize per-task budgets in their submission's `eval_config.yaml` (under the `evaluation:` key), which `scripts/inference.py` loads via `load_submission_eval_config()`:
 529 
 530 ```yaml
 531 evaluation:
 532   timeout_seconds: 300
 533   max_tool_calls: 10
 534   max_time_minutes: 1
 535   max_turns: 50
 536 ```
 537 
 538 | Parameter | `EvalConfig` Path | `swegemma` / `inference.py` Default | `eval_config.yaml` Key | Environment Variable Override (`scripts/inference.py`) |
 539 | :--- | :--- | :--- | :--- | :--- |
 540 | **Session Wall-Clock Time** | `config.budget.time_minutes` | `60.0` min (`1` min in sample submission) | `evaluation.max_time_minutes` | `SWE_MAX_TIME_MINUTES` |
 541 | **Max Tool Calls** | `config.budget.tool_calls` | `100` in `inference.py` / `None` in CLI (`10` in sample submission) | `evaluation.max_tool_calls` | `SWE_MAX_TOOL_CALLS` |
 542 | **Max LLM Reasoning Turns** | `config.budget.turns` | `500` turns (`None` $\to$ `500`; `50` in sample submission) | `evaluation.max_turns` | `SWE_MAX_TURNS` / `SWE_MAX_LOOP_ITERATIONS` |
 543 | **Max Monetary Cost** | `config.budget.cost_usd` | `None` | — | — |
 544 | **Max Total Tokens** | `config.budget.total_tokens` | `None` | — | — |
 545 | **Single Command Timeout** | `config.harness.command_timeout_seconds` | `300` sec | `evaluation.timeout_seconds` | `SWE_TIMEOUT_SECONDS` |
 546 | **Max Command / Diff Output** | `config.harness.max_stdout_chars` | `5,000` chars | — | — |
 547 | **Max Lines per `read_file`** | `config.harness.max_file_lines` | `150` lines | — | — |
 548 | **Max Chars per `read_file`** | `config.harness.max_file_chars` | `10,000` chars | — | — |
 549 
 550 > [!TIP]
 551 > **Container Setup Is Excluded from Agent Time Budget**: `SwegemmaContext` records `task_start_time` at creation, but `agent_start_time` is started via `context.start_agent_session()` *only after* container creation, wheel installation, and snapshot extraction finish. Your `time_minutes` budget measures strictly the time spent inside the agent loop.
 552 
 553 ### 7.2. Automatic Context Window Compaction & Prefix Caching
 554 To prevent long debugging trajectories from overflowing the 32,768-token context window on vLLM, `swegemma` configures Google ADK's context management on the `App` instance (`scripts/inference.py`):
 555 - **Events Compaction (`EventsCompactionConfig`)**:
 556   - `compaction_interval = 5`
 557   - `overlap_size = 2`
 558   - `token_threshold = 14,336`
 559   - `event_retention_size = 5`
 560 - **Context Caching (`ContextCacheConfig`)**:
 561   - `min_tokens = 2,048`, `ttl_seconds = 1,800`, `cache_intervals = 10`
 562 - **Transient Model Error Retries (`ModelRetryPlugin`)**:
 563   - Automatically intercepts transient LLM/proxy errors (`429`, `500`, `502`, `503`, `504`, `RateLimitError`, `APIConnectionError`, `APITimeoutError`) and retries up to `5` times with exponential backoff (`2.0s` initial delay, `2.0x` multiplier, max `60.0s`, $\pm 20\%$ jitter).
```
