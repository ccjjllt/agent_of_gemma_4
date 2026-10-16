"""CLI interface for SWE-gemma."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import sys
from pathlib import Path
from typing import Any

from adk_submission import ModelRegistry, discover_adapters
from adk_submission.discovery import discover_declared_models
from adk_submission.yaml_loader import load_yaml
from dotenv import load_dotenv
from google.adk.models.lite_llm import LiteLlm

from swegemma.evaluate import EvalConfig, Evaluator
from swegemma.warnings import _suppress_warnings


def _normalize_api_endpoint(url: str | None) -> str | None:
    """Normalize Model Proxy and LiteLLM API base endpoints."""
    if not url:
        return None
    url = url.rstrip('/')
    if url.endswith('/models'):
        return f'{url}/openapi'
    if '/models' in url and not (
        url.endswith('/openapi') or url.endswith('/v1') or url.endswith('/genai')
    ):
        return f'{url}/openapi'
    return url


def _build_cli_model_registry(
    submission_dir: Path | str | None = None,
    models_yaml_path: Path | str | None = None,
) -> ModelRegistry:
    """Build a ModelRegistry for CLI evaluation from models.yaml and/or declared submission models."""
    endpoint = _normalize_api_endpoint(
        os.environ.get('MODEL_PROXY_URL')
        or os.environ.get('LITELLM_API_BASE')
        or os.environ.get('LOCAL_INFERENCE_URL')
        or os.environ.get('OPENAI_BASE_URL')
        or 'http://localhost:8000/v1'
    )
    key = (
        os.environ.get('MODEL_PROXY_API_KEY')
        or os.environ.get('LITELLM_API_KEY')
        or os.environ.get('LOCAL_API_KEY')
        or os.environ.get('OPENAI_API_KEY')
        or 'EMPTY'
    )

    models = ModelRegistry()
    if submission_dir is not None and Path(submission_dir).is_dir():
        with contextlib.suppress(Exception):
            for raw_model in discover_declared_models(submission_dir):
                clean = raw_model.strip()
                target = (
                    clean
                    if clean.startswith(('openai/', 'hosted_vllm/', 'custom/'))
                    else f'openai/{clean}'
                )
                models.register(
                    clean,
                    LiteLlm(
                        model=target,
                        api_base=endpoint,
                        api_key=key,
                        num_retries=5,
                    ),
                )
        with contextlib.suppress(Exception):
            manifest = discover_adapters(submission_dir)
            if getattr(manifest, 'adapters', None):
                def _make_adapter(info: Any) -> Any:
                    name = getattr(info, 'name', str(info))
                    return LiteLlm(
                        model=f'openai/{name}',
                        api_base=endpoint,
                        api_key=key,
                        num_retries=5,
                    )

                manifest.register_all(models, _make_adapter)
                for name, info in manifest.adapters.items():
                    models.register(name, _make_adapter(info))

    yaml_path = Path(models_yaml_path).resolve() if models_yaml_path else None
    if yaml_path and yaml_path.exists():
        data = load_yaml(yaml_path, yaml_path.parent) or {}
        for model_alias, cfg in (data.get('models') or {}).items():
            model_name = cfg.get('path') or f'openai/{model_alias}'
            model_endpoint = _normalize_api_endpoint(cfg.get('api_base')) or endpoint
            model_key = cfg.get('api_key') or key
            models.register(
                model_alias,
                LiteLlm(
                    model=model_name,
                    api_base=model_endpoint,
                    api_key=model_key,
                    num_retries=5,
                ),
            )

    return models


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point for SWE-gemma."""
    load_dotenv(override=False)
    parser = argparse.ArgumentParser(
        prog='swegemma',
        description='SWE-bench evaluation system for Gemma 4 agents.',
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Subcommand: eval
    eval_parser = subparsers.add_parser(
        'eval', help='Evaluate an agent on SWE-bench tasks'
    )
    eval_parser.add_argument(
        '--tasks', required=True, type=Path, help='Path to tasks.jsonl'
    )
    eval_parser.add_argument(
        '--snapshots-dir',
        required=True,
        type=Path,
        help='Directory with repository snapshots',
    )
    eval_parser.add_argument(
        '--results-dir',
        required=True,
        type=Path,
        help='Output directory for evaluation results',
    )
    eval_parser.add_argument(
        '--submission-dir',
        required=True,
        type=Path,
        help='Directory containing agent package or notebook',
    )
    eval_parser.add_argument(
        '--image', default='swebench-sandbox:latest', help='Sandbox docker image'
    )
    eval_parser.add_argument(
        '--sandbox',
        choices=['docker', 'subprocess'],
        default='docker',
        help='Sandbox manager type',
    )
    eval_parser.add_argument(
        '--max-tool-calls',
        type=int,
        default=None,
        help='Maximum allowed tool calls per task',
    )
    eval_parser.add_argument(
        '--max-turns',
        type=int,
        default=None,
        help='Maximum allowed LLM turns per task',
    )
    eval_parser.add_argument(
        '--timeout-seconds',
        type=int,
        default=None,
        help='Single command execution timeout in seconds',
    )
    eval_parser.add_argument(
        '--verification-timeout-seconds',
        type=int,
        default=None,
        help='Phase 2 pytest verification timeout in seconds (default: 300)',
    )
    eval_parser.add_argument(
        '--max-time-minutes',
        type=float,
        default=60.0,
        help='Maximum session time per task in minutes',
    )
    eval_parser.add_argument(
        '--models-yaml',
        type=Path,
        default=None,
        help='Path to models.yaml registry config',
    )
    eval_parser.add_argument(
        '--task-id',
        type=str,
        action='append',
        dest='task_ids',
        help='Specific task instance_id(s) to evaluate (repeatable)',
    )
    eval_parser.add_argument(
        '--task-ids',
        nargs='+',
        dest='task_ids',
        help='Specific task instance_id(s) to evaluate',
    )
    eval_parser.add_argument(
        '--skip-agent-patch',
        action='store_true',
        help='Skip applying agent patch to evaluate baseline behavior',
    )
    eval_parser.add_argument(
        '--concurrency',
        type=int,
        default=1,
        help='Maximum number of parallel task evaluations (default: 1)',
    )
    eval_parser.add_argument(
        '--shard-index',
        type=int,
        default=None,
        help='0-based shard index for distributed evaluation',
    )
    eval_parser.add_argument(
        '--num-shards',
        type=int,
        default=None,
        help='Total number of shards for distributed evaluation',
    )
    eval_parser.add_argument(
        '--verbose', action='store_true', help='Enable verbose output'
    )
    eval_parser.add_argument(
        '--display',
        choices=['auto', 'dashboard', 'single', 'quiet'],
        default='auto',
        help='Display mode for evaluation: auto (dashboard if concurrency > 1 else single HUD), dashboard, single, or quiet (default: auto)',
    )

    args = parser.parse_args(argv)

    if args.command == 'eval':
        _suppress_warnings()
        models = _build_cli_model_registry(
            submission_dir=args.submission_dir,
            models_yaml_path=args.models_yaml,
        )
        config = EvalConfig(
            tasks_path=args.tasks,
            snapshots_dir=args.snapshots_dir,
            results_dir=args.results_dir,
            submission_dir=args.submission_dir,
            models=models,
            image=args.image,
            sandbox=args.sandbox,
            timeout_seconds=args.timeout_seconds,
            verification_timeout_seconds=args.verification_timeout_seconds,
            max_tool_calls=args.max_tool_calls,
            max_turns=args.max_turns,
            max_time_minutes=args.max_time_minutes,
            task_ids=args.task_ids,
            skip_agent_patch=args.skip_agent_patch,
            concurrency=args.concurrency,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            verbose=args.verbose,
            display_mode=args.display,
        )
        evaluator = Evaluator(config)
        result = asyncio.run(evaluator.run())
        if args.display != 'quiet':
            print(
                f'Evaluation complete: {result.resolved}/{result.total} resolved ({result.resolution_rate:.2%})'
            )
        return 0

    else:
        parser.print_help()
        return 0


if __name__ == '__main__':
    sys.exit(main())
