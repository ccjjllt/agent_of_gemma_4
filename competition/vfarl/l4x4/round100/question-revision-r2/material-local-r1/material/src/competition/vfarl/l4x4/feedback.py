"""Prefix-only focal selection from actual tool feedback.

These categories guide collection, never assign task rewards. Command matching
is a scheduling heuristic; only the unchanged verifier can produce 0/1 labels.
"""
import re
import shlex


FOCAL_SELECTION_RULES = {
    'any-tool-failure': 'first-real-tool-failure-with-recovery-budget-before-action',
    'execution-recovery': 'first-execution-recovery-feedback-with-budget-before-action-v1',
}


def classify_tool_feedback(tool, arguments, response, ledger, *, production_change=None):
    """Read the real response, including stdout hidden by stderr summaries."""
    response = response if isinstance(response, dict) else {}
    failed = response.get('status') == 'error'
    error_type = str(response.get('error_type', ''))
    mutations = ledger.get('production_mutation_observations', 0)
    # A successful tool name/diff is insufficient: it may only write reports
    # or protected files. Missing host content evidence remains unknown.
    mutation = (production_change.get('changed') if production_change is not None else
                None if tool in {'run_command', 'edit_file', 'write_file'} else False)
    result = dict(category='tool_success', focal_eligible=False,
        error_type=error_type, failed=failed,
        mutation_observation=mutation, prior_mutation_observations=mutations,
        interpretation='Prefix scheduling only; no semantic repair or terminal reward inferred')
    if production_change is not None:
        result['production_change'] = production_change
    if not failed:
        return result
    if any(word in error_type.lower() for word in ('budget', 'timeout', 'timeexceeded')):
        result['category'] = 'budget_or_timeout_feedback'
        return result
    if tool != 'run_command' or error_type != 'CommandError':
        result.update(category='tool_execution_error', focal_eligible=True)
        return result

    details = response.get('details') or {}
    if not isinstance(details, dict):
        details = {}
    output = '\n'.join(str(value) for value in (
        details.get('stdout', ''), details.get('stderr', ''),
        response.get('error_message', ''), response.get('error', '')))
    command = str(arguments.get('command', ''))
    try:
        # Preserve quoted tokens so grep 'a|b' is not mistaken for a shell
        # pipeline. This is lexical classification, never shell execution.
        lexer = shlex.shlex(command, posix=False, punctuation_chars=True)
        lexer.whitespace_split = True
        shell_tokens = list(lexer)
    except ValueError:
        shell_tokens = []
    segments = [[]]
    pipeline = False
    controls = {token for token in shell_tokens
                if token and all(character in '|&;()<>' for character in token)}
    for token in shell_tokens:
        if token in controls:
            pipeline = pipeline or token in {'|', '|&'}
            segments.append([])
        else:
            segments[-1].append(token)
    heads = []
    for segment in segments:
        if segment:
            try:
                head = shlex.split(segment[0])
            except ValueError:
                head = []
            if head:
                heads.append(head[0].rsplit('/', 1)[-1])
    # Search exit 1 is only a clear no-match for a simple command. A pipeline
    # containing a search is deferred as ambiguous, not promoted to an agent
    # execution error merely because the final exit is nonzero.
    diagnostics = [line for line in str(details.get('stderr', '')).splitlines()
                   if line.strip() and 'setlocale:' not in line]
    simple_search = (bool(heads) and heads[0] in {'rg', 'grep'}
        and not any(token in controls for token in shell_tokens) and '\n' not in command)
    if simple_search and details.get('exit_code') == 1 and not diagnostics:
        result['category'] = 'search_miss'
        return result
    if (pipeline and any(head in {'rg', 'grep'} for head in heads)
            and details.get('exit_code') == 1 and not diagnostics):
        result['category'] = 'ambiguous_search_failure'
        return result
    test_command = re.search(r'(?<![\w.-])(?:pytest|py\.test|unittest|tox|nox)(?![\w.-])', command)
    test_output = re.search(r'(?:FAILED\s+[^\n]*test|(?:FAIL|ERROR):\s+test|ERROR collecting)', output)
    if test_command or test_output:
        if mutations or mutation is True:
            result.update(category='repair_attempt_failed', focal_eligible=True)
        elif mutation is None:
            # An unreadable state or a newly emitted report is not evidence of
            # a repair attempt. Historical unknowns never grant eligibility.
            result['category'] = 'mutation_unknown_test_failure'
        else:
            result['category'] = 'diagnostic_test_failure'
        return result
    result.update(category='command_execution_error', focal_eligible=True)
    return result


def focal_feedback_ready(ledger, trigger):
    if trigger not in FOCAL_SELECTION_RULES:
        raise ValueError('Unknown focal trigger: ' + str(trigger))
    if not ledger.get('last_tool_error'):
        return False
    if trigger == 'any-tool-failure':
        return True
    return ledger.get('last_tool_feedback', {}).get('classification', {}).get('focal_eligible') is True
