"""P012 dev criterion: StrictEnvironment works without jinja2_time; {% now %} renders."""
import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_no_jinja2_time_needed():
    try:
        import jinja2_time  # noqa: F401
    except ImportError:
        pass
    else:
        raise AssertionError("eval env must not provide jinja2_time for this task")
    from cookiecutter.environment import StrictEnvironment

    env = StrictEnvironment(context={})
    out = env.from_string("{% now 'utc', '%Y' %}").render()
    assert out == str(datetime.datetime.now(datetime.timezone.utc).year), out
