"""P012 dev criterion: keep-project-on-failure flag keeps the project dir on hook failure."""
import os
import stat
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TEMPLATE = """{"proj": "demo"}"""
CONANFILE = "hello"


def _make_template(base):
    import json as _json

    tpl = os.path.join(base, "tpl")
    proj = os.path.join(tpl, "{{cookiecutter.proj}}")
    hooks = os.path.join(tpl, "hooks")
    os.makedirs(proj)
    os.makedirs(hooks)
    with open(os.path.join(tpl, "cookiecutter.json"), "w") as f:
        f.write('{"proj": "demo"}')
    with open(os.path.join(proj, "f.txt"), "w") as f:
        f.write("hello")
    hook = os.path.join(hooks, "post_gen_project.py")
    with open(hook, "w") as f:
        f.write("#!/usr/bin/env python3\nimport sys\nsys.exit(1)\n")
    os.chmod(hook, os.stat(hook).st_mode | stat.S_IEXEC)
    return tpl


def test_keep_project_on_failure(tmp_path):
    from cookiecutter.exceptions import FailedHookException
    from cookiecutter.main import cookiecutter

    tpl = _make_template(str(tmp_path))
    out1 = str(tmp_path / "o1")
    try:
        cookiecutter(tpl, no_input=True, output_dir=out1)
    except FailedHookException:
        pass
    assert not os.path.exists(os.path.join(out1, "demo")), "default must delete on hook failure"
    out2 = str(tmp_path / "o2")
    try:
        cookiecutter(tpl, no_input=True, output_dir=out2, keep_project_on_failure=True)
    except FailedHookException:
        pass
    assert os.path.exists(os.path.join(out2, "demo", "f.txt")), "keep flag must retain files"
