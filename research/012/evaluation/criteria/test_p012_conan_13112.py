"""P012 dev criterion: `conan remote add --force` accepts duplicate URLs."""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _conan(args, home):
    env = dict(os.environ, CONAN_HOME=str(home))
    return subprocess.run(
        [sys.executable, "-m", "conans.conan"] + args,
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )


def test_remote_add_force(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    r = _conan(["remote", "add", "r1", "http://url1"], home)
    assert r.returncode == 0, r.stderr[-500:]
    r = _conan(["remote", "add", "r2", "http://url1"], home)
    assert r.returncode != 0, "duplicate URL without --force must fail"
    r = _conan(["remote", "add", "--force", "r2", "http://url1"], home)
    assert r.returncode == 0, r.stderr[-500:] + r.stdout[-500:]
    r = _conan(["remote", "list"], home)
    assert "r2: http://url1" in r.stdout, r.stdout[-500:]
