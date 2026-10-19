"""P012 dev criterion: `conan source` resolves python_requires via remotes."""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def _conan(args, home):
    env = dict(os.environ, CONAN_HOME=str(home))
    return subprocess.run(
        [sys.executable, "-m", "conans.conan"] + args,
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )


def test_source_with_remote_python_requires(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    store = tmp_path / "store" / "recipes" / "py_req2" / "all"
    store.mkdir(parents=True)
    _write(
        str(tmp_path / "store" / "recipes" / "py_req2" / "config.yml"),
        'versions:\n  "1.0":\n    folder: all\n',
    )
    _write(
        str(store / "conanfile.py"),
        'from conan import ConanFile\n'
        "class PyReq2Conan(ConanFile):\n"
        '    name = "py_req2"\n'
        '    version = "1.0"\n',
    )
    consumer = tmp_path / "consumer"
    consumer.mkdir()
    _write(
        str(consumer / "conanfile.py"),
        'from conan import ConanFile\n'
        "class ConsumerConan(ConanFile):\n"
        '    name = "consumer"\n'
        '    version = "1.0"\n'
        '    python_requires = "py_req2/1.0"\n'
        "    def source(self):\n"
        '        self.output.info("Running source in consumer")\n',
    )
    r = _conan(
        ["remote", "add", "localstore", str(tmp_path / "store"),
         "--type", "local-recipes-index"],
        home,
    )
    assert r.returncode == 0, r.stderr[-500:]
    r = _conan(["remote", "disable", "conancenter"], home)
    assert r.returncode == 0, r.stderr[-500:]
    r = _conan(["source", str(consumer)], home)
    assert "Running source in consumer" in (r.stdout + r.stderr), (
        r.stdout[-800:] + r.stderr[-800:]
    )
