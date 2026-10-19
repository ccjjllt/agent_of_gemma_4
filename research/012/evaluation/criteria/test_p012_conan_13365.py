"""P012 dev criterion: broken subfolder custom command shows a friendly error, no crash."""
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_subfolder_import_error_surfaced(tmp_path, monkeypatch):
    import contextlib

    monkeypatch.setenv("CONAN_HOME", str(tmp_path / "home"))
    from conan.api.conan_api import ConanAPI
    from conan.cli.cli import Cli

    ext = tmp_path / "home" / "extensions" / "commands" / "mycompany"
    ext.mkdir(parents=True)
    (ext / "cmd_hello.py").write_text("import this_doesnt_exist\n")

    api = ConanAPI()
    cli = Cli(api)
    buf = io.StringIO()
    raised = None
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            cli.add_commands()
        except Exception as e:  # noqa: BLE001 - buggy behavior is a raw crash
            raised = "%s: %s" % (type(e).__name__, e)
    out = buf.getvalue()
    assert raised is None, "must not crash, got: %s" % raised
    assert "Error loading custom command mycompany.cmd_hello" in out, out[-500:]
