"""P013 dev criterion: markupsafe imports cleanly (re defined) and escapes."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


def test_import_and_escape():
    from markupsafe import Markup

    assert Markup.escape("<a>") == "&lt;a&gt;"
