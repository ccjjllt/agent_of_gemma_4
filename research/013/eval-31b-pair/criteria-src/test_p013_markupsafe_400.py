"""P013 dev criterion: Markup.escape escapes HTML special characters."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


def test_escape_html():
    from markupsafe import Markup

    assert Markup.escape("<script>alert('XSS')</script>") == "&lt;script&gt;alert(&#39;XSS&#39;)&lt;/script&gt;"
