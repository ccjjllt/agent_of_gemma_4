"""P013 dev criterion: Markup.striptags removes comments without a trace."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


def test_striptags_removes_comment():
    from markupsafe import Markup

    assert Markup("x <!-- comment -->").striptags() == "x"
