"""P013 dev criterion: Markup string methods work (_simple_escaping_wrapper defined)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


def test_markup_string_methods():
    from markupsafe import Markup

    assert Markup("ABC").lower() == "abc"
    assert Markup("abc").upper() == "ABC"
    assert Markup("  x  ").strip() == "x"
    assert Markup("aXb").replace("X", "-") == "a-b"
