"""P012 dev criterion: schema None must generate {"const": None}, not {"type": None}."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_none_uses_const():
    from schema import Schema

    s = Schema({"test": None})
    prop = s.json_schema("my-id")["properties"]["test"]
    assert prop == {"const": None}, prop
