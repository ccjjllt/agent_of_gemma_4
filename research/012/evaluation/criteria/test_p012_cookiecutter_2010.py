"""P012 dev criterion: nested-dict config merges keep new subkeys."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_nested_dict_merge_keeps_new_keys():
    from cookiecutter.generate import apply_overwrites_to_context

    context = {"mainkey": {"subkey1": "value1", "subkey2": "value2"}}
    overwrite = {
        "mainkey": {
            "subkey1": "value1",
            "subkey2": "value2",
            "subkey3": "value3",
            "subkey4": "value4",
        }
    }
    apply_overwrites_to_context(context, overwrite)
    assert context["mainkey"] == {
        "subkey1": "value1",
        "subkey2": "value2",
        "subkey3": "value3",
        "subkey4": "value4",
    }, context["mainkey"]
