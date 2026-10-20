"""P013 dev criterion: CLI filespec_parser reports missing files, no IndexError."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


def test_cli_nonexistent_file_error():
    import argparse

    import pytest

    from pydicom.cli.main import filespec_parser

    with pytest.raises(argparse.ArgumentTypeError) as excinfo:
        filespec_parser("XXxUN_sequenceXX.dcm")
    assert "XXxUN_sequenceXX.dcm" in str(excinfo.value)
