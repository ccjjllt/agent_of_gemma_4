"""P013 dev criterion: absolute paths raise ValueError, not NotImplementedError."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


def test_absolute_path_value_error():
    import pytest

    from pydicom.data import get_charset_files, get_testdata_files

    with pytest.raises(ValueError):
        get_testdata_files("/absolute/path/to/file.dcm")
    with pytest.raises(ValueError):
        get_charset_files("/absolute/path/to/file.dcm")
