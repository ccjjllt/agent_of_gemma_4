"""P012 dev criterion: PrefixFileWrapper.read(-1) respects the current offset."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "p012_lib"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_prefix_read_negative_size():
    from io import BytesIO

    from feedparser.encodings import PrefixFileWrapper

    f = PrefixFileWrapper(b"abc", BytesIO(b"def"))
    assert f.read(2) == b"ab"
    assert f.read(-1) == b"cdef"
