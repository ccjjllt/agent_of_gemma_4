"""P013 dev criterion: HTJ2K transfer syntax UIDs are defined on import."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


def test_htj2k_uid_defined():
    from pydicom.uid import HTJ2K, HTJ2KLossless, HTJ2KLosslessRPCL, UID

    for uid in (HTJ2KLossless, HTJ2KLosslessRPCL, HTJ2K):
        assert isinstance(uid, UID)
        assert len(str(uid)) > 0
