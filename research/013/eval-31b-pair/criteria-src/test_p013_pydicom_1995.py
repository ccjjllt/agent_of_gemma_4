"""P013 dev criterion: comparing an unmapped SRT code returns False, no KeyError."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


def test_srt_unknown_mapping_eq_false():
    from pydicom.sr.coding import Code

    wrong_srt_code = Code("111729", "SRT", "Specimen storage")
    sct_code = Code("127790008", "SCT", "Staining")
    assert (wrong_srt_code == sct_code) is False
