"""P013 dev criterion: Dataset.from_json converts UN VR InlineBinary to known VR."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


def test_from_json_un_vr():
    from pydicom import Dataset
    from pydicom.sequence import Sequence

    ds1 = Dataset.from_json(
        {"300A0180": {"vr": "UN", "InlineBinary": "/v8A4B4AAAAYAABRBAAAAEhGUAKMIIBAgAAADEgCjCyAQAAAAA="}}
    )
    assert "PatientSetupSequence" in ds1
    assert ds1[0x300A0180].VR == "SQ"
    assert isinstance(ds1[0x300A0180].value, Sequence)

    ds2 = Dataset.from_json({"00185100": {"vr": "UN", "InlineBinary": "SEZTIA=="}})
    assert "PatientPosition" in ds2
    assert ds2.PatientPosition == "HFS"
