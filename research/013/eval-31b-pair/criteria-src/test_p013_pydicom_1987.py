"""P013 dev criterion: DataElement repr reflects contents, like str does."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


def test_repr_reflects_contents():
    from pydicom import Dataset
    from pydicom.dataelem import DataElement

    ds = Dataset()
    elem = DataElement(0x300A00B0, "SQ", [Dataset()])
    ds.add(elem)
    assert repr(elem) == str(elem)
    assert "300A" in repr(elem)
