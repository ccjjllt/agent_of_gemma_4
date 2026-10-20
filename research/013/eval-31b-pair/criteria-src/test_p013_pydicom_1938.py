"""P013 dev criterion: empty SQ element has VM 1 and is_empty True."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


def test_empty_sequence_vm():
    from pydicom import Dataset
    from pydicom.dataelem import DataElement

    ds = Dataset()
    elem = DataElement(0x300A00B0, "SQ", [])
    ds.add(elem)
    assert elem.VM == 1
    assert elem.is_empty is True
