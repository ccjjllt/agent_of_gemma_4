"""P013 dev criterion: Dataset holding a LUT Descriptor element pickles cleanly."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


def test_dataelement_picklable():
    import pickle

    from pydicom import Dataset
    from pydicom.dataelem import DataElement

    ds = Dataset()
    ds["LUTDescriptor"] = DataElement(0x00283002, "LO", [1, 2])
    ds2 = pickle.loads(pickle.dumps(ds))
    assert list(ds2["LUTDescriptor"].value) == [1, 2]
