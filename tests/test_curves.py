import numpy as np
import pytest
from sfc_gdn2.curves import order

@pytest.mark.parametrize("name", ["raster","snake","morton","hilbert","random"])
def test_curve_is_permutation(name):
    p = order(name, 8, 17)
    assert len(p) == 8**3
    assert np.array_equal(np.sort(p), np.arange(8**3))

def test_hilbert_requires_power_of_two():
    with pytest.raises(ValueError): order("hilbert", 6)
