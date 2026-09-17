from sfc_gdn2.metrics import geometry_metrics

def test_geometry_metrics():
    m = geometry_metrics("hilbert", 4, [4,8], 1.1, 17)
    assert 0 <= m["neighbor_recall_w4"] <= 1
    assert m["step_max"] >= m["step_mean"]
