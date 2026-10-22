import numpy as np, pytest
from semidec.metrics import rmse, mae, mape, mape_coverage

def test_hand_values():
    y = [2.0, 4.0, 8.0]; p = [3.0, 4.0, 6.0]
    assert mae(y, p) == pytest.approx((1 + 0 + 2) / 3)
    assert rmse(y, p) == pytest.approx(np.sqrt((1 + 0 + 4) / 3))
    assert mape(y, p) == pytest.approx((0.5 + 0 + 0.25) / 3)
    assert mape(y, p, percent=True) == pytest.approx(100 * (0.5 + 0 + 0.25) / 3)

def test_mape_zero_policy():
    y = [0.0, 2.0]; p = [1.0, 1.0]
    assert mape(y, p) == pytest.approx(0.5)            # zero entry masked
    assert mape_coverage(y) == 0.5
    with pytest.raises(ValueError): mape(y, p, zero_policy="strict")
    with pytest.raises(ValueError): mape([0.0], [1.0])

def test_shape_mismatch():
    with pytest.raises(ValueError): rmse([1, 2], [1])
