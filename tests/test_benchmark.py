import math, pytest
torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
from semidec.delay.benchmark import measure_layer_time_ms, measure_mp_time_ms
from semidec.model.hetgnn_lstm import RELATIONS

EI = {r: torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]) for r in RELATIONS}

def test_timings_are_positive_and_finite():
    for t in (measure_layer_time_ms(EI, 3, repeats=3, warmup=1), measure_mp_time_ms(EI, 3, repeats=3, warmup=1)):
        assert t > 0 and math.isfinite(t)

def test_bad_repeats():
    with pytest.raises(ValueError):
        measure_layer_time_ms(EI, 3, repeats=0)
