"""Evaluation metrics, exactly as defined in Section IV-C of the paper.

RMSE(x, x^) = sqrt( (1/n) sum (x_i - x^_i)^2 )
MAE (x, x^) = (1/n) sum |x_i - x^_i|
MAPE(x, x^) = (1/n) sum |(x_i - x^_i) / x_i|          (no factor 100 in the paper)

The paper's MAPE is undefined when x_i == 0 (zero demand/supply is common).
We therefore never silently divide by zero: the policy is explicit
(see docs/ASSUMPTIONS.md, A-05).
"""
from __future__ import annotations
import numpy as np


def _pair(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("empty input")
    return y_true, y_pred


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = _pair(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = _pair(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true, y_pred, zero_policy: str = "mask", percent: bool = False) -> float:
    """zero_policy: 'mask'  -> ignore entries with y_true == 0 (reported by mape_coverage)
                    'strict'-> raise if any y_true == 0
    """
    y_true, y_pred = _pair(y_true, y_pred)
    nz = y_true != 0
    if zero_policy == "strict":
        if not nz.all():
            raise ValueError("MAPE undefined: y_true contains zeros")
    elif zero_policy != "mask":
        raise ValueError("zero_policy must be 'mask' or 'strict'")
    if not nz.any():
        raise ValueError("MAPE undefined: all y_true are zero")
    val = float(np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])))
    return val * 100.0 if percent else val


def mape_coverage(y_true) -> float:
    """Fraction of entries actually used by mape(zero_policy='mask')."""
    y_true = np.asarray(y_true, dtype=float)
    return float(np.mean(y_true != 0))
