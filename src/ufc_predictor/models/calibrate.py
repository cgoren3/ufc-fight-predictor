from __future__ import annotations

import numpy as np
import pandas as pd


def expected_calibration_error(
    y_true: np.ndarray | pd.Series | list[float],
    y_prob: np.ndarray | pd.Series | list[float],
    n_bins: int = 10,
) -> float:
    y_true_array = np.asarray(y_true, dtype=float)
    y_prob_array = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(y_true_array)
    if total == 0:
        return float("nan")
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (y_prob_array >= left) & (y_prob_array < right)
        if right == 1.0:
            mask = (y_prob_array >= left) & (y_prob_array <= right)
        if not mask.any():
            continue
        confidence = y_prob_array[mask].mean()
        accuracy = y_true_array[mask].mean()
        ece += mask.mean() * abs(accuracy - confidence)
    return float(ece)


def calibration_curve_data(
    y_true: np.ndarray | pd.Series | list[float],
    y_prob: np.ndarray | pd.Series | list[float],
    n_bins: int = 10,
) -> pd.DataFrame:
    y_true_array = np.asarray(y_true, dtype=float)
    y_prob_array = np.asarray(y_prob, dtype=float)
    rows = []
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    for index, (left, right) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (y_prob_array >= left) & (y_prob_array < right)
        if right == 1.0:
            mask = (y_prob_array >= left) & (y_prob_array <= right)
        rows.append(
            {
                "bin": index,
                "left": left,
                "right": right,
                "count": int(mask.sum()),
                "mean_predicted_probability": float(y_prob_array[mask].mean()) if mask.any() else np.nan,
                "observed_win_rate": float(y_true_array[mask].mean()) if mask.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def calibrate_estimator(estimator, x_train, y_train, method: str = "sigmoid", cv: int = 3):
    """Wrap an estimator in CalibratedClassifierCV when enough data exists."""

    try:
        from sklearn.calibration import CalibratedClassifierCV
    except Exception as exc:  # pragma: no cover - dependency path
        raise RuntimeError("scikit-learn is required for calibration.") from exc

    values, counts = np.unique(np.asarray(y_train), return_counts=True)
    if len(values) < 2 or counts.min() < cv:
        estimator.fit(x_train, y_train)
        return estimator
    calibrated = CalibratedClassifierCV(estimator, method=method, cv=cv)
    calibrated.fit(x_train, y_train)
    return calibrated
