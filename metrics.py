"""Independent NumPy/SciPy/sklearn implementations of reference HI formulas.

Reference commit: 157ebfae7baeee1e0cd52180f8e58ead3dcb448e.
"""

import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression


def mae(predicted, reference):
    return float(np.mean(np.abs(np.asarray(predicted) - np.asarray(reference))))


def mape(predicted, reference):
    reference, predicted = np.asarray(reference), np.asarray(predicted)
    valid = reference != 0
    return float(100 * np.mean(np.abs((reference[valid] - predicted[valid]) / reference[valid])))


def monotonicity(values):
    differences = np.diff(np.asarray(values))
    positive = np.count_nonzero(differences > 0)
    nonpositive = np.count_nonzero(differences <= 0)
    return float(abs(nonpositive - positive) / len(differences))


def spearman(values, cycles):
    return float(spearmanr(values, cycles).statistic)


def failure_consistency(groups):
    ends = np.array([values[-1] for values in groups])
    drops = np.array([values[0] - values[-1] for values in groups])
    return float(np.exp(-ends.std(ddof=0) / drops.mean()))


def transformed_mi(values, rul, seed):
    information = mutual_info_regression(np.asarray(values)[:, None], np.asarray(rul), random_state=seed)[0]
    return float(1 - np.exp(-information))


def result(value, count, reason=None):
    if not np.isfinite(value):
        raise ValueError("Nonfinite metric result; a degenerate-input policy must be confirmed")
    return {"value": float(value), "count": int(count), "reason": reason}


def evaluate_metrics(predicted, reference, units, cycles, rul, seed):
    arrays = [np.asarray(values).reshape(-1) for values in (predicted, reference, units, cycles, rul)]
    if not arrays[0].size or any(len(values) != len(arrays[0]) or not np.isfinite(values).all() for values in arrays):
        raise ValueError("Metrics need nonempty matching finite arrays")
    predicted, reference, units, cycles, rul = arrays
    if not (units == np.floor(units)).all() or not (cycles == np.floor(cycles)).all():
        raise ValueError("Metric unit/cycle keys must be integers")
    if len(set(zip(units.tolist(), cycles.tolist()))) != len(units):
        raise ValueError("Duplicate metric unit/cycle keys")
    order = np.lexsort((cycles, units))
    predicted, reference, units, cycles, rul = [values[order] for values in arrays]
    per_unit, groups = {}, []
    for unit in np.unique(units):
        indices = np.flatnonzero(units == unit)
        indices = indices[np.argsort(cycles[indices])]
        p, g, c, r = predicted[indices], reference[indices], cycles[indices], rul[indices]
        nonzero = np.count_nonzero(g != 0)
        if len(p) <= 3 or nonzero == 0 or np.ptp(p) == 0 or np.ptp(r) == 0:
            raise ValueError("Degenerate per-unit metrics: handling awaits user confirmation")
        per_unit[str(int(unit))] = {"mae": result(mae(p, g), len(p)), "mape": result(mape(p, g), nonzero),
                                  "monotonicity": result(monotonicity(p), len(p) - 1),
                                  "spearman": result(spearman(p, c), len(p)),
                                  "mi": result(transformed_mi(p, r, seed), len(p))}
        groups.append(p)
    if len(groups) < 2 or np.mean([group[0] - group[-1] for group in groups]) == 0:
        raise ValueError("Degenerate failure consistency: handling awaits user confirmation")
    nonzero = np.count_nonzero(reference != 0)
    overall = {"mae": result(mae(predicted, reference), len(predicted)),
               "mape": result(mape(predicted, reference), nonzero),
               "monotonicity": result(np.mean([entry["monotonicity"]["value"] for entry in per_unit.values()]), len(groups)),
               "trend": result(abs(np.mean([entry["spearman"]["value"] for entry in per_unit.values()])), len(groups)),
               "failure_consistency": result(failure_consistency(groups), len(groups)),
               "mi": result(np.mean([entry["mi"]["value"] for entry in per_unit.values()]), len(groups))}
    if overall["failure_consistency"]["value"] > 1:
        overall["failure_consistency"]["reason"] = "Reference formula exceeds 1 because mean start-minus-end HI is negative; no clipping"
    return {"overall": overall, "per_unit": per_unit}
