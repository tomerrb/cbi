import numpy as np
import pandas as pd


def _minmax_scale_df(df: pd.DataFrame, bounds: dict):
    """
    bounds[col] = (lo, hi)
    scales each specified column to [0,1]
    """
    out = df.copy()
    for col, (lo, hi) in bounds.items():
        if hi <= lo:
            raise ValueError(f"Invalid bounds for {col}: {(lo, hi)}")
        out[col] = np.clip(out[col].astype(float), lo, hi)
        out[col] = (out[col] - lo) / (hi - lo)
    return out


def _build_normalized_marginal(df: pd.DataFrame, cols: list[str]):
    """
    Build an empirical normalized marginal over the specified columns.

    Returns a pandas Series indexed by tuples of values in cols, where the
    values sum to 1.
    """
    if not cols:
        raise ValueError("cols must be non-empty")

    marginal = df.groupby(cols, dropna=False).size().astype(float)
    total = float(marginal.sum())
    if total <= 0:
        raise ValueError("Cannot build a marginal from an empty dataframe")
    return marginal / total


def _estimate_propensity_from_marginal(
    marginal: pd.Series,
    treatment_col: str,
    adjustment_set: list[str],
) -> dict:
    """
    Estimate \bar{T}(z) = P(T=1 | Z=z) from the normalized marginal.

    Returns a dict mapping z-tuples to propensity values.
    """
    if treatment_col not in marginal.index.names:
        raise ValueError("treatment_col must be part of the marginal index")

    if not adjustment_set:
        # No confounders: just estimate marginal treatment probability.
        p_t1 = 0.0
        for idx, prob in marginal.items():
            if not isinstance(idx, tuple):
                idx = (idx,)
            row = dict(zip(marginal.index.names, idx))
            if float(row[treatment_col]) == 1.0:
                p_t1 += float(prob)
        return {(): p_t1}

    propensity_num = {}
    propensity_den = {}

    for idx, prob in marginal.items():
        if not isinstance(idx, tuple):
            idx = (idx,)
        row = dict(zip(marginal.index.names, idx))
        z = tuple(row[c] for c in adjustment_set)
        propensity_den[z] = propensity_den.get(z, 0.0) + float(prob)
        if float(row[treatment_col]) == 1.0:
            propensity_num[z] = propensity_num.get(z, 0.0) + float(prob)

    out = {}
    for z, den in propensity_den.items():
        out[z] = 0.0 if den <= 0 else propensity_num.get(z, 0.0) / den
    return out


def claim_fwl_ate_estimator(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    adjustment_set: list[str],
    bounds: dict[str, tuple[float, float]],
) -> float:
    """
    Approximate CLAIM-style FWL ATE estimator.

    This implements the estimator

        \hat{\tau}^r(D) = (1 / v) * sum_t M_r(D)[t] * t_Y * (t_T - \bar{T}(t_Z))

    over the attribute set r = {T, Y} U Z, where:
      - M_r(D) is the normalized marginal over (T, Y, Z),
      - \bar{T}(z) = P(T=1 | Z=z),
      - v = E[(T - \bar{T}(Z))^2].

    All used variables are first scaled to [0,1] according to `bounds`.

    Parameters
    ----------
    df:
        Input dataframe.
    treatment_col:
        Name of the binary treatment column.
    outcome_col:
        Name of the outcome column.
    adjustment_set:
        List of adjustment / confounder columns. This is accepted explicitly
        and is not inferred.
    bounds:
        Mapping from column name to (lo, hi) scaling bounds.

    Returns
    -------
    float
        Estimated treatment effect in scaled outcome units.
    """
    if treatment_col == outcome_col:
        raise ValueError("treatment_col and outcome_col must be different")

    if len(set(adjustment_set)) != len(adjustment_set):
        raise ValueError("adjustment_set must not contain duplicate columns")

    if treatment_col in adjustment_set or outcome_col in adjustment_set:
        raise ValueError("adjustment_set must not contain treatment or outcome columns")

    cols = [treatment_col, outcome_col] + adjustment_set
    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing dataframe columns: {missing_cols}")

    missing_bounds = [c for c in cols if c not in bounds]
    if missing_bounds:
        raise ValueError(f"Missing bounds for columns: {missing_bounds}")

    data = _minmax_scale_df(df[cols], {c: bounds[c] for c in cols})

    # Build normalized marginal M_r(D).
    marginal = _build_normalized_marginal(data, cols)

    # Estimate propensity \bar{T}(z).
    propensity = _estimate_propensity_from_marginal(
        marginal=marginal,
        treatment_col=treatment_col,
        adjustment_set=adjustment_set,
    )

    # Compute v = E[(T - \bar{T}(Z))^2].
    v = 0.0
    numerator = 0.0

    for idx, prob in marginal.items():
        if not isinstance(idx, tuple):
            idx = (idx,)
        row = dict(zip(marginal.index.names, idx))

        t_val = float(row[treatment_col])
        y_val = float(row[outcome_col])
        z_val = tuple(row[c] for c in adjustment_set) if adjustment_set else ()

        t_bar = float(propensity.get(z_val, 0.0))
        residual_t = t_val - t_bar

        v += float(prob) * (residual_t ** 2)
        numerator += float(prob) * y_val * residual_t

    if v <= 0:
        raise ValueError(
            "Degenerate treatment variance v=0. Consider clipping v or checking positivity."
        )

    ate_hat = numerator / v
    return float(ate_hat)
