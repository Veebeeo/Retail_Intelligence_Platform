"""Probabilistic customer lifetime value: BG/NBD + Gamma-Gamma.

K-means on RFM describes what a customer *has* done. It cannot say what they
will do next, and it has no notion of whether a quiet customer has churned or
is simply between purchases. That distinction is the whole problem in
non-contractual retail: nobody cancels, they just stop coming back.

The BG/NBD model (Fader, Hardie & Lee, 2005) handles it with two assumptions:

* While active, a customer buys at a Poisson rate ``lambda``, and ``lambda``
  varies across customers as Gamma(r, alpha).
* After any purchase a customer goes inactive with probability ``p``, and ``p``
  varies across customers as Beta(a, b).

From those, given a customer's (frequency, recency, tenure) you get both the
expected number of purchases in the next N days and the probability they are
still active. Gamma-Gamma (Fader & Hardie, 2013) then models spend per
transaction — it assumes spend is independent of frequency, which is checked
in :func:`spend_frequency_correlation` rather than assumed.

Implemented directly on scipy rather than via ``lifetimes``, which is
unmaintained and breaks on modern pandas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln, gammaln, hyp2f1

from retail_intel.logging_conf import get_logger

logger = get_logger(__name__)


@dataclass
class BGNBDParams:
    r: float
    alpha: float
    a: float
    b: float
    log_likelihood: float
    converged: bool

    def to_dict(self) -> dict:
        return {
            "r": round(self.r, 4), "alpha": round(self.alpha, 4),
            "a": round(self.a, 4), "b": round(self.b, 4),
            "log_likelihood": round(self.log_likelihood, 2), "converged": self.converged,
        }


@dataclass
class GammaGammaParams:
    p: float
    q: float
    v: float
    converged: bool

    def to_dict(self) -> dict:
        return {"p": round(self.p, 4), "q": round(self.q, 4), "v": round(self.v, 4),
                "converged": self.converged}


def build_cbs(df: pd.DataFrame, snapshot_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """Build the customer-by-sufficient-statistic summary these models need.

    * ``frequency`` — *repeat* purchases, so a one-off buyer scores 0. This is
      not the same "frequency" as in RFM (total orders), and conflating the two
      is the most common way to get BG/NBD wrong.
    * ``recency`` — age at the last purchase, in days from first purchase.
    * ``T`` — total observation time from first purchase to the snapshot.
    """
    snapshot_date = snapshot_date or df["invoice_date"].max() + pd.Timedelta(days=1)

    orders = (
        df.groupby(["customer_id", "invoice_no"])
        .agg(order_date=("invoice_date", "min"), order_value=("total_price", "sum"))
        .reset_index()
    )
    orders = orders[orders["order_value"] > 0]

    cbs = orders.groupby("customer_id").agg(
        first=("order_date", "min"),
        last=("order_date", "max"),
        n_orders=("invoice_no", "nunique"),
        monetary_value=("order_value", "mean"),
    )
    cbs["frequency"] = cbs["n_orders"] - 1
    cbs["recency"] = (cbs["last"] - cbs["first"]).dt.days.astype(float)
    cbs["T"] = (snapshot_date - cbs["first"]).dt.days.astype(float)

    out = cbs[["frequency", "recency", "T", "monetary_value", "n_orders"]].reset_index()
    logger.info(
        "CBS built for %d customers; %d have repeat purchases",
        len(out), int((out["frequency"] > 0).sum()),
    )
    return out


# --------------------------------------------------------------------------
# BG/NBD
# --------------------------------------------------------------------------
def _bgnbd_negative_ll(params: np.ndarray, x: np.ndarray, t_x: np.ndarray, T: np.ndarray) -> float:
    """Negative log-likelihood, equation (3) of Fader, Hardie & Lee (2005)."""
    r, alpha, a, b = np.exp(params)  # optimise in log space to keep them positive

    ln_a1 = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
    ln_a2 = betaln(a, b + x) - betaln(a, b)
    ln_a3 = -(r + x) * np.log(alpha + T)
    # The second term applies only to customers with at least one repeat.
    ln_a4 = np.where(
        x > 0,
        np.log(a) - np.log(b + np.maximum(x, 1) - 1) - (r + x) * np.log(alpha + t_x),
        -np.inf,
    )

    ll = ln_a1 + ln_a2 + np.logaddexp(ln_a3, ln_a4)
    if not np.isfinite(ll).all():
        return 1e10
    return -float(ll.sum())


def fit_bgnbd(cbs: pd.DataFrame, max_iter: int = 500) -> BGNBDParams:
    """Maximum-likelihood fit of the BG/NBD parameters."""
    x = cbs["frequency"].to_numpy(dtype=float)
    t_x = cbs["recency"].to_numpy(dtype=float)
    T = cbs["T"].to_numpy(dtype=float)

    best: tuple[float, np.ndarray] | None = None
    # The likelihood surface has local optima; try a few starts.
    for start in ([1.0, 1.0, 1.0, 1.0], [0.5, 10.0, 0.5, 2.0], [2.0, 50.0, 1.5, 5.0]):
        res = minimize(
            _bgnbd_negative_ll,
            np.log(start),
            args=(x, t_x, T),
            method="Nelder-Mead",
            options={"maxiter": max_iter * 4, "fatol": 1e-6, "xatol": 1e-6},
        )
        if best is None or res.fun < best[0]:
            best = (float(res.fun), res.x)
            converged = bool(res.success)

    assert best is not None
    r, alpha, a, b = np.exp(best[1])
    params = BGNBDParams(float(r), float(alpha), float(a), float(b), -best[0], converged)
    logger.info("BG/NBD fitted: %s", params.to_dict())
    return params


def predict_purchases(cbs: pd.DataFrame, params: BGNBDParams, days: int = 90) -> np.ndarray:
    """Expected number of purchases in the next ``days``.

    Equation (10) of Fader, Hardie & Lee (2005), using the Gaussian
    hypergeometric function.
    """
    r, alpha, a, b = params.r, params.alpha, params.a, params.b
    x = cbs["frequency"].to_numpy(dtype=float)
    t_x = cbs["recency"].to_numpy(dtype=float)
    T = cbs["T"].to_numpy(dtype=float)

    alive = probability_alive(cbs, params)

    # The closed form below is only valid for a > 1. Values of a below 1 give a
    # U-shaped Beta -- customers are polarised into "very likely to drop out
    # after a purchase" and "very unlikely" -- which is a perfectly reasonable
    # fit, but it flips the sign of the (a - 1) denominator and makes the
    # formula return negative expectations that then clip to zero. In that
    # regime use the still-alive expectation instead: the posterior purchase
    # rate (r + x) / (alpha + T), scaled by P(alive).
    if a <= 1.0 + 1e-6:
        logger.info(
            "BG/NBD fitted a=%.4f (<= 1), so the closed-form expectation is undefined. "
            "Using the posterior purchase rate weighted by P(alive) instead.", a,
        )
        return np.clip(days * (r + x) / (alpha + T) * alive, 0, None)

    first = (a + b + x - 1) / (a - 1)
    numer = 1 - ((alpha + T) / (alpha + T + days)) ** (r + x) * hyp2f1(
        r + x, b + x, a + b + x - 1, days / (alpha + T + days)
    )
    denom = 1 + np.where(x > 0, (a / (b + x - 1)) * ((alpha + T) / (alpha + t_x)) ** (r + x), 0.0)
    return np.clip(first * numer / denom, 0, None)


def probability_alive(cbs: pd.DataFrame, params: BGNBDParams) -> np.ndarray:
    """P(customer is still active | their history).

    A quiet customer with a historically slow rhythm may well be alive; a quiet
    customer who used to buy weekly probably is not. This separates them, which
    a recency cut-off cannot.
    """
    r, alpha, a, b = params.r, params.alpha, params.a, params.b
    x = cbs["frequency"].to_numpy(dtype=float)
    t_x = cbs["recency"].to_numpy(dtype=float)
    T = cbs["T"].to_numpy(dtype=float)

    ratio = np.where(x > 0, (a / (b + x - 1)) * ((alpha + T) / (alpha + t_x)) ** (r + x), 0.0)
    return np.clip(1.0 / (1.0 + ratio), 0, 1)


# --------------------------------------------------------------------------
# Gamma-Gamma
# --------------------------------------------------------------------------
def spend_frequency_correlation(cbs: pd.DataFrame) -> float:
    """Correlation between purchase count and average spend.

    Gamma-Gamma assumes these are independent. If the correlation is large
    (conventionally |rho| > 0.3) the assumption is violated and the resulting
    CLV is biased — worth checking rather than asserting.
    """
    repeat = cbs[cbs["frequency"] > 0]
    if len(repeat) < 3:
        return 0.0
    return float(np.corrcoef(repeat["frequency"], repeat["monetary_value"])[0, 1])


def _gamma_gamma_negative_ll(params: np.ndarray, x: np.ndarray, m: np.ndarray) -> float:
    p, q, v = np.exp(params)
    ll = (
        gammaln(p * x + q) - gammaln(p * x) - gammaln(q)
        + q * np.log(v) + (p * x - 1) * np.log(m) + (p * x) * np.log(x)
        - (p * x + q) * np.log(v + x * m)
    )
    if not np.isfinite(ll).all():
        return 1e10
    return -float(ll.sum())


def fit_gamma_gamma(cbs: pd.DataFrame) -> GammaGammaParams:
    """Fit the spend-per-transaction model on repeat customers only."""
    repeat = cbs[(cbs["frequency"] > 0) & (cbs["monetary_value"] > 0)]
    if len(repeat) < 10:
        raise ValueError(f"Need at least 10 repeat customers to fit Gamma-Gamma, got {len(repeat)}")

    rho = spend_frequency_correlation(cbs)
    if abs(rho) > 0.3:
        logger.warning(
            "Spend and frequency correlate at %.2f; Gamma-Gamma assumes independence, "
            "so predicted monetary values are biased.", rho,
        )

    x = repeat["frequency"].to_numpy(dtype=float)
    m = repeat["monetary_value"].to_numpy(dtype=float)

    best: tuple[float, np.ndarray] | None = None
    for start in ([1.0, 1.0, 1.0], [6.0, 4.0, 15.0], [2.0, 3.0, 5.0]):
        res = minimize(
            _gamma_gamma_negative_ll, np.log(start), args=(x, m),
            method="Nelder-Mead", options={"maxiter": 2000, "fatol": 1e-6},
        )
        if best is None or res.fun < best[0]:
            best = (float(res.fun), res.x)
            converged = bool(res.success)

    assert best is not None
    p, q, v = np.exp(best[1])
    params = GammaGammaParams(float(p), float(q), float(v), converged)
    logger.info("Gamma-Gamma fitted: %s (spend/frequency rho=%.3f)", params.to_dict(), rho)
    return params


def predict_average_value(cbs: pd.DataFrame, params: GammaGammaParams) -> np.ndarray:
    """Conditional expected spend per transaction.

    This shrinks each customer's observed average towards the population mean,
    by an amount that depends on how many transactions we have seen from them —
    which is why it beats simply using their historical average for customers
    with one or two orders.
    """
    p, q, v = params.p, params.q, params.v
    x = cbs["frequency"].to_numpy(dtype=float)
    m = cbs["monetary_value"].to_numpy(dtype=float)

    population_mean = (p * v) / max(q - 1, 1e-6)
    individual = (p * x * m + v * p) / (p * x + q - 1)
    # Customers with no repeat purchase carry no individual signal.
    return np.where(x > 0, individual, population_mean)


def predict_clv(
    cbs: pd.DataFrame,
    bgnbd: BGNBDParams,
    gg: GammaGammaParams,
    days: int = 90,
    annual_discount_rate: float = 0.10,
) -> pd.DataFrame:
    """Expected value per customer over the next ``days``, discounted.

    Discounting matters once the horizon is long: revenue a year out is not
    worth the same as revenue this month, and ignoring that overstates the case
    for expensive retention campaigns.
    """
    purchases = predict_purchases(cbs, bgnbd, days)
    avg_value = predict_average_value(cbs, gg)
    alive = probability_alive(cbs, bgnbd)

    discount = 1 / (1 + annual_discount_rate) ** (days / 365.0)
    clv = purchases * avg_value * discount

    return pd.DataFrame(
        {
            "customer_id": cbs["customer_id"].to_numpy(),
            f"predicted_purchases_{days}d": np.round(purchases, 4),
            "predicted_avg_order_value": np.round(avg_value, 2),
            f"predicted_clv_{days}d": np.round(clv, 2),
            "probability_alive": np.round(alive, 4),
            "churn_probability": np.round(1 - alive, 4),
        }
    )


def fit_predict(
    df: pd.DataFrame, days: int = 90, snapshot_date: pd.Timestamp | None = None
) -> tuple[pd.DataFrame, dict]:
    """Convenience: transactions in, CLV table and fitted parameters out."""
    cbs = build_cbs(df, snapshot_date)
    bgnbd = fit_bgnbd(cbs)
    gg = fit_gamma_gamma(cbs)
    clv = predict_clv(cbs, bgnbd, gg, days)
    return clv, {
        "bgnbd": bgnbd.to_dict(),
        "gamma_gamma": gg.to_dict(),
        "horizon_days": days,
        "spend_frequency_corr": round(spend_frequency_correlation(cbs), 4),
        "n_customers": len(cbs),
    }
