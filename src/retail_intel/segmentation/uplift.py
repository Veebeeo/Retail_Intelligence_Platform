"""Uplift modelling: who to target, not who is valuable.

The distinction this module exists to make. A CLV or RFM model ranks customers
by how much they are worth. Targeting the top of that ranking wastes most of
the budget, because the highest-value customers include a large group who would
have bought anyway — the discount sent to them is pure margin given away.

Uplift models the *causal* effect of the campaign on each individual:

    uplift(x) = P(buy | treated, x) - P(buy | not treated, x)

which sorts customers into four groups. Only the persuadables are worth
spending on; the sure things buy regardless, the lost causes never respond, and
the sleeping dogs are actively annoyed by contact.

**On the data.** Estimating this honestly requires a randomised holdout —
customers who were deliberately *not* contacted. The Online Retail II dataset
has no campaign log and no control group, so there is nothing to estimate from.
Rather than pretend otherwise, :func:`simulate_campaign` generates a labelled
experiment with a known ground-truth effect, and the evaluation below measures
whether the estimator recovers it. That makes this a validated implementation
waiting for real campaign data, not a result claimed from data that cannot
support it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from retail_intel.logging_conf import get_logger

logger = get_logger(__name__)

SEGMENT_NAMES = {
    "persuadable": "Persuadable — buys only if contacted. Target these.",
    "sure_thing": "Sure Thing — buys either way. Contacting them gives away margin.",
    "lost_cause": "Lost Cause — never buys. Contacting them wastes spend.",
    "sleeping_dog": "Sleeping Dog — contact makes them less likely to buy. Exclude.",
}


@dataclass
class UpliftEvaluation:
    qini_coefficient: float
    auuc: float
    uplift_at_10pct: float
    uplift_at_30pct: float
    overall_ate: float
    n_treated: int
    n_control: int

    def to_dict(self) -> dict:
        return {
            "qini_coefficient": round(self.qini_coefficient, 4),
            "auuc": round(self.auuc, 4),
            "uplift_at_10pct": round(self.uplift_at_10pct, 4),
            "uplift_at_30pct": round(self.uplift_at_30pct, 4),
            "overall_ate": round(self.overall_ate, 4),
            "n_treated": self.n_treated,
            "n_control": self.n_control,
        }


class TLearner:
    """Two-model uplift estimator.

    Fits one response model on the treated arm and one on the control arm, then
    takes the difference of their predicted probabilities. It is the simplest
    credible approach and the right default: the two arms can have genuinely
    different response surfaces, which a single model with a treatment flag
    tends to smooth away when the effect is small relative to the main effect.

    Its known weakness is that two independent fits each carry their own error,
    and the difference of two noisy estimates is noisier still — so it wants a
    reasonably large control arm.
    """

    def __init__(self, random_state: int = 42, **model_kwargs) -> None:
        defaults = {"n_estimators": 150, "max_depth": 3, "learning_rate": 0.05}
        self.model_kwargs = {**defaults, **model_kwargs, "random_state": random_state}
        self.treated_model: GradientBoostingClassifier | None = None
        self.control_model: GradientBoostingClassifier | None = None
        self.feature_names: list[str] | None = None

    def fit(self, X: pd.DataFrame, treatment: np.ndarray, outcome: np.ndarray) -> TLearner:
        treatment = np.asarray(treatment).astype(int)
        outcome = np.asarray(outcome).astype(int)
        self.feature_names = list(X.columns)

        for arm, name in ((1, "treated"), (0, "control")):
            mask = treatment == arm
            if mask.sum() < 20:
                raise ValueError(f"Only {mask.sum()} rows in the {name} arm; need at least 20")
            if len(np.unique(outcome[mask])) < 2:
                raise ValueError(f"The {name} arm has only one outcome class")
            model = GradientBoostingClassifier(**self.model_kwargs)
            model.fit(X[mask], outcome[mask])
            setattr(self, f"{name}_model", model)

        logger.info(
            "T-learner fitted: %d treated, %d control", int(treatment.sum()), int((1 - treatment).sum())
        )
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        if self.treated_model is None or self.control_model is None:
            raise RuntimeError("call fit() first")
        return (
            self.treated_model.predict_proba(X)[:, 1] - self.control_model.predict_proba(X)[:, 1]
        )

    def classify(self, X: pd.DataFrame, threshold: float = 0.01) -> pd.Series:
        """Bucket customers into the four uplift archetypes."""
        if self.treated_model is None or self.control_model is None:
            raise RuntimeError("call fit() first")
        p_treated = self.treated_model.predict_proba(X)[:, 1]
        p_control = self.control_model.predict_proba(X)[:, 1]
        uplift = p_treated - p_control

        labels = np.full(len(X), "lost_cause", dtype=object)
        labels[uplift > threshold] = "persuadable"
        labels[uplift < -threshold] = "sleeping_dog"
        labels[(np.abs(uplift) <= threshold) & (p_control > 0.5)] = "sure_thing"
        return pd.Series(labels, index=X.index, name="uplift_segment")


def qini_curve(
    uplift_scores: np.ndarray, treatment: np.ndarray, outcome: np.ndarray, n_bins: int = 20
) -> pd.DataFrame:
    """Qini curve: incremental conversions gained by targeting the top k%.

    At each depth it compares the treated response rate against the control
    response rate *scaled to the same population size*, which is what makes the
    curve an estimate of incremental conversions rather than raw conversions.
    """
    order = np.argsort(-np.asarray(uplift_scores))
    t = np.asarray(treatment)[order].astype(int)
    y = np.asarray(outcome)[order].astype(int)
    n = len(t)

    rows = [{"fraction": 0.0, "n_targeted": 0, "qini": 0.0, "random": 0.0}]
    total_treated, total_control = t.sum(), (1 - t).sum()
    total_gain = (
        y[t == 1].sum() - y[t == 0].sum() * (total_treated / total_control)
        if total_control > 0
        else 0.0
    )

    for i in range(1, n_bins + 1):
        k = max(1, int(n * i / n_bins))
        tk, yk = t[:k], y[:k]
        n_t, n_c = tk.sum(), (1 - tk).sum()
        gain = yk[tk == 1].sum() - (yk[tk == 0].sum() * (n_t / n_c) if n_c > 0 else 0.0)
        rows.append(
            {
                "fraction": k / n,
                "n_targeted": k,
                "qini": float(gain),
                "random": float(total_gain * k / n),
            }
        )
    return pd.DataFrame(rows)


def evaluate(
    uplift_scores: np.ndarray, treatment: np.ndarray, outcome: np.ndarray
) -> UpliftEvaluation:
    """Score an uplift model.

    The Qini coefficient is the area between the model's curve and the random
    line, normalised. Above zero means the ranking carries real causal signal;
    at or below zero the model is no better than contacting people at random,
    however good its AUC might look.
    """
    curve = qini_curve(uplift_scores, treatment, outcome)
    t = np.asarray(treatment).astype(int)
    y = np.asarray(outcome).astype(int)

    area_model = np.trapezoid(curve["qini"], curve["fraction"])
    area_random = np.trapezoid(curve["random"], curve["fraction"])
    denom = abs(area_random) if abs(area_random) > 1e-9 else 1.0

    def uplift_at(frac: float) -> float:
        order = np.argsort(-np.asarray(uplift_scores))
        k = max(1, int(len(order) * frac))
        sel_t, sel_y = t[order][:k], y[order][:k]
        rate_t = sel_y[sel_t == 1].mean() if (sel_t == 1).any() else 0.0
        rate_c = sel_y[sel_t == 0].mean() if (sel_t == 0).any() else 0.0
        return float(rate_t - rate_c)

    ate = float(y[t == 1].mean() - y[t == 0].mean()) if (t == 0).any() else 0.0

    # Qini is a difference of two response rates estimated on progressively
    # smaller slices, so it is extremely noisy on small arms. Say so rather
    # than letting a meaningless coefficient be quoted as a result.
    if min(t.sum(), (1 - t).sum()) < 500:
        logger.warning(
            "Qini estimated on %d treated / %d control. Below roughly 500 per arm the "
            "coefficient is dominated by sampling noise and should not be read as a "
            "performance figure.", int(t.sum()), int((1 - t).sum()),
        )

    return UpliftEvaluation(
        qini_coefficient=float((area_model - area_random) / denom),
        auuc=float(area_model),
        uplift_at_10pct=uplift_at(0.10),
        uplift_at_30pct=uplift_at(0.30),
        overall_ate=ate,
        n_treated=int(t.sum()),
        n_control=int((1 - t).sum()),
    )


def campaign_economics(
    uplift_scores: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    margin_per_conversion: float = 25.0,
    cost_per_contact: float = 0.50,
) -> pd.DataFrame:
    """Profit of targeting the top k% by uplift, versus contacting everyone.

    The output of the whole exercise: the depth at which incremental margin
    stops covering contact cost. Targeting everyone is almost never optimal.
    """
    curve = qini_curve(uplift_scores, treatment, outcome, n_bins=20)
    curve = curve[curve["n_targeted"] > 0].copy()
    curve["incremental_conversions"] = curve["qini"]
    curve["gross_margin"] = curve["incremental_conversions"] * margin_per_conversion
    curve["contact_cost"] = curve["n_targeted"] * cost_per_contact
    curve["net_profit"] = curve["gross_margin"] - curve["contact_cost"]
    curve["roi"] = np.where(
        curve["contact_cost"] > 0, curve["net_profit"] / curve["contact_cost"], np.nan
    )
    return curve[
        ["fraction", "n_targeted", "incremental_conversions", "gross_margin", "contact_cost",
         "net_profit", "roi"]
    ].round(3)


def simulate_campaign(
    features: pd.DataFrame, seed: int = 42, treatment_fraction: float = 0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a randomised campaign with a known heterogeneous effect.

    Returns ``(treatment, outcome, true_uplift)``. Used to validate the
    estimator: a correct implementation should recover a Qini coefficient well
    above zero and rank customers close to ``true_uplift``.

    The simulated effect is deliberately heterogeneous — strongest for lapsed
    mid-value customers and slightly *negative* for the most frequent buyers,
    who are the classic sleeping dogs — so a model that merely ranks by
    propensity to buy will score poorly.
    """
    rng = np.random.default_rng(seed)
    n = len(features)

    def norm(col: str) -> np.ndarray:
        v = features[col].to_numpy(dtype=float)
        rank = pd.Series(v).rank(pct=True).to_numpy()
        return rank

    recency = norm("recency")
    frequency = norm("frequency")
    monetary = norm("monetary")

    base_rate = 0.05 + 0.35 * frequency + 0.15 * monetary - 0.10 * recency
    base_rate = np.clip(base_rate, 0.01, 0.95)

    # Persuadable: lapsed but with mid-to-high value history.
    true_uplift = 0.30 * recency * (monetary ** 0.5) * (1 - frequency)
    # Sleeping dogs: the most frequent buyers resent the contact.
    true_uplift -= 0.12 * np.clip(frequency - 0.8, 0, None) / 0.2
    true_uplift = np.clip(true_uplift, -0.15, 0.45)

    treatment = (rng.random(n) < treatment_fraction).astype(int)
    prob = np.clip(base_rate + treatment * true_uplift, 0.001, 0.999)
    outcome = (rng.random(n) < prob).astype(int)

    logger.info(
        "Simulated campaign: %d treated / %d control, true ATE=%.4f",
        treatment.sum(), n - treatment.sum(), float(true_uplift.mean()),
    )
    return treatment, outcome, true_uplift


def run_validation(rfm: pd.DataFrame, seed: int = 42) -> dict:
    """Fit and score the T-learner on a simulated campaign.

    This is what the pipeline runs. It proves the estimator works and produces
    the numbers the model card reports, while being explicit that the campaign
    is simulated.
    """
    feature_cols = ["recency", "frequency", "monetary", "tenure", "avg_order_value"]
    X = rfm[feature_cols].copy()
    treatment, outcome, true_uplift = simulate_campaign(rfm, seed=seed)

    X_tr, X_te, t_tr, t_te, y_tr, y_te, _, u_te = train_test_split(
        X, treatment, outcome, true_uplift, test_size=0.3, random_state=seed, stratify=treatment
    )

    model = TLearner(random_state=seed).fit(X_tr, t_tr, y_tr)
    pred = model.predict_uplift(X_te)

    metrics = evaluate(pred, t_te, y_te)
    rank_corr = float(pd.Series(pred).corr(pd.Series(u_te), method="spearman"))
    economics = campaign_economics(pred, t_te, y_te)
    best = economics.loc[economics["net_profit"].idxmax()]

    segments = model.classify(X)
    logger.info("Uplift segments:\n%s", segments.value_counts().to_string())

    return {
        "metrics": metrics.to_dict(),
        "n_test": int(len(X_te)),
        "rank_correlation_with_truth": round(rank_corr, 4),
        "optimal_targeting_fraction": float(best["fraction"]),
        "net_profit_at_optimum": float(best["net_profit"]),
        "net_profit_targeting_all": float(economics.iloc[-1]["net_profit"]),
        "segment_counts": segments.value_counts().to_dict(),
        "economics": economics,
        "model": model,
        "segments": segments,
        "note": "Campaign assignment is simulated; the source data contains no control group.",
    }
