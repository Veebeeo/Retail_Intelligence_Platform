"""Segment names and the action recommended for each.

Deliberately dependency-free — pandas and scipy are not imported here, let
alone pandera or scikit-learn. The API only needs to map a label to a
recommendation, and pulling the whole modelling stack in to do a dictionary
lookup forced training-only libraries into the serving image.

Kept next to the labelling logic in ``rfm.assign_labels`` that produces these
names, so the two cannot drift apart.
"""

from __future__ import annotations

STRATEGIES: dict[str, str] = {
    "Champions": "Early access to launches and a loyalty tier. Do not discount — they buy anyway.",
    "New / Promising": "Onboarding sequence and a second-purchase incentive. Highest headroom.",
    "At-Risk High Value": "Win-back with a personalised offer. The largest recoverable revenue.",
    "Hibernating": "Low-cost automated reactivation only. Do not spend margin here.",
}


def strategy_for(label: str) -> str:
    """Recommended action for a segment label.

    Tiered labels (``"Champions (tier 2)"``) fall back to their base strategy.
    """
    return STRATEGIES.get(label.split(" (tier")[0], "Maintain standard outreach.")
