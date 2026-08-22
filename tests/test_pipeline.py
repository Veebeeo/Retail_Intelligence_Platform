"""Training, serving and end-to-end orchestration."""

from __future__ import annotations

import pytest

from retail_intel.forecasting import serving
from retail_intel.forecasting.train import BASELINE
from retail_intel.forecasting.train import run as train_run


@pytest.fixture(scope="module")
def trained(seeded_db, weekly_panel):
    serving.reset_cache()
    out = train_run(panel=weekly_panel, horizon=4, n_folds=1, log_to_mlflow=False)
    serving.reset_cache()
    return out


def test_training_produces_a_champion_for_every_sku(trained, weekly_panel):
    assert trained["n_models"] == weekly_panel["stock_code"].nunique()


def test_no_champion_is_worse_than_the_baseline(trained):
    """A model that loses to a one-line heuristic must not be promoted."""
    champs = trained["champions"]
    promoted = champs[champs["champion"] != BASELINE]
    assert (promoted["champion_mase"] < promoted["baseline_mase"]).all()


def test_the_bundle_is_written_and_loadable(trained):
    bundle = serving.load_bundle()
    assert len(bundle.models) == trained["n_models"]
    assert bundle.version >= 1
    assert set(bundle.meta) == set(bundle.models)


def test_every_bundle_entry_carries_its_provenance(trained):
    bundle = serving.load_bundle()
    for sku, meta in bundle.meta.items():
        assert meta["model"], f"{sku} has no model name"
        assert meta["residual_std"] >= 0
        assert meta["n_weeks"] > 0


def test_serving_a_forecast_matches_the_api_shape(trained):
    bundle = serving.load_bundle()
    result = serving.forecast(bundle.skus[0], 4, bundle)
    assert len(result["predictions"]) == 4
    for point in result["predictions"]:
        assert point["lower_95"] <= point["predicted_quantity"] <= point["upper_95"]


def test_forecast_weeks_run_forwards_from_the_last_observed_week(trained):
    bundle = serving.load_bundle()
    weeks = [p["week_starting"] for p in serving.forecast(bundle.skus[0], 4, bundle)["predictions"]]
    assert weeks == sorted(weeks)
    assert len(set(weeks)) == 4


def test_an_unknown_sku_raises(trained):
    with pytest.raises(KeyError):
        serving.forecast("NOT_A_SKU", 4, serving.load_bundle())


def test_bundle_summary_reports_the_model_mix(trained):
    summary = serving.bundle_summary(serving.load_bundle())
    assert summary["n_skus"] > 0
    assert sum(summary["model_mix"].values()) == summary["n_skus"]
    assert 0 <= summary["pct_skus_beating_baseline"] <= 100


def test_missing_bundle_raises_a_clear_error(tmp_path):
    serving.reset_cache()
    with pytest.raises(serving.ModelNotAvailable, match="No trained models"):
        serving.load_bundle(tmp_path / "absent.pkl")
    serving.reset_cache()


def test_market_basket_recovers_planted_associations(seeded_db):
    """The generator plants complementary SKU pairs; the miner should find some."""
    from retail_intel.recommend.market_basket import build_matrix, load_baskets, mine_rules

    baskets = load_baskets()
    rules = mine_rules(build_matrix(baskets, top_n_items=50), min_support=0.01, min_lift=1.1)
    assert not rules.empty, "no association rules recovered from planted complements"
    assert (rules["lift"] > 1).all()
    assert rules["confidence"].between(0, 1).all()


def test_full_pipeline_runs_end_to_end(seeded_db):
    """The whole chain, in order, on a clone with no raw data."""
    from pipelines.flow import run_pipeline

    results = run_pipeline(
        synthetic=False,
        skip=("train",),  # covered above; slow to repeat here
        features={"top_n": 10},
        segment={"clv_days": 90},
        baskets={"min_support": 0.01, "min_lift": 1.1},
        drift={"weeks": 6},
    )
    failed = [r.name for r in results if r.status != "success"]
    assert not failed, f"pipeline steps failed: {failed}"


def test_missing_model_library_gives_an_actionable_error(tmp_path, monkeypatch):
    """A bundle is a pickle of fitted models, so loading it needs whatever
    library produced the champion. A bare ModuleNotFoundError from inside
    pickle does not tell an operator which package to install."""
    import pickle

    bundle_path = tmp_path / "champions.pkl"

    class _Unloadable:
        def __reduce__(self):
            return (_missing_factory, ())

    bundle_path.write_bytes(
        pickle.dumps({"models": {"X": _Unloadable()}, "meta": {}, "version": 1})
    )

    real_loads = pickle.load

    def fake_load(fh):
        raise ModuleNotFoundError("No module named 'xgboost'", name="xgboost")

    monkeypatch.setattr(pickle, "load", fake_load)
    serving.reset_cache()
    with pytest.raises(serving.ModelNotAvailable, match="xgboost"):
        serving.load_bundle(bundle_path)
    monkeypatch.setattr(pickle, "load", real_loads)
    serving.reset_cache()


def _missing_factory():  # pragma: no cover - only referenced inside a pickle
    raise ModuleNotFoundError("No module named 'xgboost'", name="xgboost")
