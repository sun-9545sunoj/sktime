# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for the Tabby forecaster.

Most tests here use a small, randomly initialized Tabby model, built from the
``TabbyConfig`` overrides in ``TabbyForecaster.get_test_params``. This only
downloads the model code and reference config of the Hugging Face repository,
around 100 kB, and not the 587 MB checkpoint, so the structural contracts of
the forecaster can be tested cheaply.

The single regression test at the end of this module pins the numerical output
of the pretrained checkpoint, see the comment above it.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.tabby import TabbyForecaster
from sktime.tests.test_switch import run_test_for_class

# the released checkpoint, pinned to an immutable commit
TABBY_REVISION = "1b2381d7f1b0982aa1d5f09ce01b99eb4ee57e01"


def _tiny_forecaster(**kwargs):
    """Return a TabbyForecaster with small, randomly initialized weights."""
    params = TabbyForecaster.get_test_params()[0]
    params.update(kwargs)
    return TabbyForecaster(**params)


def _fit_tiny(y, fh=None, **kwargs):
    """Fit a small, randomly initialized TabbyForecaster, silencing warnings."""
    forecaster = _tiny_forecaster(**kwargs)
    with warnings.catch_warnings():
        # the random weight initialization warns, which is intended
        warnings.simplefilter("ignore", UserWarning)
        forecaster.fit(y, fh=fh)
    return forecaster


def _y(n=48, seed=42):
    """Return a univariate test series."""
    rng = np.random.RandomState(seed)
    values = np.arange(n, dtype="float64") + rng.normal(scale=0.5, size=n)
    return pd.Series(values, name="passengers")


@pytest.mark.skipif(
    not run_test_for_class(TabbyForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_tabby_predict():
    """Test that point forecasts have the expected index, name, and shape."""
    y = _y()
    forecaster = _fit_tiny(y, fh=[1, 2, 3])
    y_pred = forecaster.predict()

    assert isinstance(y_pred, pd.Series)
    assert len(y_pred) == 3
    assert y_pred.name == "passengers"
    assert (y_pred.index == pd.Index([48, 49, 50])).all()
    assert np.isfinite(y_pred.to_numpy()).all()


@pytest.mark.skipif(
    not run_test_for_class(TabbyForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_tabby_predict_quantiles():
    """Test that quantile forecasts are well formed and monotone in alpha."""
    y = _y()
    forecaster = _fit_tiny(y, fh=[1, 2, 3])
    alpha = [0.1, 0.5, 0.9]
    pred_q = forecaster.predict_quantiles(alpha=alpha)

    assert isinstance(pred_q, pd.DataFrame)
    assert pred_q.shape == (3, 3)
    assert isinstance(pred_q.columns, pd.MultiIndex)
    assert list(pred_q.columns) == [("passengers", a) for a in alpha]
    assert (pred_q.index == pd.Index([48, 49, 50])).all()
    # quantile forecasts must be non-decreasing in alpha
    assert (np.diff(pred_q.to_numpy(), axis=1) >= 0).all()
    # the 0.5 quantile is the point forecast
    np.testing.assert_allclose(
        pred_q[("passengers", 0.5)].to_numpy(),
        forecaster.predict().to_numpy(),
        rtol=1e-6,
    )


@pytest.mark.skipif(
    not run_test_for_class(TabbyForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_tabby_predict_quantiles_outside_native_grid():
    """Test that quantiles outside the native level grid saturate at the ends."""
    y = _y()
    forecaster = _fit_tiny(y, fh=[1, 2])
    # the test config emits levels 0.1, ..., 0.9, so 0.001 and 0.999 saturate
    pred_q = forecaster.predict_quantiles(alpha=[0.001, 0.1, 0.9, 0.999])

    np.testing.assert_allclose(
        pred_q[("passengers", 0.001)].to_numpy(),
        pred_q[("passengers", 0.1)].to_numpy(),
    )
    np.testing.assert_allclose(
        pred_q[("passengers", 0.999)].to_numpy(),
        pred_q[("passengers", 0.9)].to_numpy(),
    )


@pytest.mark.skipif(
    not run_test_for_class(TabbyForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_tabby_predict_interval():
    """Test that interval forecasts are derived from the quantile forecasts."""
    y = _y()
    forecaster = _fit_tiny(y, fh=[1, 2, 3])
    pred_int = forecaster.predict_interval(coverage=0.8)

    assert pred_int.shape == (3, 2)
    assert list(pred_int.columns) == [
        ("passengers", 0.8, "lower"),
        ("passengers", 0.8, "upper"),
    ]
    lower = pred_int[("passengers", 0.8, "lower")].to_numpy()
    upper = pred_int[("passengers", 0.8, "upper")].to_numpy()
    assert (lower <= upper).all()


@pytest.mark.skipif(
    not run_test_for_class(TabbyForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_tabby_predict_proba():
    """Test that the distribution forecast matches the predict return."""
    from skpro.distributions import HistogramQPD

    y = _y()
    forecaster = _fit_tiny(y, fh=[1, 2, 3])
    y_pred = forecaster.predict()
    pred_dist = forecaster.predict_proba()

    assert isinstance(pred_dist, HistogramQPD)
    assert (pred_dist.index == y_pred.index).all()
    assert list(pred_dist.columns) == ["passengers"]
    assert pred_dist.mean().shape == (3, 1)


@pytest.mark.skipif(
    not run_test_for_class(TabbyForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_tabby_insample_raises():
    """Test that in-sample forecasting horizons are rejected."""
    y = _y()
    forecaster = _fit_tiny(y)

    with pytest.raises(NotImplementedError):
        forecaster.predict(fh=[-1, 0])


@pytest.mark.skipif(
    not run_test_for_class(TabbyForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_tabby_horizon_too_long_raises():
    """Test that a horizon beyond the backbone window raises a ValueError."""
    y = _y()
    forecaster = _fit_tiny(y)
    model = forecaster.load()
    max_span = int(model.config.context_length) - int(model.config.patch_size)

    with pytest.raises(ValueError, match="at most"):
        forecaster.predict(fh=list(range(1, max_span + 2)))


@pytest.mark.skipif(
    not run_test_for_class(TabbyForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_tabby_missing_values():
    """Test that NaN values in the history are accepted by the model."""
    y = _y()
    y.iloc[3:7] = np.nan
    forecaster = _fit_tiny(y, fh=[1, 2])
    y_pred = forecaster.predict()

    assert np.isfinite(y_pred.to_numpy()).all()


@pytest.mark.skipif(
    not run_test_for_class(TabbyForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_tabby_panel_broadcast():
    """Test that panel input is broadcast, one forecast per instance."""
    from sktime.utils._testing.hierarchical import _make_hierarchical

    y = _make_hierarchical(
        hierarchy_levels=(2,), min_timepoints=48, max_timepoints=48, random_state=0
    )
    forecaster = _fit_tiny(y, fh=[1, 2])
    y_pred = forecaster.predict()

    assert isinstance(y_pred, pd.DataFrame)
    assert len(y_pred) == 4
    assert y_pred.index.nlevels == y.index.nlevels
    assert np.isfinite(y_pred.to_numpy()).all()


@pytest.mark.skipif(
    not run_test_for_class(TabbyForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_tabby_context_truncation():
    """Test that only the most recent ``context_length`` values are used."""
    y = _y(n=400)
    forecaster = _fit_tiny(y, fh=[1, 2], context_length=24)
    y_pred_long = forecaster.predict()

    # fitting on only the last 24 values must give the identical forecast
    forecaster_short = _fit_tiny(y.iloc[-24:], context_length=24)
    y_pred_short = forecaster_short.predict(fh=[1, 2])

    np.testing.assert_allclose(
        y_pred_long.to_numpy(), y_pred_short.to_numpy(), rtol=1e-6
    )


# Regression test against the pretrained checkpoint
# -------------------------------------------------
# The reference values below were generated on CPU from the Hugging Face
# repository paris-noah/Tabby at the immutable commit
# 1b2381d7f1b0982aa1d5f09ce01b99eb4ee57e01, through TabbyForecaster. The model
# ships its own code in that repository, loaded with trust_remote_code=True;
# sktime does not vendor any Tabby code.
#
# Tabby is deterministic in eval mode, which TabbyForecaster always uses, so no
# seeding is needed.
#
# Environment: Python 3.12.16, transformers 4.57.6, torch 2.15.0, numpy 2.3.4.
#
# The reference values were produced by the following script:
#
#     from sktime.datasets import load_airline
#     from sktime.forecasting.tabby import TabbyForecaster
#
#     y = load_airline()
#     f = TabbyForecaster(
#         device="cpu", revision="1b2381d7f1b0982aa1d5f09ce01b99eb4ee57e01"
#     )
#     f.fit(y, fh=[1, 2, 3])
#     print(repr(f.predict().to_numpy()))
#     print(repr(f.predict_quantiles(alpha=[0.1, 0.5, 0.9]).to_numpy()))
#
TABBY_AIRLINE_MEDIAN = np.array([448.55988, 442.19025, 494.63425])
TABBY_AIRLINE_QUANTILES = np.array(
    [
        [429.79974, 448.55988, 468.48462],
        [417.43213, 442.19025, 465.74478],
        [463.70422, 494.63425, 524.93298],
    ]
)


@pytest.mark.skipif(
    not run_test_for_class(TabbyForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_tabby_pretrained_regression():
    """Test the pretrained checkpoint against reference outputs."""
    from sktime.datasets import load_airline

    y = load_airline()
    forecaster = TabbyForecaster(device="cpu", revision=TABBY_REVISION)
    forecaster.fit(y, fh=[1, 2, 3])

    np.testing.assert_allclose(
        forecaster.predict().to_numpy(),
        TABBY_AIRLINE_MEDIAN,
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        forecaster.predict_quantiles(alpha=[0.1, 0.5, 0.9]).to_numpy(),
        TABBY_AIRLINE_QUANTILES,
        rtol=1e-4,
    )
