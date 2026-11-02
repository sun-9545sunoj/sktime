# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tabby foundation model forecaster."""

__all__ = ["TabbyForecaster"]
__author__ = ["paris-noah", "sun-9545sunoj"]

import warnings

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster
from sktime.utils.singleton import _multiton

_TABBY_DEFAULT_PATH = "paris-noah/Tabby"


def _resolve_device(device=None):
    """Resolve the torch device to run inference on.

    Parameters
    ----------
    device : str, torch.device, or None
        If not ``None``, returned unchanged. If ``None``, the best available
        device is selected, in the order cuda, mps, cpu.

    Returns
    -------
    device : str or torch.device
        Device to place the model on.
    """
    if device is not None:
        return device

    from sktime.utils.dependencies import _check_soft_dependencies

    if not _check_soft_dependencies("torch", severity="none"):
        return "cpu"

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None:
        if torch.backends.mps.is_available():
            return "mps"
    return "cpu"


@_multiton
class _CachedTabby:
    """Cache for Tabby models, to avoid loading the same model multiple times.

    The Tabby checkpoint is around 590 MB, so repeated loads of the same model
    are avoided by caching instances of this class, keyed by load parameters.

    Parameters
    ----------
    key : str
        Unique identifier of the parameter combination, used as cache key.
    model_path : str or None
        Hugging Face repository identifier or local directory. If ``None``,
        a randomly initialized model is created from ``config``.
    config : dict or None
        Overrides for ``TabbyConfig``, used only if ``model_path`` is ``None``.
    revision : str or None
        Hugging Face revision of the model repository.
    device : str, torch.device, or None
        Device to place the model on.
    dtype : torch.dtype or None
        Data type to cast the model to.
    hf_kwargs : dict or None
        Additional keyword arguments passed to ``from_pretrained``.
    """

    def __init__(
        self,
        key,
        model_path,
        config,
        revision,
        device,
        dtype,
        hf_kwargs,
    ):
        self.key = key
        self.model_path = model_path
        self.config = config
        self.revision = revision
        self.device = device
        self.dtype = dtype
        self.hf_kwargs = hf_kwargs
        self.model_ = None

    def load(self):
        """Load the model if needed, and return the cached instance.

        Returns
        -------
        model : transformers.PreTrainedModel
            Loaded Tabby model, in eval mode, on ``self.device``.
        """
        if self.model_ is not None:
            return self.model_

        if self.model_path is not None:
            model = self._load_from_path()
        else:
            model = self._load_randomly()

        device = _resolve_device(self.device)
        model = model.to(device)
        if self.dtype is not None:
            model = model.to(dtype=self.dtype)

        self.model_ = model.eval()
        return self.model_

    def _load_from_path(self):
        """Load Tabby weights from ``self.model_path``."""
        from transformers import AutoModel

        hf_kwargs = dict(self.hf_kwargs or {})

        return AutoModel.from_pretrained(
            self.model_path,
            revision=self.revision,
            trust_remote_code=True,
            **hf_kwargs,
        )

    def _load_randomly(self):
        """Initialize a Tabby model randomly from ``self.config``.

        Downloads the model code and reference config from
        ``_TABBY_DEFAULT_PATH``, but not the checkpoint weights.
        """
        from transformers import AutoConfig
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        warnings.warn(
            "Initializing Tabby from config creates random weights. Tabby "
            "training is not supported by TabbyForecaster, so these weights "
            "stay random, and are only suitable for tests or local "
            "experimentation.",
            UserWarning,
            stacklevel=2,
        )

        reference = AutoConfig.from_pretrained(
            _TABBY_DEFAULT_PATH,
            revision=self.revision,
            trust_remote_code=True,
        )
        model_cls = get_class_from_dynamic_module(
            reference.auto_map["AutoModel"],
            _TABBY_DEFAULT_PATH,
            revision=self.revision,
        )
        config = type(reference)(**(self.config or {}))

        return model_cls(config)


class TabbyForecaster(BaseForecaster):
    """Zero-shot probabilistic forecaster using the Tabby foundation model.

    Direct interface to the Tabby time series foundation model [1]_, a
    PatchTST style backbone with a learned prompt module, loaded from Hugging
    Face via ``transformers`` with ``trust_remote_code=True``.

    Tabby is univariate and probabilistic. It emits forecasts at 99 quantile
    levels, 0.01 through 0.99, and does not support exogenous variables or
    in-sample predictions. Multivariate and panel data are handled by
    ``sktime`` broadcasting, i.e., each series is forecast separately, which
    matches how the model is intended to be used.

    The model carries out its own normalization, and accepts ``NaN`` values in
    the history, which are treated as unobserved.

    For a requested horizon ``H``, Tabby masks ``max(H, min_forecast_span)``
    future positions, with ``min_forecast_span = 128`` in the released
    checkpoint. The usable history is therefore
    ``context_length - max(H, min_forecast_span)``, capped at
    ``history_length``, with ``context_length = 8192`` and
    ``history_length = 8096`` in the released checkpoint. Longer history is
    truncated to the most recent observations.

    Parameters
    ----------
    model_path : str or None, default="paris-noah/Tabby"
        Local model directory, or Hugging Face model repository identifier.
        If ``None``, a random model is initialized from ``config``, which is
        only useful for testing, since fitting is not supported.
    config : dict, optional
        Overrides for ``TabbyConfig``, used only if ``model_path`` is ``None``.
    device : str or torch.device, optional
        Device used for inference. If ``None``, the best available device is
        selected, in the order cuda, mps, cpu.
    dtype : torch.dtype, optional
        Data type the model is cast to after loading. If ``None``, the dtype
        of the checkpoint is used.
    revision : str, optional
        Hugging Face revision, e.g., a commit hash, of the model repository.
        Pinning a revision is recommended for reproducibility, since the model
        repository ships its own code.
    context_length : int, optional
        Additional user cap on the number of most recent observations passed
        to the model. If ``None``, only the model's own limits apply.
    hf_kwargs : dict, optional
        Additional keyword arguments passed to ``AutoModel.from_pretrained``.

    References
    ----------
    .. [1] https://huggingface.co/paris-noah/Tabby

    Notes
    -----
    The Tabby checkpoint is licensed CC BY-NC 4.0, i.e., for non-commercial
    use only. This applies to the model weights, not to this interface.

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.forecasting.tabby import TabbyForecaster
    >>> config = {
    ...     "context_length": 64,
    ...     "patch_size": 8,
    ...     "num_layers": 1,
    ...     "d_model": 32,
    ...     "head_dim": 8,
    ...     "num_quantiles": 9,
    ...     "prompt_len": 8,
    ...     "ctx_hidden": 8,
    ...     "max_segments": 2,
    ...     "min_seg_len": 4,
    ...     "seg_attn_dim": 8,
    ...     "seg_attn_heads": 2,
    ...     "season_period": 4,
    ...     "history_length": 48,
    ...     "prediction_length": 8,
    ...     "min_forecast_span": 0,
    ...     "dropout": 0.0,
    ... }
    >>> forecaster = TabbyForecaster(
    ...     model_path=None, config=config, device="cpu"
    ... )  # doctest: +SKIP
    >>> y = pd.Series(range(48), dtype="float64")
    >>> _ = forecaster.fit(y, fh=[1, 2, 3, 4])  # doctest: +SKIP
    >>> y_pred = forecaster.predict()  # doctest: +SKIP
    """

    _tags = {
        "authors": ["paris-noah", "sun-9545sunoj"],
        "maintainers": ["sun-9545sunoj"],
        "python_dependencies": [
            "transformers[torch]>=4.40.0,<5.0.0",
            "einops",
        ],
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
        "capability:exogenous": False,
        "capability:multivariate": False,
        "capability:missing_values": True,
        "capability:insample": False,
        "capability:pred_int": True,
        "capability:pred_int:insample": False,
        "requires-fh-in-fit": False,
        # CI and test flags
        # -----------------
        "tests:vm": True,
        "tests:specific": ["sktime.forecasting.tests.test_tabby"],
    }

    def __init__(
        self,
        model_path="paris-noah/Tabby",
        config=None,
        device=None,
        dtype=None,
        revision=None,
        context_length=None,
        hf_kwargs=None,
    ):
        self.model_path = model_path
        self.config = config
        self.device = device
        self.dtype = dtype
        self.revision = revision
        self.context_length = context_length
        self.hf_kwargs = hf_kwargs
        self.model = None
        super().__init__()

    def __getstate__(self):
        """Exclude the loaded model from serialized estimator state."""
        state = self.__dict__.copy()
        state["model"] = None
        return state

    def __setstate__(self, state):
        """Restore serialized estimator state."""
        self.__dict__.update(state)

    def _get_unique_key(self):
        """Return the cache key for the model load parameters."""
        config = self.config or {}
        key_parts = [
            str(self.model_path),
            str(self.revision),
            str(_resolve_device(self.device)),
            str(self.dtype),
            str(sorted(config.items())),
            str(sorted((self.hf_kwargs or {}).items())),
        ]
        return "_".join(key_parts)

    def load(self):
        """Load and cache the underlying Tabby model.

        Returns
        -------
        model : transformers.PreTrainedModel
            Loaded Tabby model, in eval mode.
        """
        if self.model is not None:
            return self.model

        self.model = _CachedTabby(
            key=self._get_unique_key(),
            model_path=self.model_path,
            config=self.config,
            revision=self.revision,
            device=self.device,
            dtype=self.dtype,
            hf_kwargs=self.hf_kwargs,
        ).load()

        return self.model

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : pd.Series
            Time series to which to fit the forecaster.
        X : pd.DataFrame, optional (default=None)
            Ignored, Tabby does not support exogenous data.
        fh : ForecastingHorizon, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.

        Returns
        -------
        self : reference to self
        """
        self.model = self.load()
        self._context = y.copy()
        self._y_name = y.name
        return self

    def _relative_horizon(self, fh):
        """Return the relative forecasting horizon as a positive integer array.

        Raises
        ------
        NotImplementedError
            If the horizon contains in-sample indices.
        """
        relative = fh.to_relative(self.cutoff).to_numpy(dtype=int)
        if len(relative) == 0 or np.any(relative <= 0):
            raise NotImplementedError(
                "TabbyForecaster does not support in-sample predictions."
            )
        return relative

    def _forecast_grid(self, fh):
        """Forecast the native quantile grid for the horizon ``fh``.

        Parameters
        ----------
        fh : ForecastingHorizon
            Out-of-sample forecasting horizon.

        Returns
        -------
        values : 2D np.ndarray, shape (len(fh), n_quantiles)
            Quantile forecasts, rows in the order of ``fh``.
        levels : 1D np.ndarray, shape (n_quantiles,)
            Quantile levels the model emits, strictly increasing.
        """
        model = self.load()
        cfg = model.config

        relative = self._relative_horizon(fh)
        horizon = int(np.max(relative))

        span = max(horizon, int(cfg.min_forecast_span or 0))
        max_span = int(cfg.context_length) - int(cfg.patch_size)
        if span > max_span:
            raise ValueError(
                f"TabbyForecaster was asked to forecast {horizon} steps ahead, "
                f"but the model supports at most {max_span} steps ahead, due "
                f"to its backbone window of {cfg.context_length} positions."
            )

        max_history = min(int(cfg.history_length), int(cfg.context_length) - span)
        if self.context_length is not None:
            max_history = min(max_history, int(self.context_length))

        context = self._context.to_numpy(dtype="float32")[-max_history:]

        output = model.predict(context, prediction_length=horizon)
        output = output.detach().to("cpu").numpy()
        if output.ndim != 3 or output.shape[0] != 1:
            raise ValueError(
                "Tabby forecast output must have shape (1, n_quantiles, horizon)."
            )

        values = output[0].T[relative - 1]
        levels = np.asarray(
            [round(float(q), 6) for q in model.quantile_levels], dtype=float
        )

        if levels.ndim != 1 or len(levels) == 0:
            raise ValueError("Tabby model must define a non-empty quantile sequence.")
        if np.any(np.diff(levels) <= 0):
            raise ValueError("Tabby model quantiles must be strictly increasing.")
        if values.shape[1] != len(levels):
            raise ValueError(
                "Tabby output quantile dimension does not match the model's "
                "quantile levels."
            )

        return values, levels

    def _predict(self, fh, X=None):
        """Forecast time series at future horizon.

        private _predict containing the core logic, called from predict

        Parameters
        ----------
        fh : ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X : pd.DataFrame, optional (default=None)
            Ignored, Tabby does not support exogenous data.

        Returns
        -------
        y_pred : pd.Series
            Point predictions, the median of the quantile forecast.
        """
        values, levels = self._forecast_grid(fh)
        median_index = int(np.argmin(np.abs(levels - 0.5)))

        return pd.Series(
            values[:, median_index],
            index=fh.to_absolute(self.cutoff)._values,
            name=self._y_name,
        )

    def _predict_quantiles(self, fh, X=None, alpha=None):
        """Compute/return quantile forecasts.

        private _predict_quantiles containing the core logic,
        called from predict_quantiles and default _predict_interval

        Parameters
        ----------
        fh : ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X : pd.DataFrame, optional (default=None)
            Ignored, Tabby does not support exogenous data.
        alpha : list of float, optional (default=[0.05, 0.95])
            A list of probabilities at which quantile forecasts are computed.

        Returns
        -------
        quantiles : pd.DataFrame
            Column has multi-index: first level is variable name from ``y`` in
            fit, second level being the values of ``alpha`` passed to the
            function. Row index is ``fh``. Entry at row ``i``, column
            ``(var, p)`` is a quantile forecast at quantile probability ``p``
            for variable ``var``.
        """
        values, levels = self._forecast_grid(fh)
        requested = levels if alpha is None else np.asarray(alpha, dtype=float)

        quantile_values = np.column_stack(
            [
                np.array([np.interp(level, levels, row) for row in values])
                for level in requested
            ]
        )

        name = self._y_name if self._y_name is not None else 0
        columns = pd.MultiIndex.from_product([[name], requested])

        return pd.DataFrame(
            quantile_values,
            index=fh.to_absolute(self.cutoff)._values,
            columns=columns,
        )

    def _predict_proba(self, fh, X=None, marginal=True):
        """Compute/return fully probabilistic forecasts.

        private _predict_proba containing the core logic, called from
        predict_proba

        Parameters
        ----------
        fh : ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X : pd.DataFrame, optional (default=None)
            Ignored, Tabby does not support exogenous data.
        marginal : bool, optional (default=True)
            Ignored, only marginal distributions are returned.

        Returns
        -------
        pred_dist : skpro HistogramQPD distribution
            Predictive distribution, with the same index and columns as the
            return of ``predict``.
        """
        from skpro.distributions import HistogramQPD

        _, levels = self._forecast_grid(fh)
        preds = self._predict_quantiles(fh=fh, X=X, alpha=levels)

        pred_index = preds.index
        name = self._y_name if self._y_name is not None else 0
        columns = pd.Index([name])
        row_index = pd.MultiIndex.from_product([levels, pred_index])
        data = preds.to_numpy().T.reshape(-1, 1)
        q_df = pd.DataFrame(data, index=row_index, columns=columns)

        return HistogramQPD(q_df, tails="mass", index=pred_index, columns=columns)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests.

        Returns
        -------
        params : list of dict
            Parameters to create testing instances of the class.
        """
        params1 = {
            "model_path": None,
            "config": {
                "context_length": 64,
                "patch_size": 8,
                "num_layers": 1,
                "d_model": 32,
                "head_dim": 8,
                "num_quantiles": 9,
                "prompt_len": 8,
                "ctx_hidden": 8,
                "max_segments": 2,
                "min_seg_len": 4,
                "seg_attn_dim": 8,
                "seg_attn_heads": 2,
                "season_period": 4,
                "history_length": 48,
                "prediction_length": 8,
                "min_forecast_span": 0,
                "dropout": 0.0,
            },
            "device": "cpu",
        }
        params2 = {
            "model_path": None,
            "config": {
                "context_length": 48,
                "patch_size": 8,
                "num_layers": 1,
                "d_model": 16,
                "head_dim": 8,
                "num_quantiles": 5,
                "prompt_len": 4,
                "ctx_hidden": 8,
                "max_segments": 2,
                "min_seg_len": 4,
                "seg_attn_dim": 8,
                "seg_attn_heads": 2,
                "season_period": 4,
                "history_length": 32,
                "prediction_length": 8,
                "min_forecast_span": 0,
                "dropout": 0.0,
            },
            "device": "cpu",
            "context_length": 24,
        }

        return [params1, params2]
