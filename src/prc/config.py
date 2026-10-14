"""Experiment configuration: one YAML file per experiment under ``experiments/``.

An experiment is fully described by (features, cv scheme, model, target transform,
seeds). Everything else is derived. Configs are validated with pydantic so a typo
fails before any compute is spent — see :class:`StrictModel`, which is what makes that
sentence true (it was not, before 2026-09-19).

The mixture keys (``target.mode`` / ``target.split_groups`` / ``mixture``) are optional and
default to the plain single-model path, so every YAML written before them keeps its exact
behaviour *and* its config digest (see :meth:`ExperimentConfig.digest`).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

CorruptionType = Literal["sched_default", "rollover", "stamp_early"]
Regime = Literal["A", "B", "C"]


class StrictModel(BaseModel):
    """Base for every config model: an unknown key is an ERROR, not a shrug.

    Pydantic v2 defaults to ``extra="ignore"``, and until 2026-09-19 every class here
    took that default while this module's docstring claimed the opposite. The code review
    of that date reproduced the consequence: ``model.paramz``, ``target.clipmin``,
    ``target_encodings``, ``seedz``, ``mixture.corruption.normalweight`` and
    ``cv.schemee`` all load silently, and the run is then its own *parent* — with the
    parent's digest, so nothing downstream can tell. An audit of exp0023-exp0101 found no
    experiment that actually lost a knob this way, but that is a result, not a guarantee.

    Nothing in the repo relies on extra keys being accepted.
    """

    model_config = ConfigDict(extra="forbid")


#: Log-link objectives (P3a, 2026-09-23): the booster fits log E[y|x] under a Bregman loss
#: (Poisson / Tweedie / Gamma deviance), so it targets the conditional mean on the label's
#: own scale and predicts on that scale. ``ModelConfig.log_link`` reads these.
XGB_LOG_LINK = frozenset({"count:poisson", "reg:tweedie", "reg:gamma"})
CAT_LOG_LINK = frozenset({"Poisson", "Tweedie"})

#: The ``model.params`` keys each ``model.kind`` accepts (TODO item 12, 2026-09-21).
#:
#: ``StrictModel`` forbids unknown *config* keys, but ``model.params`` is a free dict handed to
#: the library, so ``min_child_wieght: 3000`` loaded silently and XGBoost ignored it with a
#: warning nobody reads. The lists are every key a registered YAML uses, the wrapper's own
#: ``DEFAULTS`` (``prc.models.<kind>``), the hyper-parameter search spaces and the common
#: documented knobs (with their library aliases). Keys the wrapper sets itself and would
#: silently override (the seed, ``device`` for xgb, CatBoost's ``iterations`` / ``od_*`` /
#: ``task_type``) are deliberately absent. To use a documented parameter that is missing
#: here, add it to the list; validation never changes a digest (it only rejects).
MODEL_PARAMS: dict[str, frozenset[str]] = {
    "xgb": frozenset(
        {
            "objective",
            "eval_metric",
            "learning_rate",
            "eta",
            "max_depth",
            "max_leaves",
            "min_child_weight",
            "subsample",
            "sampling_method",
            "colsample_bytree",
            "colsample_bylevel",
            "colsample_bynode",
            "reg_lambda",
            "lambda",
            "reg_alpha",
            "alpha",
            "gamma",
            "min_split_loss",
            "max_delta_step",
            "grow_policy",
            "tree_method",
            "max_bin",
            "max_cat_to_onehot",
            "max_cat_threshold",
            "huber_slope",
            "quantile_alpha",
            "tweedie_variance_power",
            "base_score",
            "scale_pos_weight",
            "monotone_constraints",
            "interaction_constraints",
            "num_parallel_tree",
            "nthread",
            "verbosity",
            "disable_default_eval_metric",
            "multi_strategy",
        }
    ),
    "lgbm": frozenset(
        {
            "objective",
            "metric",
            "learning_rate",
            "num_leaves",
            "max_depth",
            "min_data_in_leaf",
            "min_child_samples",
            "min_sum_hessian_in_leaf",
            "min_child_weight",
            "feature_fraction",
            "colsample_bytree",
            "feature_fraction_bynode",
            "bagging_fraction",
            "subsample",
            "bagging_freq",
            "subsample_freq",
            "lambda_l1",
            "reg_alpha",
            "lambda_l2",
            "reg_lambda",
            "min_gain_to_split",
            "min_split_gain",
            "max_bin",
            "extra_trees",
            "path_smooth",
            "cat_smooth",
            "cat_l2",
            "max_cat_to_onehot",
            "min_data_per_group",
            "max_cat_threshold",
            "huber_delta",
            "alpha",
            "linear_tree",
            "boost_from_average",
            "monotone_constraints",
            "verbose",
            "verbosity",
            "deterministic",
            "force_row_wise",
            "force_col_wise",
            "num_threads",
            "device_type",
            "gpu_device_id",
        }
    ),
    "cat": frozenset(
        {
            "loss_function",
            "eval_metric",
            "learning_rate",
            "depth",
            "max_depth",
            "grow_policy",
            "min_data_in_leaf",
            "min_child_samples",
            "max_leaves",
            "l2_leaf_reg",
            "reg_lambda",
            "border_count",
            "max_bin",
            "random_strength",
            "bootstrap_type",
            "subsample",
            "bagging_temperature",
            "rsm",
            "colsample_bylevel",
            "one_hot_max_size",
            "max_ctr_complexity",
            "leaf_estimation_iterations",
            "leaf_estimation_method",
            "boosting_type",
            "score_function",
            "model_size_reg",
            "boost_from_average",
            "thread_count",
            "verbose",
            "logging_level",
            "metric_period",
            "used_ram_limit",
            "gpu_ram_part",
            "devices",
        }
    ),
    # prc.models.tabm.DEFAULTS (2026-09-22): the network's own knobs, all of them.
    "tabm": frozenset(
        {
            "k",
            "n_blocks",
            "d_block",
            "dropout",
            "learning_rate",
            "weight_decay",
            "batch_size",
            "eval_batch_size",
            "max_epochs",
            "d_num_emb",
            "n_frequencies",
            "frequency_scale",
            "d_cat_emb",
            "cat_max_levels",
            "cat_min_count",
            "n_quantiles",
            "qt_subsample",
            "amp",
            "verbose",
            "lr_scale",
            "lr_base_batch",
            "warmup_epochs",
            "arch",
            "num_emb",
            "n_bins",
            "slog_patterns",
            "slog_qt",
        }
    ),
    # prc.models.setenc.DEFAULTS (2026-09-22): the set-attention member's knobs, all of them.
    # The four `ctx_*` keys are the context-set parameters (prc.features.context_sets).
    "setenc": frozenset(
        {
            "d_row",
            "d_num_emb",
            "n_frequencies",
            "frequency_scale",
            "d_cat_emb",
            "cat_max_levels",
            "cat_min_count",
            "n_quantiles",
            "qt_subsample",
            "ctx_k",
            "ctx_mvt_before_min",
            "ctx_mvt_after_min",
            "ctx_aobt_before_min",
            "d_set",
            "n_attn",
            "n_heads",
            "d_ctx_cat",
            "k",
            "n_head_blocks",
            "d_head",
            "dropout",
            "learning_rate",
            "weight_decay",
            "batch_size",
            "eval_batch_size",
            "max_epochs",
            "amp",
            "verbose",
            "slog_patterns",
            "slog_qt",
        }
    ),
    # prc.models.realmlp.DEFAULTS (2026-09-22): the RealMLP-TD member's knobs, all of them.
    "realmlp": frozenset(
        {
            "n_hidden_layers",
            "d_hidden",
            "act",
            "use_parametric_act",
            "p_drop",
            "p_drop_sched",
            "learning_rate",
            "lr_sched",
            "weight_decay",
            "wd_sched",
            "beta1",
            "beta2",
            "bias_lr_factor",
            "act_lr_factor",
            "scale_lr_factor",
            "plr_lr_factor",
            "emb_lr_factor",
            "plr_sigma",
            "plr_hidden_1",
            "plr_hidden_2",
            "max_one_hot_cat_size",
            "embedding_size",
            "cat_max_levels",
            "cat_min_count",
            "smooth_clip",
            "stat_subsample",
            "init_rows",
            "clamp_output",
            "batch_size",
            "eval_batch_size",
            "max_epochs",
            "amp",
            "verbose",
            "slog_patterns",
            "nm_dropout",
            "nm_dropout_cols",
            "nm_key_col",
        }
    ),
    # prc.models.modernnca.DEFAULTS (2026-09-23): the retrieval member's knobs, all of them.
    "modernnca": frozenset(
        {
            "d_num_emb",
            "n_frequencies",
            "frequency_scale",
            "d_cat_emb",
            "cat_max_levels",
            "cat_min_count",
            "n_quantiles",
            "qt_subsample",
            "slog_patterns",
            "slog_qt",
            "n_blocks",
            "d_block",
            "d_emb",
            "dropout",
            "temperature",
            "n_cand",
            "cand_grad",
            "cand_chunk",
            "exclude_same_col",
            "top_k",
            "bank_chunk",
            "residual_head",
            "learning_rate",
            "weight_decay",
            "batch_size",
            "eval_batch_size",
            "max_epochs",
            "amp",
            "verbose",
        }
    ),
    # prc.models.seqenc.DEFAULTS (2026-09-23): the sequence member = setenc's knobs without
    # `n_attn`, plus the GRU and relative-time encoding.
    "seqenc": frozenset(
        {
            "d_row",
            "d_num_emb",
            "n_frequencies",
            "frequency_scale",
            "d_cat_emb",
            "cat_max_levels",
            "cat_min_count",
            "n_quantiles",
            "qt_subsample",
            "ctx_k",
            "ctx_mvt_before_min",
            "ctx_mvt_after_min",
            "ctx_aobt_before_min",
            "d_set",
            "n_heads",
            "d_ctx_cat",
            "k",
            "n_head_blocks",
            "d_head",
            "dropout",
            "learning_rate",
            "weight_decay",
            "batch_size",
            "eval_batch_size",
            "max_epochs",
            "amp",
            "verbose",
            "slog_patterns",
            "slog_qt",
            "n_gru",
            "d_gru",
            "n_time_freq",
        }
    ),
    "proxy": frozenset({"column", "lo", "hi"}),
    "mean": frozenset(),
    "median": frozenset(),
}


class ModelConfig(StrictModel):
    # `tabm` (2026-09-22): the PyTorch TabM regressor, prc.models.tabm (needs `--group nn`).
    # `setenc` (2026-09-22): the set-attention member, prc.models.setenc (needs `--group nn`
    # and the `context_sets` feature block).
    # `realmlp` (2026-09-22): RealMLP-TD (Holzmueller et al. 2024), prc.models.realmlp (`--group nn`).
    # `modernnca` (2026-09-23): the ModernNCA retrieval member, prc.models.modernnca (`--group nn`).
    # `seqenc` (2026-09-23): the GRU sequence member, prc.models.seqenc (`--group nn` and the
    # `context_sets` feature block).
    kind: Literal[
        "lgbm",
        "xgb",
        "cat",
        "tabm",
        "setenc",
        "realmlp",
        "modernnca",
        "seqenc",
        "mean",
        "median",
        "proxy",
    ] = "lgbm"
    params: dict[str, Any] = Field(default_factory=dict)
    num_boost_round: int = 5000
    early_stopping_rounds: int = 200
    device: Literal["cpu", "cuda"] = "cpu"
    # The space early stopping measures RMSE in (2026-09-21, task T6). Default "transformed" =
    # the pre-2026-09-21 behaviour: the booster's built-in RMSE on the transformed labels
    # (sqrt(y) for a sqrt member). "raw" squares prediction and label before the RMSE, so a
    # sqrt member stops where its *seconds* RMSE on the validation fold is lowest. Only for
    # `target.transform: sqrt` and `kind: xgb | cat` (checked by ExperimentConfig). Dropped
    # from the digest while it holds its default, so every registered run keeps its identity.
    # P3a (2026-09-23): a log-link objective (`log_link()`, labels in seconds) is also allowed.
    # There "transformed" is the loss's own space — early stopping on its deviance (xgb:
    # `poisson-nloglik` / `tweedie-nloglik@p` unless params name a metric; cat: the loss) —
    # and "raw" is the seconds RMSE of exp(score).
    eval_metric_space: Literal["transformed", "raw"] = "transformed"
    # P3a (2026-09-23): a lower bound on the booster's *training* labels (after the target
    # transform), applied inside the xgb / cat wrapper. Poisson / Tweedie / Gamma losses need
    # y >= 0 (Gamma: y > 0) and the raw label goes negative on 369 rows (min -12 s). The
    # early-stopping labels and every metric keep the real labels. None = the pre-2026-09-23
    # behaviour; dropped from the digest while it holds that default.
    label_clip_min: float | None = None

    def log_link(self) -> bool:
        """True for the mean-targeting log-link losses:
        the booster's raw score is log E[y|x] and its prediction is on the
        label's own scale."""
        if self.kind == "xgb":
            return str(self.params.get("objective", "")) in XGB_LOG_LINK
        if self.kind == "cat":
            return str(self.params.get("loss_function", "")).split(":")[0] in CAT_LOG_LINK
        return False

    @model_validator(mode="after")
    def _check_log_link(self) -> ModelConfig:
        if self.label_clip_min is not None and self.kind not in ("xgb", "cat"):
            raise ValueError(
                f"model.label_clip_min is implemented for xgb / cat, not {self.kind!r}"
            )
        loss = str(self.params.get("loss_function", ""))
        if self.kind == "cat" and self.device == "cuda" and loss.startswith("Tweedie"):
            # measured 2026-09-23 on CatBoost 1.2.10 (RTX 3070): every GPU Tweedie fit keeps
            # raw score 0 (prediction 1.0 s) whatever the variance power, leaf method or
            # learning rate; CPU Tweedie and GPU Poisson learn. Use xgb `reg:tweedie`.
            raise ValueError(
                "CatBoost Tweedie does not learn on the GPU (1.2.10: raw score stays 0); use "
                "kind: xgb with objective reg:tweedie, or loss_function: Poisson"
            )
        return self

    @model_validator(mode="after")
    def _check_params(self) -> ModelConfig:
        allowed = MODEL_PARAMS[self.kind]
        unknown = sorted(set(self.params) - allowed)
        if unknown:
            import difflib

            hints = []
            for k in unknown:
                close = difflib.get_close_matches(k, sorted(allowed), n=1)
                hints.append(f"{k!r}" + (f" (did you mean {close[0]!r}?)" if close else ""))
            raise ValueError(
                f"model.params: unknown key(s) for kind {self.kind!r}: {', '.join(hints)}. "
                "Known keys are listed in prc.config.MODEL_PARAMS; add a documented library "
                "parameter there if it is missing."
            )
        return self


class TargetConfig(StrictModel):
    transform: Literal["none", "log1p", "sqrt"] = "none"
    clip_min: float | None = None  # clip predictions (seconds) after inverse transform
    clip_max: float | None = None
    # "raw"                  : predict the target as received (the default; M0-M1 behaviour)
    # "residual_proxy"       : one model, regime as a feature, regime-A rows learn
    #                          y - (MVT - AOBT_3) and the proxy is added back at prediction
    # "residual_proxy_split" : the same offset, but one model per group of `split_groups`
    mode: Literal["raw", "residual_proxy", "residual_proxy_split"] = "raw"
    # only for mode=residual_proxy_split; must partition {A, B, C}
    split_groups: list[list[Regime]] | None = None

    @model_validator(mode="after")
    def _check(self) -> TargetConfig:
        if self.mode != "raw" and self.transform != "none":
            raise ValueError("target.mode != raw requires transform: none (residuals go negative)")
        if self.mode == "residual_proxy_split":
            groups = self.split_groups or [["A"], ["B", "C"]]
            flat = [r for g in groups for r in g]
            if sorted(flat) != ["A", "B", "C"]:
                raise ValueError(f"target.split_groups must partition A/B/C, got {groups}")
            self.split_groups = groups
        elif self.split_groups is not None and self.mode != "residual_proxy_split":
            raise ValueError(
                "target.split_groups is only meaningful with mode=residual_proxy_split"
            )
        return self


class CorruptionConfig(StrictModel):
    """The fold-safe corruption head (plan M3 / A2).

    Labels are derived from the target, so a classifier is fitted inside every CV fold on that
    fold's training rows only; the test-time probabilities come from a head fitted on all
    training rows. See ``prc.mixture.corruption`` for the label and corrupt-value definitions.
    """

    types: list[CorruptionType] = Field(
        default_factory=lambda: ["sched_default", "rollover", "stamp_early"]
    )
    # "all"   : the normal regressor is trained on every row (safe fallback)
    # "clean" : ... on rows with no corruption label (the theoretically right expert)
    train_normal_on: Literal["all", "clean"] = "all"
    # --- options added 2026-09-18; every one of them defaults to the pre-existing behaviour
    # and is dropped from the config digest while it holds its default (see
    # ``ExperimentConfig.digest``), so runs registered before them keep their identity. ---
    #
    # Feature blocks the *classifier* sees in addition to the regressor's matrix, joined
    # unmasked (prefix ``clfx_``) even when ``mask_for_no_nm`` nulls them for the regressor.
    extra_blocks: list[str] = Field(default_factory=list)
    # Engineered head-only signals (local hour, MVT-SCHED, airport-day no-NM share, stand
    # turnaround gaps, fold-safe 24 h BLOCK==SCHED count). See ``prc.mixture.clf_features``.
    extra_signals: bool = False
    # Head-only *identity* signals (2026-09-18): the
    # FLIGHT_mvt parse (airline prefix, doubled-designator flag, shape), the stand
    # re-occupation bit, the domestic-destination bit, and two fold-scoped encodings of the
    # schedule-default label per (airport, airline) and (airport, ADES). Classifier only.
    identity_signals: bool = False
    # Restrict the regressor columns the classifier sees (memory / noise). [] = all of them.
    clf_columns: list[str] = Field(default_factory=list)
    # --- the schedule-default head's target (2026-09-18) ---
    # The mixture's error on a row is ``(C - yhat)·(p - z)`` with ``z = (y - yhat)/(C - yhat)``,
    # so under RMSE the optimal ``p`` is the conditional mean of the *continuous* ``z`` under
    # weights ``(C - yhat)²`` — not a calibrated probability of the 0/1 label (``z ~ 1`` on a
    # schedule default, ``~ 1.26`` on a day rollover, ``~ -1.4`` on a clean short taxi). A
    # classifier cannot represent either tail and AUC cannot see the weights, which is why
    # ``exp0057`` raised the no-NM LIRF AUC 0.821 -> 0.858 and lost 10 s (EDA §1.6, §5).
    # "classifier"   : the 0/1 schedule-default label (the pre-2026-09-18 head; the default)
    # "z_regression" : a weighted squared-error regression of ``z``, clipped to ``p_clip``
    # Only ``sched_default`` is affected; ``rollover`` / ``stamp_early`` stay classifiers, and
    # the ``normal_weight`` expert weights keep coming from classifiers in every configuration
    # (``1 - sum_k p_k`` is a probability; the z head's output is a regression coefficient).
    sched_head: Literal["classifier", "z_regression"] = "classifier"
    z_max_depth: int = 6
    z_learning_rate: float = 0.05
    z_num_boost_round: int = 500
    # The weights are normalised to mean 1 over the rows that carry any, so this keeps its
    # usual "effective number of rows in a leaf" meaning (raw ``(C - yhat)²`` is ~1e7 s²).
    z_min_child_weight: float = 50.0
    z_min_gap_s: float = 60.0  # |C - yhat| below this gets weight 0: z is noise there
    z_clip_lo: float = -2.0  # z is winsorised to [lo, hi] for TRAINING only, so that a few
    z_clip_hi: float = 2.0  # extreme ratios cannot dominate the fit
    p_clip: list[float] = Field(default_factory=lambda: [0.0, 1.0])  # z head only, at predict
    # Per-(type, airport) isotonic calibration of p, fitted on the head's own internal
    # holdout (never on the CV validation rows).
    calibrate: Literal["none", "isotonic"] = "none"
    calibrate_min_positives: int = 200  # below this an airport falls back to the pooled curve
    # How the normal regressor is trained against the corrupted rows:
    # "none"         : the hard `train_normal_on` switch (all rows, or the clean ones)
    # "clean"        : weight 1 on rows with no corruption label, 0 on the rest
    # "one_minus_p"  : weight 1 - sum_k p_k from probabilities fitted *inside* the fold's own
    #                  training rows (nested split), so no validation label reaches the weights
    normal_weight: Literal["none", "clean", "one_minus_p"] = "none"
    weight_inner_splits: int = 2  # nested splits used to get in-fold out-of-sample p
    weight_floor: float = 0.0  # the weight is clipped to [floor, 1]
    # 2026-09-22. The day-rollover rows
    # (``y ~ 86400 + a normal taxi``, the head's own ``rollover`` label) get p_rollover ~ 7e-6
    # from the head, so under ``one_minus_p`` they train the normal expert at ~full weight
    # with a target ~11x a typical one in sqrt space.
    # "soft" : every row keeps the ``normal_weight`` weight (the pre-2026-09-22 behaviour)
    # "zero" : training rows whose *training label* is ``rollover`` get weight 0 in the
    #          normal regressor's fit (per fold and under ``test_fit: full``); every other
    #          row keeps its soft weight. The head, its inputs and validation/test scoring
    #          are unchanged: rollover rows stay in validation.
    rollover_weight: Literal["soft", "zero"] = "soft"
    # --- 2026-09-19, from the pipeline code review -----------------------------------------
    # F7.1. The head's RNG seed. ``None`` = ``cfg.seeds[0]``, the pre-2026-09-19 behaviour.
    #
    # Why it needs to be settable: a seed-farm job array runs ONE seed per cell, so each cell had
    # its own head, and ``merge_seeds.py merge-seeds`` then averaged three different mixtures.
    # A merged run was a 3-head ensemble while a native multi-seed run is a 1-head model —
    # measured on the exp0091 cells, mean ``p_stamp_early`` 0.00139 / 0.00332 / 0.00149 — so
    # R5's gate number (native) and its month_loo number (merged) are not the same model, and
    # ``head_delta_rmse`` is about -69 s, so that is not a rounding difference. Pin this in a
    # farm's YAML and every cell fits the identical head.
    head_seed: int | None = None
    # F4. Leave-one-out schedule-default label encodings for the rows the head is FITTED on.
    #
    # ``clf_features.sched_default_encodings`` is a smoothed group mean with no LOO, and the
    # encoding for a fitted row is computed from a set that contains that row — self-share up
    # to 4.8 % in cells with n <= 5, ``corr(encoding, own label) ~ 0.27`` — while the scored
    # rows get clean out-of-sample encodings. That is train/serve skew, it exists only under
    # ``identity_signals: true``, and ``identity_signals`` is exactly what exp0057 / exp0058 /
    # exp0062 turned on and what was rejected for losing 8-31 s. Default False keeps every
    # registered result; True subtracts the row's own label before smoothing.
    sched_default_te_loo: bool = False
    # F5. Nested splits used to build the z head's ``yhat_normal``. ``None`` = reuse
    # ``weight_inner_splits`` (2), the pre-2026-09-19 behaviour.
    #
    # ``z = (y - yhat)/(C - yhat)`` is built from a regressor fitted on ``1 - 1/splits`` of the
    # fold and then applied to the prediction of a FULL-size model, so at 2 splits the head is
    # calibrated against a half-size yhat. Raising this makes each nested fit approach full
    # size (10 splits => 90 % of the fold) at the cost of that many regressor fits per fold —
    # the honest way to give the z head a full-size target. Plausible cause of the ratio-head
    # rejection (exp0061 +16.8, exp0062 +31.5).
    z_inner_splits: int | None = None
    max_depth: int = 6
    learning_rate: float = 0.05
    num_boost_round: int = 500
    early_stopping_rounds: int = 50
    params: dict[str, Any] = Field(default_factory=dict)
    holdout_frac: float = 0.1  # internal, stratified, for AUC early stopping
    min_positives: int = 20  # below this a type falls back to its base rate
    p_max: float = 1.0  # the probabilities are scaled down where they sum above this
    sched_default_tol_s: float = 10.0  # |BLOCK - SCHED| <= tol  =>  schedule default
    rollover_s: float = 86400.0
    rollover_window_s: float = 3600.0  # |y - 86400| < window  =>  rollover
    stamp_early_min_s: float = 3600.0  # y > this, not the two above, NM matched
    stamp_early_offset_s: float = 3055.0  # median airport-vs-NM gap of the stamp-early rows, 2025

    @model_validator(mode="after")
    def _check_corruption(self) -> CorruptionConfig:
        if self.z_inner_splits is not None and self.z_inner_splits < 2:
            raise ValueError("z_inner_splits must be >= 2 (it is a nested split) or null")
        if self.sched_default_te_loo and not self.identity_signals:
            raise ValueError(
                "sched_default_te_loo only affects the identity-signal encodings; "
                "set identity_signals: true as well, or drop it"
            )
        if self.normal_weight != "none" and self.train_normal_on != "all":
            raise ValueError(
                "mixture.corruption.normal_weight replaces the hard train_normal_on switch; "
                "set train_normal_on: all when weighting"
            )
        if self.rollover_weight == "zero":
            if self.normal_weight == "none":
                raise ValueError(
                    "rollover_weight: zero acts on the normal expert's sample weights; "
                    "set normal_weight (one_minus_p or clean) as well"
                )
            if "rollover" not in self.types:
                raise ValueError("rollover_weight: zero needs rollover in mixture types")
        if self.weight_inner_splits < 2:
            raise ValueError("weight_inner_splits must be >= 2 (it is a nested split)")
        if not 0.0 <= self.weight_floor <= 1.0:
            raise ValueError("weight_floor must be in [0, 1]")
        if len(self.p_clip) != 2 or self.p_clip[0] > self.p_clip[1]:
            raise ValueError(f"p_clip must be [lo, hi] with lo <= hi, got {self.p_clip}")
        if self.z_clip_lo > self.z_clip_hi:
            raise ValueError("z_clip_lo must be <= z_clip_hi")
        if self.z_min_gap_s < 0:
            raise ValueError("z_min_gap_s must be >= 0")
        if self.sched_head == "z_regression" and "sched_default" not in self.types:
            raise ValueError("sched_head: z_regression needs sched_default in mixture types")
        return self


#: Keys added to :class:`CorruptionConfig` after ``exp0043_xgb_head_4blocks_masked`` was
#: registered (2026-09-18). While one holds the value listed here it is dropped from the
#: config digest, so every head run registered before them keeps its identity
#: (``exp0033`` = ``1f826eb05d``, ``exp0043`` = ``82b8187bc3``). Never remove an entry.
_CORRUPTION_KEYS_ADDED_AFTER_EXP0043: dict[str, Any] = {
    "extra_blocks": [],
    "extra_signals": False,
    "identity_signals": False,
    "clf_columns": [],
    "calibrate": "none",
    "calibrate_min_positives": 200,
    "normal_weight": "none",
    "weight_inner_splits": 2,
    "weight_floor": 0.0,
    # the z-regression schedule-default head (2026-09-18, head-paradox EDA §5); every run
    # registered before it — including ``exp0047`` = ``8b1d8a40ef``, ``exp0057`` =
    # ``204e107996`` and ``exp0058`` = ``cda5f38a84`` — keeps its digest while these hold
    "sched_head": "classifier",
    "z_max_depth": 6,
    "z_learning_rate": 0.05,
    "z_num_boost_round": 500,
    "z_min_child_weight": 50.0,
    "z_min_gap_s": 60.0,
    "z_clip_lo": -2.0,
    "z_clip_hi": 2.0,
    "p_clip": [0.0, 1.0],
    # added 2026-09-19 from the pipeline code review (F7.1, F4, F5); every run registered
    # before them keeps its digest while these hold
    "head_seed": None,
    "sched_default_te_loo": False,
    "z_inner_splits": None,
    # 2026-09-22: zero weight for the rollover-labelled rows in the normal expert's fit
    "rollover_weight": "soft",
}


class MixtureConfig(StrictModel):
    corruption: CorruptionConfig | None = None


class RuntimeConfig(StrictModel):
    """Performance switches. **Nothing in this section may change a number.**

    Every field here exists to make a run *faster*, never different: matrix reuse, an
    on-disk cache, parallel dispatch, thread counts. Each one defaults to the pre-2026-09-19
    behaviour, each is required to leave the predictions unchanged, and the
    whole section is dropped from :meth:`ExperimentConfig.digest` (see there) so turning a
    switch on does not fork an experiment's identity.

    Every field is also overridable by environment variable, because ``experiments/*.yaml``
    are immutable once registered and a cluster job must be able to pick a packing strategy
    without rewriting one. The env var wins; see :mod:`prc.runtime`.
    """

    #: Build each fold's booster matrix (xgb DMatrix / CatBoost Pool) ONCE and reuse it for
    #: every seed of that fold, instead of rebuilding it inside every fit. Costs the memory
    #: of one fold's matrices (cleared at the end of the fold). Env: ``PRC_REUSE_FOLD_MATRIX``.
    reuse_fold_matrix: bool = False
    #: Cache the assembled feature frame (blocks + mask + folds + sample) and the fold-local
    #: target encodings under ``data/cache/``. Env: ``PRC_CACHE_FEATURES``.
    cache_features: bool = False
    #: Cache the corruption head's outputs (out-of-fold p_k, nested normal-expert weights,
    #: test p_k) so capacity variants of one head recompute nothing. Env: ``PRC_CACHE_HEAD``.
    cache_head: bool = False
    #: Dispatch the (fold, seed, group) regressor fits across this many worker processes,
    #: one per visible GPU when the model is cuda. 1 = the sequential path.
    #: Env: ``PRC_FIT_WORKERS``.
    fit_workers: int = 1
    #: Threads handed to the booster (``nthread``) and to polars. None = leave the library
    #: default / whatever the job environment exported. Env: ``PRC_N_THREADS``.
    n_threads: int | None = None
    #: Torch members only (``tabm`` / ``setenc``, 2026-09-22): fit this many
    #: (fold, seed) models concurrently in ``spawn`` worker processes that share the one GPU,
    #: while the parent builds the next fold's matrix. 1 = the sequential path. Results are
    #: assembled in fold order either way. Env: ``PRC_NN_FOLD_WORKERS``. See
    #: ``prc.eval.fitpool.SpawnFitPool`` and internal note 2026-09-22_nn-throughput.
    nn_fold_workers: int = 1
    #: Threads for the torch members' quantile transform (``prc.models.tabm._Prep``), which
    #: then runs over 32 k-row chunks in a thread pool — elementwise, so bit-identical to the
    #: single call. 0 = the single sklearn call. Env: ``PRC_NN_PREP_THREADS``.
    nn_prep_threads: int = 0
    #: Where the fold matrices handed to the spawn workers are spilled (one pickle per fold,
    #: deleted when its fits are done). None = ``$TMPDIR``. Env: ``PRC_NN_SPILL_DIR``.
    nn_spill_dir: str | None = None

    @model_validator(mode="after")
    def _check_runtime(self) -> RuntimeConfig:
        if self.fit_workers < 1:
            raise ValueError("runtime.fit_workers must be >= 1")
        if self.nn_fold_workers < 1:
            raise ValueError("runtime.nn_fold_workers must be >= 1")
        if self.nn_prep_threads < 0:
            raise ValueError("runtime.nn_prep_threads must be >= 0")
        if self.n_threads is not None and self.n_threads < 1:
            raise ValueError("runtime.n_threads must be >= 1 or null")
        return self


class CVConfig(StrictModel):
    """The fold scheme, and how the normal-row regressor is split across airports.

    ``group_by_airport`` was declared on day one and **never read** — campaign 12's
    ``exp0132``-``exp0134`` set it and came out identical to their global twins. Implemented 2026-09-20:

    * ``false`` (default) — one global regressor per (fold, seed, regime group); every byte
      of every pre-2026-09-20 run is reproduced.
    * ``true`` — one regressor per **airport** per (fold, seed, regime group). A model is
      fitted on that airport's training rows only and predicts that airport's validation and
      test rows only.
    * ``[LIRF, LTFM]`` — the hybrid: the named airports get their own model, every other
      airport shares one ``rest`` model. The named airports must exist in the data.

    The corruption head stays **global** in all three cases (it is fitted once per fold over
    all rows), and the target encodings are unchanged (fold-local, fitted on all of the
    fold's training rows) — splitting the regressor does not re-fit either.
    """

    scheme: str = "month_jan_jul"  # see prc.cv.folds.SCHEMES
    group_by_airport: bool | list[str] = False

    @model_validator(mode="after")
    def _check_cv(self) -> CVConfig:
        g = self.group_by_airport
        if isinstance(g, list):
            if not g:
                raise ValueError(
                    "cv.group_by_airport: [] selects no airport; use false for one global model"
                )
            aps = [str(a).strip().upper() for a in g]
            if len(set(aps)) != len(aps):
                raise ValueError(f"cv.group_by_airport has duplicates: {aps}")
            self.group_by_airport = sorted(aps)
        return self


class SeasonWeightConfig(StrictModel):
    """Season-targeted training weights for the normal regressor (2026-09-23, :mod:`prc.eval.season`).

    ``w_row *= max(floor, exp(-d / tau_months))`` with ``d`` the circular month distance
    between the row's calendar month and the fit's target month: a fold's validation
    month(s), and under ``test_fit: full`` one full fit per test month. The corruption head
    and the target encodings stay unweighted.
    """

    tau_months: float = 2.0
    floor: float = 0.25

    @model_validator(mode="after")
    def _check_season(self) -> SeasonWeightConfig:
        if not self.tau_months > 0:
            raise ValueError("season_weight.tau_months must be > 0")
        if not 0.0 < self.floor <= 1.0:
            raise ValueError("season_weight.floor must be in (0, 1]")
        return self


class StackMemberConfig(StrictModel):
    """H1 (2026-09-23, :mod:`prc.eval.stack`).

    Another member's normal-expert prediction as regressor columns, fold-safe: fold ``k``'s
    training rows get the member's inner-fold prediction (``inner_groups`` interleaved month
    groups of fold ``k``'s training months, never month ``k``), its validation rows the member
    fitted on fold ``k``'s training months, the test rows the member fitted on all of 2025.
    The values are read from the table ``scripts/stack_inner.py build`` writes.
    """

    config: str  # the member's experiment name (experiments/<config>.yaml)
    inner_groups: int = 3
    # "mean": the member's seed average (stk_mean); "matched": the member seed equal to the
    # run's single seed (stk_matched)
    columns: list[Literal["mean", "matched"]] = Field(default_factory=lambda: ["mean"])
    table: str | None = None  # table directory; default data/stack/<config>__g<inner_groups>

    @model_validator(mode="after")
    def _check_stack(self) -> StackMemberConfig:
        if self.inner_groups < 2:
            raise ValueError("stack_member.inner_groups must be >= 2")
        if not self.columns or len(set(self.columns)) != len(self.columns):
            raise ValueError("stack_member.columns must be a non-empty list without repeats")
        return self


class ValidationConfig(StrictModel):
    """Which rows the fold models early-stop on (2026-09-23; review 2026-09-23 B, finding 9).

    * ``early_stop: fold`` (default) — the fold's validation rows, i.e. the month the fold is
      scored on. The pre-2026-09-23 behaviour; the whole section is dropped from the digest
      while it is absent. It makes the OOF optimistic, most for the neural members, which
      restore their best epoch on those rows.
    * ``early_stop: inner`` — a seeded random ``inner_frac`` of the **training airport-days**
      (airport x UTC date of the movement) is held out of every fold's fit and used as the
      early-stopping set; the model then predicts the validation month, which it never saw.
      An airport-day is in the inner set in every fold that trains on it (the draw is keyed by
      the airport-day and ``inner_seed``, not by the fold or the model seed), so one fold's
      seeds share their matrices. The full fit (``test_fit: full``) has no early stopping and
      is unchanged; its round count is scaled from the inner-trained row counts.
    """

    early_stop: Literal["fold", "inner"] = "fold"
    inner_frac: float = 0.1
    inner_seed: int = 0

    @model_validator(mode="after")
    def _check_validation(self) -> ValidationConfig:
        if not 0.0 < self.inner_frac < 0.5:
            raise ValueError("validation.inner_frac must be in (0, 0.5)")
        return self


class ExperimentConfig(StrictModel):
    name: str
    description: str = ""
    features: list[str]  # names registered in prc.features.registry
    cv: CVConfig = Field(default_factory=CVConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    target: TargetConfig = Field(default_factory=TargetConfig)
    mixture: MixtureConfig | None = None
    # Performance switches (2026-09-19). Never part of the experiment's identity — see
    # `RuntimeConfig` and the `digest()` note below.
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    # fold-safe target-statistic groups (names from prc.features.target_enc.DEFAULT_GROUPS)
    target_encoding: list[str] = Field(default_factory=list)
    # Feature blocks whose columns are set to null on departures with no NM flight-plan record
    # (regime C). One model, but the tree cannot use surface-state features to fit the
    # catastrophic no-flight-plan labels (every such block cost 14-25 s on that lane, 2026-09-18).
    mask_for_no_nm: list[str] = Field(default_factory=list)
    te_alpha: float = 20.0
    # F6 (2026-09-19 code review). Fit the target statistics on rows whose label is a taxi
    # time (`prc.features.target_enc.clean_mask`: y <= 3600 s and not the schedule-default
    # artefact) instead of on every training row. Filters the FIT set only, inside the fold,
    # so it cannot select the scored population. Default off = the pre-2026-09-19 behaviour.
    te_clean_labels: bool = False
    te_stats: list[Literal["mean", "p10", "p50", "n"]] = Field(default_factory=lambda: ["mean"])
    # T1 (2026-09-21). How the TRAINING rows'
    # target encodings are fitted; validation and test rows are unaffected.
    #   "self"        : `encode(train, train, ...)`, each row's statistics include its own label
    #                   (the 2026-09-20 parity audit). The pre-2026-09-21 behaviour; dropped
    #                   from the digest while it holds this default.
    #   "inner_month" : a training row of month m is encoded from the fit frame's rows of the
    #                   other months (`prc.features.target_enc.encode_inner_month`), the same
    #                   out-of-month semantics the served rows have.
    te_train_mode: Literal["self", "inner_month"] = "self"
    # Stage-1 latents (2026-09-22). Model-based latent
    # estimates added to the regressor's matrices like target encodings, fitted inside each
    # fold (`prc.features.latent_stage1`): `s1_unimpeded` (a small GBM on clean, low-congestion
    # rows, static features only), `s1_additional` (a small GBM on the residual from the
    # queue / weather / calendar features) and their sum. Training rows get inner
    # leave-one-month-out values; served rows get values fitted on the whole fit frame.
    #   "off" : the pre-2026-09-22 behaviour; dropped from the digest while it holds it.
    stage1: Literal["off", "unimpeded_additional"] = "off"
    seeds: list[int] = Field(default_factory=lambda: [42])
    sample_frac: float | None = None  # subsample train rows for fast iteration
    # F6 (2026-09-19 code review). Which rows `sample_frac` thins.
    #   "all"        : every training-table row, the pre-2026-09-19 behaviour. Because the
    #                  sample is taken BEFORE the fold split it thins the validation rows
    #                  too, so a sub-sampled run's pooled RMSE is not comparable to a full
    #                  run's (exp0063 scored 86,220 OOF rows instead of 344,419, and the
    #                  "learning curve" reads backwards: 25 % = 342.3 < 50 % = 375.3 <
    #                  100 % = 407.0, because the small samples were scored on easier rows).
    #   "train_only" : keep every row that any fold validates (`fold >= 0`) and thin only the
    #                  never-validated rest, so the fast loop scores the SAME population as
    #                  the gate and its number can be compared to one.
    sample_frac_scope: Literal["all", "train_only"] = "all"
    # Feature columns removed from the regressor's matrices after the blocks are assembled and
    # before the matrices are built, for train and test alike (2026-09-20). It is the
    # column-level counterpart of leaving a block out of `features:` — for the case where a
    # block earns its place through some of its columns and loses through one of them, so
    # dropping the whole block is too coarse. Names are *column* names, not block names; a
    # name that no assembled column carries is an error, so a typo cannot silently do nothing.
    # Nothing else consults it: the head's `extra_blocks` / `clf_columns`, `mask_for_no_nm`
    # and the target encodings are untouched. Default [] = the pre-2026-09-20 behaviour, and
    # it is dropped from the digest while it holds that default, so every registered run keeps
    # its identity.
    drop_columns: list[str] = Field(default_factory=list)
    # Which models produce `test_pred.parquet` (2026-09-20). Default "folds" = the
    # pre-2026-09-20 behaviour, and it is dropped from the digest while it holds that
    # default, so every registered run keeps its identity.
    #   "folds" : the mean of the fold models, as before. Under `month_jan_jul` those models
    #             never saw Jan 2025 or Jul 2025, so the 2026 test months are predicted by
    #             models that are blind to the two months most like them; under `month_loo`
    #             every one of them is an 11-month model.
    #   "full"  : after CV, one model per (seed, regime group, airport unit) is fitted on ALL
    #             training rows for a fixed number of rounds (no early stopping, no
    #             validation set) and those models write `test_pred`. OOF and every metric
    #             are untouched — the fold models still produce them. The number of rounds is
    #             derived from the fold models' own stopping points; see
    #             `prc.eval.run._full_fit_rounds`.
    test_fit: Literal["folds", "full"] = "folds"
    # Lever 1 (2026-09-23). Optional season-targeted weights for the normal regressor
    # (`SeasonWeightConfig`). None = the pre-2026-09-23 behaviour; dropped from the digest
    # while it holds that default. Requires `test_fit: full`: the fold models are targeted at
    # their own validation month, so only the per-test-month full fits may write `test_pred`.
    season_weight: SeasonWeightConfig | None = None
    # H1 (2026-09-23). Optional stacked member (`StackMemberConfig`). None = the
    # pre-2026-09-23 behaviour; dropped from the digest while it holds that default.
    stack_member: StackMemberConfig | None = None
    # Measurement P2 (2026-09-23). Optional early-stopping source of the fold models
    # (`ValidationConfig`). None = the pre-2026-09-23 behaviour (early-stop on the fold's
    # validation rows); dropped from the digest while it holds that default.
    validation: ValidationConfig | None = None
    # Metadata only: nothing in `src/` reads it. It is hashed by `digest()` (so it cannot
    # be edited on a registered YAML without forking the identity) and read by humans.
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_mixture(self) -> ExperimentConfig:
        # `mixture: {}` (or `mixture: {corruption: null}`) used to load and mean "no head",
        # which is what `mixture:` omitted already says; a half-deleted head block would pass
        # silently. No registered YAML uses it (checked 2026-09-21), so it is an error.
        if self.mixture is not None and self.mixture.corruption is None:
            raise ValueError(
                "mixture is set but has no corruption head (mixture: {} or corruption: null); "
                "omit `mixture:` for the plain single-model path, or give mixture.corruption"
            )
        return self

    @model_validator(mode="after")
    def _check_season_weight(self) -> ExperimentConfig:
        if self.season_weight is not None and self.test_fit != "full":
            raise ValueError(
                "season_weight targets each fit at the month it predicts; the fold models are "
                "targeted at their validation month, so set test_fit: full (one full fit per "
                "test month writes test_pred)"
            )
        return self

    @model_validator(mode="after")
    def _check_eval_metric_space(self) -> ExperimentConfig:
        m = self.model
        if m.log_link():
            # the wrappers' raw metric and label handling assume the labels are seconds
            if self.target.transform != "none":
                raise ValueError(
                    "a log-link objective (Poisson / Tweedie / Gamma) already targets the mean "
                    f"in seconds; it needs target.transform: none, got {self.target.transform!r}"
                )
            gamma = str(m.params.get("objective", "")) == "reg:gamma"
            if (
                m.label_clip_min is None
                or m.label_clip_min < 0
                or (gamma and m.label_clip_min <= 0)
            ):
                raise ValueError(
                    "a log-link objective needs non-negative training labels (Gamma: positive) "
                    "and the raw label goes below 0; set model.label_clip_min: 0 "
                    f"(> 0 for reg:gamma), got {m.label_clip_min!r}"
                )
            return self
        if m.eval_metric_space == "raw":
            if self.target.transform != "sqrt":
                raise ValueError(
                    "model.eval_metric_space: raw inverts a sqrt target; it needs "
                    f"target.transform: sqrt, got {self.target.transform!r} (or a log-link "
                    "objective with target.transform: none)"
                )
            if self.model.kind not in ("xgb", "cat"):
                raise ValueError(
                    f"model.eval_metric_space: raw is implemented for xgb / cat, not "
                    f"{self.model.kind!r}"
                )
        return self

    @classmethod
    def load(cls, path: str | Path) -> ExperimentConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)

    def digest(self) -> str:
        """Stable identity of the config.

        Keys added after a run was registered are dropped when they hold their default, so a
        config that does not use the mixture machinery keeps the digest it was registered with
        (``exp0018_xgb_cap_d12_m1000`` = ``1bf12191dd``). A config that *does* use them hashes
        them like everything else.

        ``runtime`` is the one section dropped **unconditionally**, at its default and away
        from it (2026-09-19). It holds performance switches only — matrix reuse, the on-disk
        cache, parallel dispatch, thread counts — every one of which is required to leave
        the predictions bit-identical. Two runs
        that differ only in how fast they got there are the same experiment, so a packed
        suite that turns the cache on must not silently fork the digest of a registered run.
        """
        return hashlib.sha1(self._digest_payload()).hexdigest()[:10]

    def _digest_payload(self, prose: bool = True) -> bytes:
        """The canonical bytes :meth:`digest` and :meth:`content_digest` hash.

        ``prose=False`` drops ``name`` / ``description`` / ``tags`` — see
        :meth:`content_digest` for why that is a separate method rather than a change here.
        """
        d = self.model_dump()
        d.pop("runtime", None)
        if not prose:
            for key in ("name", "description", "tags"):
                d.pop(key, None)
        if d.get("model", {}).get("eval_metric_space") == "transformed":
            d["model"].pop("eval_metric_space", None)
        if d.get("model", {}).get("label_clip_min", 0) is None:
            d["model"].pop("label_clip_min", None)
        target = d.get("target", {})
        if target.get("mode") == "raw":
            target.pop("mode", None)
        if target.get("split_groups") is None:
            target.pop("split_groups", None)
        if not d.get("mask_for_no_nm"):
            d.pop("mask_for_no_nm", None)
        if d.get("sample_frac_scope") == "all":
            d.pop("sample_frac_scope", None)
        if not d.get("te_clean_labels"):
            d.pop("te_clean_labels", None)
        if d.get("te_train_mode") == "self":
            d.pop("te_train_mode", None)
        if d.get("stage1") == "off":
            d.pop("stage1", None)
        if not d.get("drop_columns"):
            d.pop("drop_columns", None)
        if d.get("test_fit") == "folds":
            d.pop("test_fit", None)
        if d.get("season_weight") is None:
            d.pop("season_weight", None)
        if d.get("stack_member") is None:
            d.pop("stack_member", None)
        # absent, or `early_stop: fold` (whose other knobs are then unread): the old identity
        if (d.get("validation") or {}).get("early_stop", "fold") == "fold":
            d.pop("validation", None)
        if d.get("mixture") is None:
            d.pop("mixture", None)
        else:
            corr = (d["mixture"] or {}).get("corruption")
            if corr is not None:
                for key, default in _CORRUPTION_KEYS_ADDED_AFTER_EXP0043.items():
                    if corr.get(key) == default:
                        corr.pop(key, None)
        return json.dumps(d, sort_keys=True, default=str).encode()

    def content_digest(self) -> str:
        """Identity of the *experiment*, ignoring its prose: no ``name``, ``description`` or
        ``tags``.

        This is the duplicate detector. :meth:`digest` hashes the prose too, which the
        2026-09-19 code review flagged (F14): two configs differing only in ``name`` give
        different digests, so "same digest, different YAML" can never fire, and editing a
        ``description:`` forks an experiment's identity.

        The obvious fix — drop the three keys from :meth:`digest` — would change **every one
        of the 55 digests already written into ``experiments/registry.csv``**, and the only
        way to keep those valid would be a hand-pinned table of old values that then has to
        be maintained forever. So ``digest()`` is deliberately left alone (it is the
        registry's key and must stay stable) and the detector it should have been lives here
        instead. Use ``content_digest()`` to ask "have we already run this?", ``digest()`` to
        ask "which registry row is this?".
        """
        return hashlib.sha1(self._digest_payload(prose=False)).hexdigest()[:10]

    def corruption(self) -> CorruptionConfig | None:
        return self.mixture.corruption if self.mixture else None

    def regime_groups(self) -> list[list[str]]:
        """The groups of regimes that get their own model (one group = the plain single model)."""
        if self.target.mode == "residual_proxy_split":
            return [list(g) for g in (self.target.split_groups or [["A"], ["B", "C"]])]
        return [["A", "B", "C"]]

    def inner_early_stop(self) -> bool:
        """``validation.early_stop: inner`` — the fold models early-stop on held-out training
        airport-days instead of the scored month."""
        return self.validation is not None and self.validation.early_stop == "inner"

    def uses_regimes(self) -> bool:
        return self.target.mode != "raw" or self.corruption() is not None
