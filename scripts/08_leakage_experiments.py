"""Phase 8 driver: measure what each kind of leakage would have bought us.

    PYTHONPATH=. python scripts/08_leakage_experiments.py

Four experiments, each a matched pair (or triple) differing in exactly one
decision, all scored on the *same* chronological holdout unless stated:

A. **Temporal leakage** — random shuffled split vs chronological split.
B. **Feature leakage** — three target-encoding variants: full-dataset,
   zero-lag expanding, 30-day-lagged expanding.
C. **Preprocessing leakage** — imputer + scaler fitted on the full frame vs on
   the training fold only.
D. **Static PIT check** — the three machine audits from src/leakage/audit.py.

Writes reports/results/08_*.csv|json and reports/figures/08_leakage_impact.png.
"""
from __future__ import annotations

import json
import logging
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src import config
from src.data.split import chronological_split, expanding_window_folds
from src.evaluation.evaluate import load_matrices
from src.evaluation.metrics import evaluate_scores
from src.features.build_features import FeatureBuilder, build_feature_matrix
from src.leakage.audit import (
    audit_fit_on_train,
    audit_future_invariance,
    audit_point_in_time,
    leaky_target_encode_full_data,
)
from src.models.base import view_columns
from src.models.gbm import GBMModel

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")

SEED = config.RANDOM_SEED


# ---------------------------------------------------------------------------
def experiment_a_temporal(X, meta) -> pd.DataFrame:
    """Chronological vs random split, same model, same 20% test size.

    The comparison is deliberately favourable to the random split: it gets the
    same model, the same features and the same amount of training data. The
    only difference is *which* rows are held out.
    """
    y = meta[config.TARGET].to_numpy()
    cols = view_columns(X, "base")
    rng = np.random.default_rng(SEED)

    rows = []
    sp = chronological_split(meta)
    tr_c = np.concatenate([sp.train_idx, sp.valid_idx])
    m = GBMModel(n_estimators=600, early_stopping_rounds=0)
    m.fit(X.iloc[tr_c][cols], y[tr_c])
    r = evaluate_scores(y[sp.test_idx], m.predict_proba(X.iloc[sp.test_idx][cols]))
    r["split"] = "chronological (correct)"
    rows.append(r)

    perm = rng.permutation(len(meta))
    n_test = len(sp.test_idx)
    te_r, tr_r = perm[:n_test], perm[n_test:]
    m2 = GBMModel(n_estimators=600, early_stopping_rounds=0)
    m2.fit(X.iloc[tr_r][cols], y[tr_r])
    r2 = evaluate_scores(y[te_r], m2.predict_proba(X.iloc[te_r][cols]))
    r2["split"] = "random shuffled (leaky)"
    rows.append(r2)
    return pd.DataFrame(rows)


def experiment_b_target_encoding(df_joined) -> pd.DataFrame:
    """Three target-encoding variants, scored on the same chronological holdout.

    Only the four `*_te` columns differ between arms; every other feature is
    byte-identical, so any difference in the holdout number is attributable to
    the encoding alone.
    """
    order = np.argsort(df_joined[config.TIME_COL].to_numpy(), kind="stable")
    d = df_joined.iloc[order].reset_index(drop=True)
    meta = d[[config.ID_COL, config.TIME_COL, config.TARGET]]
    y = d[config.TARGET].to_numpy()
    sp = chronological_split(meta)
    train_mask = np.zeros(len(d), bool)
    train_mask[sp.train_idx] = True
    tr = np.concatenate([sp.train_idx, sp.valid_idx])

    arms = {}
    for tag, lag in [("expanding, 30-day label lag (used)", 30),
                     ("expanding, zero lag (subtly leaky)", 0)]:
        fb = FeatureBuilder(label_lag_days=lag).fit(d, train_mask)
        arms[tag] = fb.transform(d)

    # Fully leaky arm: overwrite the *_te columns with a full-dataset encoding
    # that includes the row's own label and every future label.
    leaky = arms["expanding, 30-day label lag (used)"].copy()
    from src.features.build_features import TARGET_ENCODE_COLS, _concat_key
    for col in TARGET_ENCODE_COLS:
        src_col = (d[col] if col in d.columns
                   else _concat_key(d, col.split("_x_")))
        gid = pd.factorize(src_col.astype("string").fillna("NA"),
                           use_na_sentinel=False)[0]
        leaky[f"{col}_te"] = leaky_target_encode_full_data(gid, y)
    arms["full-dataset target encoding (leaky)"] = leaky

    rows = []
    for tag, Xa in arms.items():
        cols = view_columns(Xa, "base")
        m = GBMModel(n_estimators=600, early_stopping_rounds=0)
        m.fit(Xa.iloc[tr][cols], y[tr])
        r = evaluate_scores(y[sp.test_idx], m.predict_proba(Xa.iloc[sp.test_idx][cols]))
        r["arm"] = tag
        if m.feature_importance() is not None:
            imp = m.feature_importance()
            total = imp.sum()
            r["te_share_of_gain"] = float(
                imp[[c for c in imp.index if c.endswith("_te")]].sum() / total)
        rows.append(r)
        logging.info("B %-42s PR-AUC %.4f", tag, r["pr_auc"])
    return pd.DataFrame(rows)


def experiment_c_preprocessing(X, meta) -> pd.DataFrame:
    """Imputer + scaler fitted on the full frame vs on the training fold only.

    Uses a plain logistic regression so the effect is not absorbed by a model
    that is insensitive to scaling. The leak here is quieter than the others:
    no label is involved, only the covariate distribution of the future.
    """
    y = meta[config.TARGET].to_numpy()
    # Drop the categorical codes: one-hot would dominate the runtime and the
    # question here is about the *numeric* preprocessing statistics.
    cols = [c for c in view_columns(X, "compact") if not c.endswith("_code")]
    sp = chronological_split(meta)
    tr = np.concatenate([sp.train_idx, sp.valid_idx])
    te = sp.test_idx
    Xn = X[cols].replace([np.inf, -np.inf], np.nan)

    rows = []
    for tag, fit_idx in [("fitted on train only (correct)", tr),
                         ("fitted on train+test (leaky)", np.arange(len(X)))]:
        imp = SimpleImputer(strategy="median").fit(Xn.iloc[fit_idx])
        sc = StandardScaler().fit(imp.transform(Xn.iloc[fit_idx]))
        Xtr = np.clip(sc.transform(imp.transform(Xn.iloc[tr])), -10, 10)
        Xte = np.clip(sc.transform(imp.transform(Xn.iloc[te])), -10, 10)
        clf = LogisticRegression(C=0.1, max_iter=200, class_weight="balanced",
                                 random_state=SEED).fit(Xtr, y[tr])
        r = evaluate_scores(y[te], clf.predict_proba(Xte)[:, 1])
        r["arm"] = tag
        rows.append(r)
        logging.info("C %-34s PR-AUC %.4f", tag, r["pr_auc"])
    return pd.DataFrame(rows)


def experiment_d_static_audits(df_joined) -> dict:
    """The three machine checks, on a 120k-row contiguous time slice."""
    from src.features.build_features import UID_DEFINITIONS

    d = (df_joined.sort_values(config.TIME_COL).reset_index(drop=True)
         .iloc[:120_000].copy())
    mask = np.zeros(len(d), bool)
    mask[:int(len(d) * 0.65)] = True
    Xa, _ = build_feature_matrix(d, mask)

    pit = audit_point_in_time(d, Xa, UID_DEFINITIONS["uid_card_addr"],
                              "uid_card_addr", n_sample=300)
    fut = audit_future_invariance(d, mask)
    fit = audit_fit_on_train(d, mask)

    unexpected = [f for f in fit["feature"]
                  if not (f.endswith("_freq") or f.endswith("_te")
                          or f.endswith("_code") or f == "amt_decile")]
    return {
        "pit_rows_checked": int(len(pit)),
        "pit_prior_count_exact": bool(pit["count_ok"].all()),
        "pit_prior_mean_exact": bool(pit["mean_ok"].all()),
        "n_features": int(len(fut)),
        "n_features_leaking_future": int(fut["leaks_future"].sum()),
        "features_leaking_future": fut.loc[fut["leaks_future"], "feature"].tolist(),
        "n_features_depending_on_fit_population": int(len(fit)),
        "unexpected_fit_dependencies": unexpected,
    }


def main() -> None:
    X, meta = load_matrices()
    df_joined = pd.read_parquet(config.INTERIM_DIR / "train_joined.parquet")

    logging.info("=== D: static audits ===")
    d_res = experiment_d_static_audits(df_joined)
    (config.RESULTS_DIR / "08_static_audits.json").write_text(
        json.dumps(d_res, indent=2))
    print(json.dumps(d_res, indent=2))

    logging.info("=== A: temporal leakage ===")
    a = experiment_a_temporal(X, meta)
    a.to_csv(config.RESULTS_DIR / "08_temporal.csv", index=False)
    print(a[["split", "n", "base_rate", "pr_auc", "pr_auc_lift", "roc_auc",
             "precision@1%", "recall@1%"]].round(4).to_string(index=False))

    logging.info("=== B: target-encoding leakage ===")
    b = experiment_b_target_encoding(df_joined)
    b.to_csv(config.RESULTS_DIR / "08_target_encoding.csv", index=False)
    print(b[["arm", "pr_auc", "pr_auc_lift", "roc_auc", "precision@1%",
             "te_share_of_gain"]].round(4).to_string(index=False))

    logging.info("=== C: preprocessing leakage ===")
    c = experiment_c_preprocessing(X, meta)
    c.to_csv(config.RESULTS_DIR / "08_preprocessing.csv", index=False)
    print(c[["arm", "pr_auc", "roc_auc"]].round(4).to_string(index=False))

    # --- figure -----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, frame, keycol, title in [
        (axes[0], a, "split", "Temporal: split choice"),
        (axes[1], b, "arm", "Feature: target encoding"),
        (axes[2], c, "arm", "Preprocessing: scaler fit"),
    ]:
        labels = [str(v).replace(" (", "\n(") for v in frame[keycol]]
        colours = ["#4C72B0" if "correct" in str(v) or "used" in str(v)
                   else "#C44E52" for v in frame[keycol]]
        ax.bar(labels, frame["pr_auc"], color=colours)
        ax.set_ylabel("PR-AUC"); ax.set_title(title)
        ax.tick_params(axis="x", labelsize=7)
        for i, v in enumerate(frame["pr_auc"]):
            ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "08_leakage_impact.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
