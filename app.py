"""
AML Project 4 — Streamlit Presentation App
Course: CSCI-6767 Applied Machine Learning & Data Analytics — Spring 2026
Team: Gabil Gurbanov, Hamida Hagverdiyeva

All models are pre-computed on first launch and cached — subsequent page
navigation is instant.
"""

import io
import os
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import joblib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import streamlit as st
from scipy.interpolate import interp1d

from statsmodels.nonparametric.smoothers_lowess import lowess

from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.preprocessing import (
    PolynomialFeatures, StandardScaler, LabelEncoder, SplineTransformer
)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc as sklearn_auc, log_loss
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.datasets import fetch_kddcup99
from pygam import LinearGAM, s
from pygam import terms as gam_terms
import patsy
from ucimlrepo import fetch_ucirepo

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AML Project 4 — Non-Linearity, Trees & SVM",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

SEED = 42
np.random.seed(SEED)

PALETTE = {
    "blue":   "#2E86AB",
    "red":    "#E84855",
    "orange": "#F18F01",
    "green":  "#3BB273",
    "purple": "#7B2D8B",
    "gray":   "#6C757D",
    "teal":   "#00B4D8",
    "gold":   "#E9C46A",
}
CLIST = list(PALETTE.values())

plt.rcParams.update({
    "figure.dpi": 100,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})

# ─────────────────────────────────────────────────────────────────────────────
#  DISK CACHE  — models saved here so retraining only happens once ever
# ─────────────────────────────────────────────────────────────────────────────
_CACHE_DIR  = Path("model_cache")
_CACHE_PKL  = _CACHE_DIR / "results.pkl"
_FIGS_DIR   = _CACHE_DIR / "figs"


def _save_to_disk(R: dict) -> None:
    """Persist the pre-computation results to disk."""
    _CACHE_DIR.mkdir(exist_ok=True)
    _FIGS_DIR.mkdir(exist_ok=True)
    figs = R.get("figs", {})
    # Save every figure as a PNG file
    for name, buf in figs.items():
        buf.seek(0)
        (_FIGS_DIR / f"{name}.png").write_bytes(buf.read())
        buf.seek(0)
    # Save everything else (models, arrays, DataFrames) with joblib
    R_no_figs = {k: v for k, v in R.items() if k != "figs"}
    joblib.dump(R_no_figs, _CACHE_PKL, compress=3)


def _load_from_disk() -> dict | None:
    """Load previously saved results; returns None if cache is missing or corrupt."""
    if not _CACHE_PKL.exists():
        return None
    try:
        R = joblib.load(_CACHE_PKL)
        figs: dict[str, io.BytesIO] = {}
        if _FIGS_DIR.exists():
            for png_path in _FIGS_DIR.glob("*.png"):
                figs[png_path.stem] = io.BytesIO(png_path.read_bytes())
        R["figs"] = figs
        return R
    except Exception:
        return None  # corrupt cache → recompute



# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def fig_to_img(fig):
    """Convert matplotlib figure to PNG bytes (for st.image)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def reg_row(name, y_true, y_pred):
    return {
        "Model": name,
        "RMSE":  round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "MAE":   round(float(mean_absolute_error(y_true, y_pred)), 2),
        "R²":    round(float(r2_score(y_true, y_pred)), 4),
    }


def clf_row(name, y_true, y_pred, y_prob=None):
    auc_val = roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan
    return {
        "Model":     name,
        "Accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "Precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "Recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "F1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "AUC-ROC":   round(float(auc_val), 4) if not np.isnan(auc_val) else "N/A",
    }


def highlight_min(s):
    best = s.min()
    return ["background-color:#d4edda;font-weight:700" if v == best else "" for v in s]


def highlight_max(s):
    best = s.max()
    return ["background-color:#d4edda;font-weight:700" if v == best else "" for v in s]


def info_box(title, body, color="#2E86AB"):
    st.markdown(
        f"""<div style="border-left:4px solid {color};padding:10px 16px;
        background:#f8f9fa;border-radius:4px;margin:8px 0;">
        <b>{title}</b><br>{body}</div>""",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  MASTER PRE-COMPUTATION  — runs once, cached for the session
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def precompute_all():
    """
    Train ALL models and pre-generate ALL static figures.
    On first run: trains models and saves everything to model_cache/ on disk.
    On subsequent runs: loads instantly from disk — no retraining needed.
    Returns a dict R with everything pages need.
    """
    # ── Try loading from disk first ───────────────────────────────────────
    cached = _load_from_disk()
    if cached is not None:
        return cached

    R = {}

    # ── 1. Load raw data ──────────────────────────────────────────────────
    bike_repo = fetch_ucirepo(id=560)
    bike_df = pd.concat([bike_repo.data.features, bike_repo.data.targets], axis=1)
    bike_df.columns = bike_df.columns.str.strip()
    R["bike_df"] = bike_df

    kdd_raw = fetch_kddcup99(subset=None, percent10=True, as_frame=True, random_state=SEED)
    kdd_df = kdd_raw.frame.copy()
    kdd_df.columns = kdd_df.columns.tolist()[:-1] + ["label"]
    for col in kdd_df.select_dtypes(include="object").columns:
        kdd_df[col] = kdd_df[col].apply(
            lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
        )
    kdd_df["binary_label"] = (kdd_df["label"] != "normal.").astype(int)
    R["kdd_df"] = kdd_df

    # ── 2. Regression pre-processing ─────────────────────────────────────
    TARGET_BIKE = "Rented Bike Count"
    bike = bike_df.copy()
    if "Date" in bike.columns:
        bike = bike.drop(columns=["Date"])
    cat_cols = [c for c in ["Seasons", "Holiday", "Functioning Day"] if c in bike.columns]
    bike = pd.get_dummies(bike, columns=cat_cols, drop_first=True)
    Xb = bike.drop(columns=[TARGET_BIKE])
    yb = bike[TARGET_BIKE].values.astype(float)
    Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(Xb, yb, test_size=0.2, random_state=SEED)
    R["bike_feat_names"] = Xb.columns.tolist()
    R["Xb_tr"] = Xb_tr; R["Xb_te"] = Xb_te
    R["yb_tr"] = yb_tr; R["yb_te"] = yb_te

    # ── 3. Regression models ──────────────────────────────────────────────
    reg = {}

    # Baseline
    m = LinearRegression().fit(Xb_tr, yb_tr)
    reg["Baseline (Linear)"] = {"model": m, "pred": m.predict(Xb_te)}

    # Polynomial degrees 2–4 (with scaling)
    for deg in [2, 3, 4]:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("poly",   PolynomialFeatures(degree=deg, include_bias=False)),
            ("ridge",  RidgeCV(alphas=[1, 10, 100, 1000, 10000])),
        ]).fit(Xb_tr, yb_tr)
        key = f"Polynomial deg={deg}"
        reg[key] = {
            "model": pipe,
            "pred":  pipe.predict(Xb_te),
            "alpha": pipe["ridge"].alpha_,
            "n_feats": pipe["poly"].n_output_features_,
        }

    # Step Functions
    def make_step(X_df, cols, nbins):
        out = X_df.copy()
        for c, nb in zip(cols, nbins):
            if c not in X_df.columns:
                continue
            binned = pd.cut(X_df[c], bins=nb, labels=False)
            dummies = pd.get_dummies(binned, prefix=c + "_step")
            out = pd.concat([out.drop(columns=[c]), dummies], axis=1)
        return out

    sc, sn = ["Temperature", "Hour", "Humidity"], [6, 8, 5]
    Xb_tr_s = make_step(Xb_tr, sc, sn)
    Xb_te_s  = make_step(Xb_te, sc, sn)
    Xb_tr_s, Xb_te_s = Xb_tr_s.align(Xb_te_s, join="left", axis=1, fill_value=0)
    step_m = Ridge(alpha=1.0).fit(Xb_tr_s, yb_tr)
    reg["Step Functions"] = {"model": step_m, "pred": step_m.predict(Xb_te_s),
                             "Xb_te_step": Xb_te_s}

    # B-Splines
    sp_pipe = Pipeline([
        ("spline", SplineTransformer(n_knots=6, degree=3, include_bias=False)),
        ("ridge",  RidgeCV(alphas=[1, 10, 100, 1000])),
    ]).fit(Xb_tr, yb_tr)
    reg["B-Splines"] = {"model": sp_pipe, "pred": sp_pipe.predict(Xb_te)}

    # Natural Cubic Splines
    def ncs_transform(X_df, df=5):
        parts = []
        for col in X_df.columns:
            vals = X_df[col].values
            try:
                dm = patsy.dmatrix(f"cr(x, df={df}) - 1", {"x": vals}, return_type="matrix")
                part = pd.DataFrame(np.asarray(dm),
                                    columns=[f"{col}_ns{i}" for i in range(dm.shape[1])])
            except Exception:
                part = pd.DataFrame(vals, columns=[col])
            parts.append(part)
        return pd.concat(parts, axis=1).reset_index(drop=True)

    Xb_tr_ncs = ncs_transform(Xb_tr)
    Xb_te_ncs  = ncs_transform(Xb_te)
    Xb_tr_ncs, Xb_te_ncs = Xb_tr_ncs.align(Xb_te_ncs, join="left", axis=1, fill_value=0)
    ncs_m = RidgeCV(alphas=[1, 10, 100, 1000]).fit(Xb_tr_ncs, yb_tr)
    reg["Natural Cubic Splines"] = {"model": ncs_m, "pred": ncs_m.predict(Xb_te_ncs)}

    # GAM
    n_feats = Xb_tr.shape[1]
    gam_term = gam_terms.TermList(*[s(i) for i in range(n_feats)])
    gam = LinearGAM(gam_term).gridsearch(Xb_tr.values.astype(float), yb_tr, progress=False)
    reg["GAM (LinearGAM)"] = {
        "model": gam,
        "pred":  gam.predict(Xb_te.values.astype(float)),
        "gam_obj": gam,
    }
    R["reg"] = reg

    # Regression comparison table
    rows_r = [reg_row(k, yb_te, v["pred"]) for k, v in reg.items()]
    R["reg_df"] = pd.DataFrame(rows_r).sort_values("R²", ascending=False).reset_index(drop=True)

    # ── 4. Classification pre-processing ─────────────────────────────────
    TARGET_KDD = "binary_label"
    kdd = kdd_df.drop(columns=["label"])
    for col in ["protocol_type", "service", "flag"]:
        if col in kdd.columns:
            kdd[col] = LabelEncoder().fit_transform(kdd[col].astype(str))
    Xk = kdd.drop(columns=[TARGET_KDD]).values.astype(float)
    yk = kdd[TARGET_KDD].values
    feat_names_k = kdd.drop(columns=[TARGET_KDD]).columns.tolist()
    R["kdd_feat_names"] = feat_names_k

    Xk_tr, Xk_te, yk_tr, yk_te = train_test_split(
        Xk, yk, test_size=0.2, stratify=yk, random_state=SEED
    )
    scaler_k = StandardScaler()
    Xk_tr_s = scaler_k.fit_transform(Xk_tr)
    Xk_te_s  = scaler_k.transform(Xk_te)
    rng = np.random.default_rng(SEED)
    svm_idx = rng.choice(len(Xk_tr_s), size=min(15000, len(Xk_tr_s)), replace=False)
    Xk_svm_tr = Xk_tr_s[svm_idx]
    yk_svm_tr  = yk_tr[svm_idx]

    R["Xk_tr"] = Xk_tr; R["Xk_te"] = Xk_te
    R["yk_tr"] = yk_tr; R["yk_te"] = yk_te
    R["Xk_te_s"] = Xk_te_s

    # ── 5. Tree models ────────────────────────────────────────────────────
    clf = {}

    bag = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=SEED),
        n_estimators=100, oob_score=True, random_state=SEED, n_jobs=-1
    ).fit(Xk_tr, yk_tr)
    clf["Bagging"] = {
        "model": bag,
        "pred":  bag.predict(Xk_te),
        "prob":  bag.predict_proba(Xk_te)[:, 1],
        "oob":   bag.oob_score_,
    }

    rf = RandomForestClassifier(
        n_estimators=100, oob_score=True, random_state=SEED, n_jobs=-1
    ).fit(Xk_tr, yk_tr)
    clf["Random Forest"] = {
        "model":        rf,
        "pred":         rf.predict(Xk_te),
        "prob":         rf.predict_proba(Xk_te)[:, 1],
        "oob":          rf.oob_score_,
        "importances":  rf.feature_importances_,
    }

    gb = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=4,
        subsample=0.8, random_state=SEED
    ).fit(Xk_tr, yk_tr)
    test_deviance = np.array([
        log_loss(yk_te, yp) for yp in gb.staged_predict_proba(Xk_te)
    ])
    clf["Gradient Boosting"] = {
        "model":        gb,
        "pred":         gb.predict(Xk_te),
        "prob":         gb.predict_proba(Xk_te)[:, 1],
        "train_score":  gb.train_score_,
        "test_deviance": test_deviance,
    }

    # SVM kernels — train on scaled subset, predict on full scaled test
    for kname, kwargs in [
        ("SVM — Linear",     dict(kernel="linear",  C=1.0)),
        ("SVM — RBF",        dict(kernel="rbf",     C=10.0, gamma="scale")),
        ("SVM — Polynomial", dict(kernel="poly",    C=1.0,  degree=3, gamma="scale")),
        ("SVM — Sigmoid",    dict(kernel="sigmoid", C=1.0,  gamma="scale")),
    ]:
        svm = SVC(probability=True, random_state=SEED, **kwargs).fit(Xk_svm_tr, yk_svm_tr)
        clf[kname] = {
            "model": svm,
            "pred":  svm.predict(Xk_te_s),
            "prob":  svm.predict_proba(Xk_te_s)[:, 1],
        }

    R["clf"] = clf

    # Classification comparison table (tree models on full test; SVMs on same test with scale)
    rows_c = []
    for name, d in clf.items():
        yt = yk_te  # all use the same test set labels
        rows_c.append(clf_row(name, yt, d["pred"], d["prob"]))
    R["clf_df"] = (
        pd.DataFrame(rows_c)
        .sort_values("F1", ascending=False)
        .reset_index(drop=True)
    )

    # ── 6. Pre-generate static figures ───────────────────────────────────
    figs = {}

    # --- Regression EDA: 3-panel ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(bike_df["Rented Bike Count"], bins=40,
                 color=PALETTE["blue"], edgecolor="white")
    axes[0].set_title("Target Distribution: Rented Bike Count")
    axes[0].set_xlabel("Hourly Rental Count"); axes[0].set_ylabel("Frequency")

    hourly = bike_df.groupby("Hour")["Rented Bike Count"].mean()
    axes[1].plot(hourly.index, hourly.values, marker="o",
                 color=PALETTE["red"], linewidth=2)
    axes[1].fill_between(hourly.index, hourly.values, alpha=0.15, color=PALETTE["red"])
    axes[1].set_title("Average Rentals by Hour of Day")
    axes[1].set_xlabel("Hour"); axes[1].set_ylabel("Avg Bike Count")
    axes[1].set_xticks(range(0, 24, 3))

    axes[2].scatter(bike_df["Temperature"], bike_df["Rented Bike Count"],
                    alpha=0.25, s=5, color=PALETTE["green"])
    axes[2].set_title("Temperature vs Bike Count")
    axes[2].set_xlabel("Temperature (°C)"); axes[2].set_ylabel("Bike Count")
    plt.tight_layout()
    figs["bike_eda1"] = fig_to_img(fig)

    # --- Regression EDA: seasonal boxplot + humidity ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    bike_plot = bike_df.copy()
    season_map = {1: "Spring", 2: "Summer", 3: "Autumn", 4: "Winter"}
    bike_plot["Season Name"] = bike_plot["Seasons"].map(season_map)
    # Use matplotlib boxplot directly — seaborn 0.13.x has a broken boxplot+hue+legend bug
    _season_order = ["Spring", "Summer", "Autumn", "Winter"]
    _season_data = [
        bike_plot[bike_plot["Season Name"] == s]["Rented Bike Count"].values
        for s in _season_order
    ]
    _bp = axes[0].boxplot(
        _season_data, tick_labels=_season_order, patch_artist=True,
        medianprops={"color": "black", "linewidth": 2},
        boxprops={"linewidth": 1.5},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
    )
    for patch, col in zip(_bp["boxes"], ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3"]):
        patch.set_facecolor(col); patch.set_alpha(0.75)
    axes[0].set_title("Rentals by Season"); axes[0].set_xlabel("")
    axes[1].scatter(bike_df["Humidity"], bike_df["Rented Bike Count"],
                    alpha=0.2, s=5, color=PALETTE["purple"])
    axes[1].set_title("Humidity vs Bike Count")
    axes[1].set_xlabel("Humidity (%)"); axes[1].set_ylabel("Bike Count")
    plt.tight_layout()
    figs["bike_eda2"] = fig_to_img(fig)

    # --- Correlation heatmap ---
    num_cols = bike_df.select_dtypes(include=np.number).columns.tolist()
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(bike_df[num_cols].corr(), annot=True, fmt=".2f",
                cmap="coolwarm", center=0, linewidths=0.5, square=True,
                ax=ax, annot_kws={"size": 8})
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    figs["bike_corr"] = fig_to_img(fig)

    # --- Regression comparison bar ---
    rd = R["reg_df"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors_r = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(rd)))
    for ax, metric, better in zip(axes, ["RMSE", "R²"], ["lower", "higher"]):
        vals = rd[metric]
        cols = colors_r if better == "higher" else colors_r[::-1]
        bars = ax.barh(rd["Model"], vals, color=cols)
        ax.set_xlabel(f"{metric} ({better} is better)")
        ax.set_title(f"Regression Models — {metric}")
        ax.invert_yaxis()
        fmt = "{:.1f}" if metric == "RMSE" else "{:.3f}"
        pad = vals.max() * 0.02
        for bar, val in zip(bars, vals):
            ax.text(bar.get_width() + pad,
                    bar.get_y() + bar.get_height() / 2,
                    fmt.format(val), va="center", fontsize=9)
        ax.set_xlim(0, vals.max() * 1.15)
    plt.tight_layout()
    figs["reg_compare"] = fig_to_img(fig)

    # --- Actual vs predicted (3 best models) ---
    top3 = [k for k in ["GAM (LinearGAM)", "Polynomial deg=3", "B-Splines"]
            if k in reg]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, name in zip(axes, top3):
        ypred = reg[name]["pred"]
        ax.scatter(yb_te, ypred, alpha=0.25, s=6, color=PALETTE["blue"])
        lims = [0, max(float(yb_te.max()), float(ypred.max()))]
        ax.plot(lims, lims, "r--", lw=1.5, label="Perfect fit")
        r2v = r2_score(yb_te, ypred)
        ax.set_title(f"{name}\nR²={r2v:.3f}")
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
        ax.legend(fontsize=8)
    plt.suptitle("Actual vs Predicted — Top 3 Regression Models", y=1.01, fontsize=13)
    plt.tight_layout()
    figs["reg_avp"] = fig_to_img(fig)

    # --- Residual plots ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, name in zip(axes, top3):
        ypred = reg[name]["pred"]
        resid = yb_te - ypred
        ax.scatter(ypred, resid, alpha=0.25, s=5, color=PALETTE["orange"])
        ax.axhline(0, color="black", lw=1.5, linestyle="--")
        ax.set_title(f"Residuals — {name}")
        ax.set_xlabel("Fitted Values"); ax.set_ylabel("Residuals")
    plt.suptitle("Residual Plots — Top 3 Regression Models", y=1.01, fontsize=13)
    plt.tight_layout()
    figs["reg_resid"] = fig_to_img(fig)

    # --- Polynomial degree comparison ---
    degrees = [2, 3, 4]
    rmse_v = [np.sqrt(mean_squared_error(yb_te, reg[f"Polynomial deg={d}"]["pred"])) for d in degrees]
    r2_v   = [r2_score(yb_te, reg[f"Polynomial deg={d}"]["pred"]) for d in degrees]
    nf_v   = [reg[f"Polynomial deg={d}"]["n_feats"] for d in degrees]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(len(degrees))
    w = 0.35
    ax_l, ax_r = axes[0], axes[0].twinx()
    ax_l.bar(x - w/2, rmse_v, w, color=PALETTE["blue"], alpha=0.85, label="RMSE")
    ax_r.bar(x + w/2, r2_v,   w, color=PALETTE["red"],  alpha=0.85, label="R²")
    ax_l.set_xticks(x)
    ax_l.set_xticklabels([f"deg={d}\n({nf} feats)" for d, nf in zip(degrees, nf_v)])
    ax_l.set_ylabel("RMSE", color=PALETTE["blue"]); ax_l.tick_params(axis="y", labelcolor=PALETTE["blue"])
    ax_r.set_ylabel("R²",   color=PALETTE["red"]);  ax_r.tick_params(axis="y", labelcolor=PALETTE["red"])
    ax_l.set_title("Polynomial Regression — Degree Comparison")
    h1, l1 = ax_l.get_legend_handles_labels(); h2, l2 = ax_r.get_legend_handles_labels()
    ax_l.legend(h1+h2, l1+l2, loc="upper right")

    best_d = degrees[int(np.argmin(rmse_v))]
    ypred_best = reg[f"Polynomial deg={best_d}"]["pred"]
    axes[1].scatter(yb_te, ypred_best, alpha=0.3, s=6, color=PALETTE["blue"])
    lims = [0, max(float(yb_te.max()), float(ypred_best.max()))]
    axes[1].plot(lims, lims, "r--", lw=1.5, label="Perfect fit")
    axes[1].set_title(f"Actual vs Predicted — best degree ({best_d})")
    axes[1].set_xlabel("Actual"); axes[1].set_ylabel("Predicted")
    axes[1].legend()
    plt.tight_layout()
    figs["poly_compare"] = fig_to_img(fig)

    # --- Step function fit on temperature ---
    temp_te = Xb_te["Temperature"].values
    ypred_step = reg["Step Functions"]["pred"]
    sidx = np.argsort(temp_te)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(temp_te, yb_te, alpha=0.2, s=8, color="gray", label="Actual")
    ax.scatter(temp_te[sidx], ypred_step[sidx], s=5, alpha=0.5,
               color=PALETTE["orange"], label="Step Function Fit")
    ax.set_title("Step Function Fit — Temperature vs Bike Count")
    ax.set_xlabel("Temperature (°C)"); ax.set_ylabel("Bike Count")
    ax.legend()
    plt.tight_layout()
    figs["step_fit"] = fig_to_img(fig)

    # --- Spline comparison (1D on temperature) ---
    temp_tr = Xb_tr["Temperature"].values
    temp_grid = np.linspace(temp_te.min(), temp_te.max(), 200).reshape(-1, 1)
    sp1d = Pipeline([
        ("spline", SplineTransformer(n_knots=6, degree=3, include_bias=False)),
        ("ridge",  RidgeCV(alphas=[1, 10, 100, 1000])),
    ]).fit(temp_tr.reshape(-1, 1), yb_tr)
    ncs_tr_dm  = patsy.dmatrix("cr(t, df=6)", {"t": temp_tr}, return_type="dataframe")
    ncs_grid_dm = patsy.dmatrix("cr(t, df=6)", {"t": temp_grid.ravel()}, return_type="dataframe")
    ncs1d = RidgeCV(alphas=[1, 10, 100, 1000]).fit(ncs_tr_dm, yb_tr)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(temp_te, yb_te, alpha=0.15, s=6, color="gray", label="Data")
    ax.plot(temp_grid, sp1d.predict(temp_grid), color=PALETTE["blue"], lw=2.5, label="B-Spline")
    ax.plot(temp_grid, ncs1d.predict(ncs_grid_dm), color=PALETTE["red"],
            lw=2.5, linestyle="--", label="Natural Cubic Spline")
    ax.set_title("Spline Fits — Temperature vs Bike Count (1D illustration)")
    ax.set_xlabel("Temperature (°C)"); ax.set_ylabel("Bike Count")
    ax.legend()
    plt.tight_layout()
    figs["spline_1d"] = fig_to_img(fig)

    # --- LOESS bandwidth comparison (static — 3 bands) ---
    temp_all = bike_df["Temperature"].values
    cnt_all  = bike_df["Rented Bike Count"].values
    sidx_all = np.argsort(temp_all)
    xs_all, ys_all = temp_all[sidx_all], cnt_all[sidx_all]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].scatter(xs_all, ys_all, alpha=0.1, s=5, color="gray", label="Data")
    for frac_c, col, lbl in [
        (0.05, PALETTE["red"],    "frac=0.05 (flexible)"),
        (0.10, PALETTE["blue"],   "frac=0.10"),
        (0.20, PALETTE["green"],  "frac=0.20"),
        (0.35, PALETTE["orange"], "frac=0.35 (smooth)"),
    ]:
        sm = lowess(ys_all, xs_all, frac=frac_c, return_sorted=True)
        axes[0].plot(sm[:, 0], sm[:, 1], lw=2, color=col, label=lbl)
    axes[0].set_title("LOESS — Bandwidth Comparison")
    axes[0].set_xlabel("Temperature (°C)"); axes[0].set_ylabel("Bike Count")
    axes[0].legend(fontsize=8)

    # Hour LOESS
    hour_all = bike_df["Hour"].values
    sidx_h = np.argsort(hour_all)
    xh, yh = hour_all[sidx_h], cnt_all[sidx_h]
    axes[1].scatter(xh, yh, alpha=0.1, s=5, color="gray", label="Data")
    sm_h = lowess(yh, xh, frac=0.15, return_sorted=True)
    axes[1].plot(sm_h[:, 0], sm_h[:, 1], lw=2.5, color=PALETTE["blue"], label="LOESS")
    axes[1].set_title("LOESS — Hour vs Bike Count")
    axes[1].set_xlabel("Hour of Day"); axes[1].set_ylabel("Bike Count")
    axes[1].legend()
    plt.tight_layout()
    figs["loess_static"] = fig_to_img(fig)

    # --- GAM partial dependence (top 5 features) ---
    feat_names_b = R["bike_feat_names"]
    pdp_feats = ["Hour", "Temperature", "Humidity", "Wind speed", "Solar Radiation"]
    pdp_idxs  = [feat_names_b.index(f) for f in pdp_feats if f in feat_names_b]
    fig, axes = plt.subplots(1, len(pdp_idxs), figsize=(4*len(pdp_idxs), 4))
    for ax, fi in zip(axes, pdp_idxs):
        XX = gam.generate_X_grid(term=fi)
        pdep, confi = gam.partial_dependence(term=fi, X=XX, width=0.95)
        ax.plot(XX[:, fi], pdep, color=PALETTE["blue"], lw=2)
        ax.fill_between(XX[:, fi], confi[:, 0], confi[:, 1],
                        alpha=0.2, color=PALETTE["blue"])
        ax.set_title(f"{feat_names_b[fi]}")
        ax.set_xlabel(feat_names_b[fi]); ax.set_ylabel("f(x)")
    plt.suptitle("GAM — Partial Dependence Plots (95% CI)", fontsize=13)
    plt.tight_layout()
    figs["gam_pdp"] = fig_to_img(fig)

    # --- Classification EDA ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    vc = kdd_df["label"].value_counts().head(8)
    axes[0].barh(vc.index, vc.values, color=PALETTE["blue"])
    axes[0].set_title("Top 8 Connection Labels"); axes[0].set_xlabel("Count")
    axes[0].invert_yaxis()
    bvc = kdd_df["binary_label"].value_counts()
    axes[1].pie([bvc[0], bvc[1]], labels=["Normal", "Attack"],
                autopct="%1.1f%%",
                colors=[PALETTE["green"], PALETTE["red"]],
                startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[1].set_title("Binary Class Distribution")
    plt.tight_layout()
    figs["kdd_eda1"] = fig_to_img(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    proto = kdd_df.groupby(["protocol_type", "binary_label"]).size().unstack(fill_value=0)
    proto.plot(kind="bar", ax=axes[0],
               color=[PALETTE["green"], PALETTE["red"]], edgecolor="white", width=0.6)
    axes[0].set_title("Protocol Type vs Class"); axes[0].set_xlabel("")
    axes[0].legend(["Normal", "Attack"]); axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
    top_svc = kdd_df["service"].value_counts().head(10).index
    svc_d = kdd_df[kdd_df["service"].isin(top_svc)]
    svc_g = svc_d.groupby(["service", "binary_label"]).size().unstack(fill_value=0)
    svc_g.plot(kind="bar", ax=axes[1],
               color=[PALETTE["green"], PALETTE["red"]], edgecolor="white", width=0.6)
    axes[1].set_title("Top 10 Services vs Class")
    axes[1].legend(["Normal", "Attack"])
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    figs["kdd_eda2"] = fig_to_img(fig)

    # --- Feature distributions by class ---
    num_feats_k = ["duration", "src_bytes", "dst_bytes", "hot", "count", "srv_count"]
    fig, axes3 = plt.subplots(2, 3, figsize=(14, 6))
    for ax, feat in zip(axes3.ravel(), num_feats_k):
        for lbl, col, nm in [(0, PALETTE["green"], "Normal"), (1, PALETTE["red"], "Attack")]:
            vals = np.log1p(kdd_df[kdd_df["binary_label"] == lbl][feat].clip(0, None))
            ax.hist(vals, bins=40, alpha=0.55, color=col, label=nm, density=True)
        ax.set_title(f"log(1+{feat})"); ax.legend(fontsize=7)
    plt.suptitle("Feature Distributions by Class (log scale)", fontsize=13)
    plt.tight_layout()
    figs["kdd_feat_dist"] = fig_to_img(fig)

    # --- Confusion matrices (all classifiers) ---
    all_clf_names = list(clf.keys())
    n_clf = len(all_clf_names)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, name in zip(axes.ravel(), all_clf_names):
        cm_mat = confusion_matrix(yk_te, clf[name]["pred"])
        ConfusionMatrixDisplay(cm_mat, display_labels=["Normal", "Attack"]).plot(
            ax=ax, cmap="Blues", colorbar=False)
        f1v = f1_score(yk_te, clf[name]["pred"])
        ax.set_title(f"{name}\nF1={f1v:.4f}")
    for ax in axes.ravel()[n_clf:]:
        ax.set_visible(False)
    plt.suptitle("Confusion Matrices — All Classification Models", fontsize=13)
    plt.tight_layout()
    figs["clf_conf_all"] = fig_to_img(fig)

    # --- RF Feature importances ---
    importances = rf.feature_importances_
    top_idx = np.argsort(importances)[-15:]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh([feat_names_k[i] for i in top_idx], importances[top_idx], color=PALETTE["blue"])
    ax.set_xlabel("Feature Importance (Gini)")
    ax.set_title("Random Forest — Top 15 Feature Importances")
    plt.tight_layout()
    figs["rf_importance"] = fig_to_img(fig)

    # --- GB deviance ---
    iters = np.arange(1, gb.n_estimators_ + 1)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(iters, clf["Gradient Boosting"]["train_score"],
            color=PALETTE["blue"], lw=2, label="Train Deviance")
    ax.plot(iters, clf["Gradient Boosting"]["test_deviance"],
            color=PALETTE["red"], lw=2, label="Test Deviance")
    ax.set_xlabel("Boosting Iteration"); ax.set_ylabel("Log Loss")
    ax.set_title("Gradient Boosting — Learning Curve (Train vs Test Deviance)")
    ax.legend()
    plt.tight_layout()
    figs["gb_deviance"] = fig_to_img(fig)

    # --- ROC curves (all classifiers) ---
    fig, ax = plt.subplots(figsize=(9, 6))
    palette_roc = list(plt.cm.tab10.colors)
    for i, (name, d) in enumerate(clf.items()):
        fpr, tpr, _ = roc_curve(yk_te, d["prob"])
        auc_val = sklearn_auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, color=palette_roc[i % 10],
                label=f"{name} (AUC={auc_val:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.5)")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Classification Models")
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    figs["roc_all"] = fig_to_img(fig)

    # --- Classification comparison bar ---
    cd = R["clf_df"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors_c = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(cd)))

    bars = axes[0].barh(cd["Model"], cd["F1"].astype(float), color=colors_c)
    axes[0].set_xlabel("F1 Score (higher is better)")
    axes[0].set_title("Classification Models — F1")
    axes[0].invert_yaxis()
    axes[0].set_xlim(cd["F1"].astype(float).min() * 0.97, 1.005)
    for bar, val in zip(bars, cd["F1"].astype(float)):
        axes[0].text(bar.get_width() + 0.0005,
                     bar.get_y() + bar.get_height() / 2,
                     f"{val:.4f}", va="center", fontsize=9)

    auc_vals = cd["AUC-ROC"].apply(lambda x: float(x) if x != "N/A" else 0.0)
    bars2 = axes[1].barh(cd["Model"], auc_vals, color=colors_c)
    axes[1].set_xlabel("AUC-ROC (higher is better)")
    axes[1].set_title("Classification Models — AUC-ROC")
    axes[1].invert_yaxis()
    axes[1].set_xlim(auc_vals.min() * 0.97, 1.005)
    for bar, val in zip(bars2, auc_vals):
        axes[1].text(bar.get_width() + 0.0005,
                     bar.get_y() + bar.get_height() / 2,
                     f"{val:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    figs["clf_compare"] = fig_to_img(fig)

    R["figs"] = figs
    R["best_poly_deg"] = best_d
    R["pdp_feats"] = [feat_names_b[fi] for fi in pdp_idxs]
    R["loess_xs"] = xs_all
    R["loess_ys"] = ys_all
    R["bike_temp_data"] = (bike_df["Temperature"].values, bike_df["Rented Bike Count"].values)

    # ── Save to disk so next launch is instant ────────────────────────────
    try:
        _save_to_disk(R)
    except Exception as e:
        pass  # disk save failure is non-fatal

    return R


# ─────────────────────────────────────────────────────────────────────────────
#  KICK OFF PRE-COMPUTATION  (runs on every app start)
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("🔄 Pre-computing all models (first launch only — will be instant afterwards)…"):
    R = precompute_all()

F = R["figs"]   # shorthand to pre-generated figures


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 📊 AML Project 4")
st.sidebar.markdown("*CSCI-6767 · Spring 2026*")
st.sidebar.markdown("---")

PAGES = {
    "🏠  Overview":                "overview",
    "📂  Datasets":                "datasets",
    "📊  Regression EDA":          "reg_eda",
    "📈  Non-Linear Models":       "nonlinear",
    "🔍  Classification EDA":      "clf_eda",
    "🌲  Tree-Based Methods":      "trees",
    "🔷  Support Vector Machines": "svm",
    "🏆  Final Comparison":        "comparison",
    "📝  Conclusions":             "conclusions",
}

page = PAGES[st.sidebar.radio("Navigate to:", list(PAGES.keys()),
                               label_visibility="collapsed")]

st.sidebar.markdown("---")
st.sidebar.markdown("**Team**  \nGabil Gurbanov  \nHamida Hagverdiyeva")
st.sidebar.markdown("**Course:** CSCI-6767")


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if page == "overview":
    st.title("Non-Linearity, Tree-Based Classification & SVM")
    st.markdown("**CSCI-6767 Applied Machine Learning & Data Analytics — Spring 2026**  \n"
                "**Team:** Gabil Gurbanov · Hamida Hagverdiyeva")
    st.divider()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### Project Tasks")
        st.markdown("""
| # | Task | Weight |
|---|------|:------:|
| 1 | Non-Linear Models (5 methods) | 30% |
| 2 | Tree-Based Methods (3 methods) | 30% |
| 3 | SVM (4 kernels) | 20% |
| 4 | Model Comparison & Analysis | 20% |
| ★ | Web Application (this app) | +20% |
        """)
    with c2:
        st.markdown("### Strategy")
        st.markdown("""
**Two-dataset approach:**

🔵 **Dataset 1 (Regression)**
→ Seoul Bike Sharing Demand
→ Non-linear models (Task 1)
→ Continuous target ideal for GAMs & splines

🔴 **Dataset 2 (Classification)**
→ KDD Cup 1999 Network Intrusion
→ Tree methods (Task 2) + SVM (Task 3)
→ Same dataset = direct comparison
        """)
    with c3:
        st.markdown("### Quick Results")
        reg_df = R["reg_df"]
        best_reg = reg_df.iloc[0]
        clf_df = R["clf_df"]
        best_clf = clf_df.iloc[0]
        st.metric("Best Regression Model", best_reg["Model"],
                  f"R² = {best_reg['R²']:.4f}")
        st.metric("Best Classification Model", best_clf["Model"],
                  f"F1 = {best_clf['F1']:.4f}")
        st.metric("Regression Dataset", "Seoul Bike", "8,760 records")
        st.metric("Classification Dataset", "KDD Cup 1999", "494,021 records")

    st.divider()
    st.markdown("### All Methods at a Glance")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Task 1 — Non-Linear Models (Regression)**")
        for m in ["Polynomial Regression (deg 2–4)", "Step Functions", "B-Splines + Natural Cubic Splines", "LOESS Local Regression", "GAMs (Generalized Additive Models)"]:
            st.markdown(f"• {m}")
    with c2:
        st.markdown("**Task 2 — Tree-Based Methods (Classification)**")
        for m in ["Bagging (100 Decision Trees)", "Random Forest (100 trees + feature sampling)", "Gradient Boosting (200 trees, staged)"]:
            st.markdown(f"• {m}")
    with c3:
        st.markdown("**Task 3 — SVM Kernels (Classification)**")
        for m in ["Linear kernel (C=1.0)", "RBF kernel (C=10, γ=scale)", "Polynomial kernel (degree=3)", "Sigmoid kernel"]:
            st.markdown(f"• {m}")


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE: DATASETS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "datasets":
    st.title("📂 Datasets")
    st.markdown("Detailed information about both datasets used in this project.")
    st.divider()

    tab_bike, tab_kdd = st.tabs(["🚲 Seoul Bike Sharing Demand", "🌐 KDD Cup 1999 Network Intrusion"])

    with tab_bike:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("""
### Seoul Bike Sharing Demand
**Used for:** Task 1 — Non-Linear Regression Models

**Source:** UCI Machine Learning Repository
**Fetch:** `fetch_ucirepo(id=560)` (automatic download)

**Description:**
Contains the count of public bicycles rented per hour in the Seoul Bike Sharing System.
Recorded from **December 2017 to November 2018** — 8,760 hourly records (365 days × 24 hours).
The dataset is clean with no missing values, making it ideal for algorithm evaluation.

**Why this dataset?**
- Strong **non-linear** relationships (temperature-demand, hourly bimodal pattern)
- Seasonal effects that step functions and GAMs can capture
- Mix of continuous (temperature, humidity) and ordinal (hour, season) predictors
- Well-known benchmark: prior published results for comparison
            """)
        with c2:
            bike_df = R["bike_df"]
            st.metric("Rows", f"{len(bike_df):,}")
            st.metric("Features", f"{bike_df.shape[1]-1}")
            st.metric("Target", "Rented Bike Count")
            st.metric("Missing Values", "0")

        st.markdown("**Feature Descriptions:**")
        feat_table = pd.DataFrame([
            ["Date", "Categorical", "Date in DD/MM/YYYY format"],
            ["Rented Bike Count", "Target (int)", "Number of bikes rented per hour"],
            ["Hour", "Ordinal (0–23)", "Hour of the day"],
            ["Temperature", "Continuous (°C)", "Temperature in Celsius"],
            ["Humidity", "Continuous (%)", "Relative humidity percentage"],
            ["Wind speed", "Continuous (m/s)", "Wind speed"],
            ["Visibility", "Continuous (10m)", "Visibility distance"],
            ["Dew point temperature", "Continuous (°C)", "Dew point temperature"],
            ["Solar Radiation", "Continuous (MJ/m²)", "Solar radiation"],
            ["Rainfall", "Continuous (mm)", "Rainfall amount"],
            ["Snowfall", "Continuous (cm)", "Snowfall amount"],
            ["Seasons", "Categorical (1–4)", "Season: 1=Spring, 2=Summer, 3=Autumn, 4=Winter"],
            ["Holiday", "Binary", "0=No holiday, 1=Public holiday"],
            ["Functioning Day", "Binary", "0=Non-operational, 1=Functional day"],
        ], columns=["Feature", "Type", "Description"])
        st.dataframe(feat_table, use_container_width=True, hide_index=True)

        st.markdown("**Sample Data:**")
        st.dataframe(bike_df.head(5), use_container_width=True)

    with tab_kdd:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("""
### KDD Cup 1999 Network Intrusion Detection
**Used for:** Task 2 (Tree Methods) + Task 3 (SVM)

**Source:** `sklearn.datasets.fetch_kddcup99` (built-in, no download required)
**Subset used:** 10% sample (`percent10=True`) → 494,021 records

**Description:**
A classic network intrusion detection dataset originally prepared for the KDD Cup 1999 competition.
Simulates a military network environment with both normal connections and various attack types
(DoS, R2L, U2R, Probe). This project uses a **binary classification** formulation:
`normal` vs. `attack` (all other labels).

**Why this dataset?**
- Rich mix of **categorical** and **continuous** features (41 features total)
- Challenges both tree ensembles (feature interactions) and SVMs (kernel selection)
- Large enough to demonstrate scalability differences between model families
- Well-studied benchmark with expected near-perfect performance, allowing focus on model comparison

**Attack types present:**
- **DoS** (Denial of Service): smurf, neptune, back, teardrop, pod, land
- **Probe**: satan, ipsweep, nmap, portsweep
- **R2L** (Remote to Local): ftp_write, guess_passwd, imap, phf, multihop, spy, warezclient, warezmaster
- **U2R** (User to Root): buffer_overflow, loadmodule, rootkit, perl
            """)
        with c2:
            kdd_df = R["kdd_df"]
            st.metric("Total Records", f"{len(kdd_df):,}")
            st.metric("Features", "41")
            st.metric("Normal", f"{(kdd_df['binary_label']==0).sum():,}")
            st.metric("Attack", f"{(kdd_df['binary_label']==1).sum():,}")
            attack_rate = kdd_df["binary_label"].mean() * 100
            st.metric("Attack Rate", f"{attack_rate:.1f}%")

        st.markdown("**Key Feature Categories:**")
        feat_cat_table = pd.DataFrame([
            ["duration", "Continuous", "Length of connection in seconds"],
            ["protocol_type", "Categorical", "TCP, UDP, ICMP"],
            ["service", "Categorical", "Network service (http, ftp, telnet, …)"],
            ["flag", "Categorical", "Normal or error status of connection"],
            ["src_bytes", "Continuous", "Bytes from source to destination"],
            ["dst_bytes", "Continuous", "Bytes from destination to source"],
            ["logged_in", "Binary", "1 if successfully logged in"],
            ["count", "Continuous", "Connections to same host in past 2 sec"],
            ["srv_count", "Continuous", "Connections to same service in past 2 sec"],
            ["serror_rate", "Continuous", "% SYN error connections (same host)"],
        ], columns=["Feature", "Type", "Description"])
        st.dataframe(feat_cat_table, use_container_width=True, hide_index=True)

        st.markdown("""
**Preprocessing steps applied:**
1. Decode byte-string labels (Python 3 artifact from sklearn)
2. Binary label: `normal.` → 0, all attack types → 1
3. Label-encode three categorical columns: `protocol_type`, `service`, `flag`
4. Standard-scale all features (required for SVM)
5. For SVM: sub-sample to **15,000** training records (O(n²) complexity constraint)
        """)


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE: REGRESSION EDA
# ─────────────────────────────────────────────────────────────────────────────
elif page == "reg_eda":
    st.title("📊 Regression EDA — Seoul Bike Sharing Demand")
    st.markdown("Exploratory analysis of the regression dataset before model fitting.")
    st.divider()

    bike_df = R["bike_df"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", f"{len(bike_df):,}")
    c2.metric("Features", f"{bike_df.shape[1]-1}")
    c3.metric("Max Hourly Demand", f"{bike_df['Rented Bike Count'].max():,}")
    c4.metric("Avg Hourly Demand", f"{bike_df['Rented Bike Count'].mean():.0f}")

    st.subheader("Distribution, Hourly Pattern & Temperature Effect")
    st.image(F["bike_eda1"], use_container_width=True)

    st.subheader("Seasonal Variation & Humidity")
    st.image(F["bike_eda2"], use_container_width=True)

    st.subheader("Feature Correlation Matrix")
    st.image(F["bike_corr"], use_container_width=True)

    st.info(
        "**Key observations motivating non-linear models:**  \n"
        "• **Hour**: Bimodal demand (8am rush + 6pm rush) — linear models cannot capture this.  \n"
        "• **Temperature**: Non-monotonic sweet spot around 20–25°C; cold and very hot reduce demand.  \n"
        "• **Seasons**: Autumn peak > Summer > Spring > Winter — strong seasonal structure.  \n"
        "• **Humidity**: Mild negative correlation; high humidity discourages cycling."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE: NON-LINEAR MODELS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "nonlinear":
    st.title("📈 Non-Linear Models — Regression")
    st.markdown("All five non-linear methods applied to the **Seoul Bike Sharing Demand** dataset.")
    st.divider()

    # Summary table + bar chart
    st.subheader("Performance Comparison")
    c_t, c_c = st.columns([1, 1])
    with c_t:
        rd = R["reg_df"].copy()
        rd.index = range(1, len(rd)+1)
        styled = (
            rd.style
            .apply(highlight_min, subset=["RMSE", "MAE"])
            .apply(highlight_max, subset=["R²"])
            .format({"RMSE": "{:.1f}", "MAE": "{:.1f}", "R²": "{:.4f}"})
        )
        st.dataframe(styled, use_container_width=True, height=300)
    with c_c:
        st.image(F["reg_compare"], use_container_width=True)

    st.divider()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📐 Polynomial", "📏 Step Functions", "🌀 Splines", "〰 LOESS", "📉 GAM",
        "🎯 Actual vs Predicted"
    ])

    # ── Polynomial ────────────────────────────────────────────────────────
    with tab1:
        st.markdown("### Polynomial Regression")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("""
**Concept:** Extend linear regression by adding polynomial terms of the input features.

A degree-$d$ polynomial model fits:
            """)
            st.latex(r"\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_1^2 + \cdots + \beta_d x_1^d + \ldots")
            st.markdown("""
**Implementation:**
- `PolynomialFeatures(degree=d)` generates all degree-$d$ interaction terms
- `StandardScaler` normalises features first (essential — raw polynomial features span huge ranges)
- `RidgeCV` auto-selects regularisation $\\lambda$ via cross-validation

**Trade-off:** Higher degree → more expressive but more features (119 / 679 / 3,059 for d=2/3/4)
and more regularisation needed. Degree 3 delivers the best RMSE here.
            """)
        with c2:
            reg = R["reg"]
            degrees = [2, 3, 4]
            rows = []
            for d in degrees:
                k = f"Polynomial deg={d}"
                r2v = r2_score(R["yb_te"], reg[k]["pred"])
                rmse = float(np.sqrt(mean_squared_error(R["yb_te"], reg[k]["pred"])))
                rows.append({"Degree": d, "RMSE": round(rmse,1), "R²": round(r2v,4),
                             "Features": reg[k]["n_feats"], "Best α": reg[k]["alpha"]})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.image(F["poly_compare"], use_container_width=True)
        bd = R["best_poly_deg"]
        st.success(f"**Best degree:** {bd}  ·  RMSE = {R['reg_df'][R['reg_df']['Model'].str.contains('Polynomial')]['RMSE'].min():.1f}")

    # ── Step Functions ────────────────────────────────────────────────────
    with tab2:
        st.markdown("### Step Functions (Piecewise Constants)")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("""
**Concept:** Discretise a continuous predictor into $K$ intervals (bins) and fit a
constant within each bin. Equivalent to a piecewise-constant function.
            """)
            st.latex(r"f(X) = \sum_{j=1}^{K} c_j \cdot \mathbf{1}(X \in [t_{j-1}, t_j))")
            st.markdown("""
**Implementation:**
- `pd.cut(x, bins=K)` partitions each feature into equal-width intervals
- Bins are one-hot encoded → sparse binary matrix
- `Ridge` regression learns a value per bin

**Bins used:**
| Feature | Bins |
|---------|------|
| Temperature | 6 |
| Hour | 8 |
| Humidity | 5 |

**Limitation:** Discontinuous jumps at bin boundaries — smoother methods (splines, GAMs) generally outperform.
            """)
        with c2:
            r2_step = r2_score(R["yb_te"], R["reg"]["Step Functions"]["pred"])
            rmse_step = float(np.sqrt(mean_squared_error(R["yb_te"], R["reg"]["Step Functions"]["pred"])))
            st.metric("RMSE", f"{rmse_step:.1f}")
            st.metric("R²", f"{r2_step:.4f}")
        st.image(F["step_fit"], use_container_width=True)

    # ── Splines ───────────────────────────────────────────────────────────
    with tab3:
        st.markdown("### Regression Splines (B-Splines & Natural Cubic Splines)")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("**B-Splines:** piecewise polynomials joined smoothly at knots:")
            st.latex(r"f(x) = \sum_{j=1}^{K+d} \beta_j B_{j,d}(x)")
            st.markdown("""
where $B_{j,d}$ are B-spline basis functions of degree $d$.

**Natural Cubic Splines (NCS):** cubic splines (d=3) with the additional constraint
of **linearity outside the boundary knots**, reducing extrapolation artefacts:
            """)
            st.latex(r"f(x) = \beta_0 + \beta_1 x + \sum_{k=1}^{K-2} \theta_k d_k(x)")
            st.markdown("""
**Implementation:**
- B-Splines: `sklearn.SplineTransformer(n_knots=6, degree=3)` + `RidgeCV`
- NCS: `patsy.dmatrix('cr(x, df=5)')` + `RidgeCV`
            """)
        with c2:
            for name in ["B-Splines", "Natural Cubic Splines"]:
                r2v = r2_score(R["yb_te"], R["reg"][name]["pred"])
                rmse = float(np.sqrt(mean_squared_error(R["yb_te"], R["reg"][name]["pred"])))
                st.metric(f"{name} — RMSE", f"{rmse:.1f}", f"R²={r2v:.4f}")
        st.image(F["spline_1d"], use_container_width=True)

    # ── LOESS ─────────────────────────────────────────────────────────────
    with tab4:
        st.markdown("### Local Regression (LOESS / LOWESS)")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("""
**Concept:** LOESS (*Locally Estimated Scatterplot Smoothing*) fits a **local polynomial**
to each neighbourhood of points, weighted by distance from the focal point.
            """)
            st.latex(r"\hat{f}(x_0) = \arg\min_{\beta} \sum_{i=1}^n K\!\left(\frac{x_i - x_0}{h}\right)(y_i - \beta_0 - \beta_1(x_i-x_0))^2")
            st.markdown("""
where $K(\\cdot)$ is a kernel (tri-cube weight) and $h$ is the bandwidth.

**Bandwidth parameter `frac`:** fraction of data used in each local fit.
- Small `frac` → flexible, follows noise closely
- Large `frac` → smooth trend, more bias

**Important note:** LOESS produces no closed-form global model and operates in 1D.
It is a **diagnostic / visualization tool** — not included in the RMSE comparison table.
            """)
        with c2:
            st.markdown("**Interactive bandwidth selector:**")
            frac_val = st.slider("LOESS bandwidth (frac)", 0.03, 0.40, 0.10, 0.01, key="loess_sl")
            xs_l, ys_l = R["loess_xs"], R["loess_ys"]
            sm = lowess(ys_l, xs_l, frac=frac_val, return_sorted=True)
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.scatter(xs_l, ys_l, alpha=0.1, s=5, color="gray")
            ax.plot(sm[:, 0], sm[:, 1], color=PALETTE["blue"], lw=2.5,
                    label=f"LOESS frac={frac_val:.2f}")
            ax.set_xlabel("Temperature (°C)"); ax.set_ylabel("Bike Count")
            ax.set_title("Interactive LOESS Fit")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
        st.image(F["loess_static"], use_container_width=True)

    # ── GAM ───────────────────────────────────────────────────────────────
    with tab5:
        st.markdown("### Generalized Additive Models (GAMs)")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("""
**Concept:** GAMs extend linear regression by replacing each linear term with an
arbitrary smooth function, while retaining additivity for interpretability:
            """)
            st.latex(r"g(\mu) = \beta_0 + f_1(x_1) + f_2(x_2) + \cdots + f_p(x_p)")
            st.markdown("""
where each $f_j$ is a smooth spline estimated via penalised regression:

$$\\text{minimize} \\sum_i (y_i - \\hat{f}(x_i))^2 + \\lambda_j \\int [f_j''(t)]^2 \\, dt$$

The smoothness penalty $\\lambda_j$ is tuned by grid search.

**Key advantage:** Partial dependence plots reveal each feature's **marginal effect**
while holding others constant — combining performance with interpretability.

**Implementation:** `pygam.LinearGAM` with `s(i)` spline terms for all features.
            """)
        with c2:
            gam_r2 = r2_score(R["yb_te"], R["reg"]["GAM (LinearGAM)"]["pred"])
            gam_rmse = float(np.sqrt(mean_squared_error(R["yb_te"], R["reg"]["GAM (LinearGAM)"]["pred"])))
            st.metric("RMSE", f"{gam_rmse:.1f}")
            st.metric("R²", f"{gam_r2:.4f}")
            st.success("**Best regression model** — highest R², lowest RMSE")

        st.markdown("#### Partial Dependence Plots (95% CI)")
        st.markdown("Each plot shows one feature's smooth effect on bike demand, "
                    "with all other features held at their average.")
        st.image(F["gam_pdp"], use_container_width=True)

        # Interactive single-feature PDP
        st.markdown("#### Interactive Partial Dependence Explorer")
        feat_names_b = R["bike_feat_names"]
        pdp_options = [f for f in ["Hour", "Temperature", "Humidity",
                                    "Wind speed", "Solar Radiation",
                                    "Dew point temperature"]
                       if f in feat_names_b]
        sel_feat = st.selectbox("Select feature:", pdp_options)
        fi = feat_names_b.index(sel_feat)
        gam_obj = R["reg"]["GAM (LinearGAM)"]["gam_obj"]
        XX = gam_obj.generate_X_grid(term=fi)
        pdep, confi = gam_obj.partial_dependence(term=fi, X=XX, width=0.95)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(XX[:, fi], pdep, color=PALETTE["blue"], lw=2.5)
        ax.fill_between(XX[:, fi], confi[:, 0], confi[:, 1],
                        alpha=0.2, color=PALETTE["blue"], label="95% CI")
        ax.set_xlabel(sel_feat); ax.set_ylabel("Marginal effect f(x)")
        ax.set_title(f"GAM Partial Dependence — {sel_feat}")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── Actual vs Predicted ───────────────────────────────────────────────
    with tab6:
        st.markdown("### Actual vs Predicted & Residuals")
        st.image(F["reg_avp"], use_container_width=True)
        st.image(F["reg_resid"], use_container_width=True)

        # Interactive single-model selector
        st.markdown("#### Inspect any model:")
        sel_model = st.selectbox("Choose model:", list(R["reg"].keys()))
        ypred = R["reg"][sel_model]["pred"]
        r2v  = r2_score(R["yb_te"], ypred)
        rmse = float(np.sqrt(mean_squared_error(R["yb_te"], ypred)))
        c1, c2, c3 = st.columns(3)
        c1.metric("R²", f"{r2v:.4f}"); c2.metric("RMSE", f"{rmse:.1f}")
        c3.metric("MAE", f"{mean_absolute_error(R['yb_te'], ypred):.1f}")

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        axes[0].scatter(R["yb_te"], ypred, alpha=0.3, s=6, color=PALETTE["blue"])
        lims = [0, max(float(R["yb_te"].max()), float(ypred.max()))]
        axes[0].plot(lims, lims, "r--", lw=1.5, label="Perfect fit")
        axes[0].set_title(f"Actual vs Predicted — {sel_model}")
        axes[0].set_xlabel("Actual"); axes[0].set_ylabel("Predicted")
        axes[0].legend()
        resid = R["yb_te"] - ypred
        axes[1].scatter(ypred, resid, alpha=0.3, s=6, color=PALETTE["orange"])
        axes[1].axhline(0, color="black", lw=1.5, linestyle="--")
        axes[1].set_title(f"Residuals — {sel_model}")
        axes[1].set_xlabel("Fitted"); axes[1].set_ylabel("Residuals")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE: CLASSIFICATION EDA
# ─────────────────────────────────────────────────────────────────────────────
elif page == "clf_eda":
    st.title("🔍 Classification EDA — KDD Cup 1999 Network Intrusion")
    st.divider()
    kdd_df = R["kdd_df"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Records", f"{len(kdd_df):,}")
    c2.metric("Features", "41")
    c3.metric("Normal", f"{(kdd_df['binary_label']==0).sum():,}")
    c4.metric("Attack", f"{(kdd_df['binary_label']==1).sum():,}")
    c5.metric("Attack Rate", f"{kdd_df['binary_label'].mean()*100:.1f}%")

    st.subheader("Label & Class Distribution")
    st.image(F["kdd_eda1"], use_container_width=True)

    st.subheader("Protocol & Service Analysis")
    st.image(F["kdd_eda2"], use_container_width=True)

    st.subheader("Feature Distributions by Class")
    st.image(F["kdd_feat_dist"], use_container_width=True)

    st.info(
        "**Key observations:**  \n"
        "• `src_bytes` and `dst_bytes` are the most discriminative features — DoS attacks generate "
        "massive byte counts. Tree models pick this up automatically via feature importance.  \n"
        "• `protocol_type=icmp` is almost exclusively associated with `smurf` DoS attacks.  \n"
        "• The dataset is highly separable, explaining why all models achieve >95% accuracy."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE: TREE-BASED METHODS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "trees":
    st.title("🌲 Tree-Based Methods — Classification")
    st.markdown("Applied to **KDD Cup 1999 Network Intrusion** — Binary: Normal vs Attack")
    st.divider()

    tab1, tab2, tab3 = st.tabs(["🎒 Bagging", "🌳 Random Forest", "⚡ Gradient Boosting"])

    def clf_metrics_row(name, d, yk_te):
        pred, prob = d["pred"], d["prob"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy",  f"{accuracy_score(yk_te, pred):.4f}")
        c2.metric("F1 Score",  f"{f1_score(yk_te, pred):.4f}")
        c3.metric("Precision", f"{precision_score(yk_te, pred, zero_division=0):.4f}")
        c4.metric("AUC-ROC",   f"{roc_auc_score(yk_te, prob):.4f}")

    yk_te = R["yk_te"]

    with tab1:
        st.markdown("### Bagging — Bootstrap Aggregating")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("""
**Concept:** Train $B$ independent base classifiers on **bootstrap samples** of the
training data, then aggregate via majority vote (classification) or averaging (regression).
            """)
            st.latex(r"\hat{f}(x) = \frac{1}{B} \sum_{b=1}^{B} f_b(x)")
            st.markdown("""
where each $f_b$ is trained on a bootstrap sample $Z^{*b}$ drawn with replacement.

**Variance reduction:** averaging uncorrelated weak learners reduces variance while
keeping bias approximately the same.

**OOB (Out-of-Bag) estimate:** ~37% of samples are left out of each bootstrap — these
form a free validation set without needing a separate hold-out.

**Settings used:**
- Base estimator: `DecisionTreeClassifier`
- $B = 100$ estimators
- `oob_score=True`
            """)
        with c2:
            d = R["clf"]["Bagging"]
            clf_metrics_row("Bagging", d, yk_te)
            st.caption(f"OOB Score: {d['oob']:.4f}")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        cm = confusion_matrix(yk_te, R["clf"]["Bagging"]["pred"])
        ConfusionMatrixDisplay(cm, display_labels=["Normal", "Attack"]).plot(
            ax=axes[0], cmap="Blues", colorbar=False)
        axes[0].set_title("Confusion Matrix — Bagging")
        fpr, tpr, _ = roc_curve(yk_te, R["clf"]["Bagging"]["prob"])
        axes[1].plot(fpr, tpr, color=PALETTE["blue"], lw=2,
                     label=f"AUC={sklearn_auc(fpr,tpr):.4f}")
        axes[1].plot([0,1],[0,1],"k--",lw=1)
        axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
        axes[1].set_title("ROC Curve — Bagging"); axes[1].legend()
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    with tab2:
        st.markdown("### Random Forest")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("""
**Concept:** Random Forests extend bagging with an additional **random feature subsampling**
at each split, decorrelating the trees and reducing variance further:
            """)
            st.latex(r"m = \lfloor \sqrt{p} \rfloor \text{ features sampled per split}")
            st.markdown("""
Each tree is grown to maximum depth without pruning, then:
$$\\hat{f}_{RF}(x) = \\frac{1}{B}\\sum_{b=1}^B T_b(x; \\Theta_b)$$

**Why better than Bagging?**
If a few features dominate (like `src_bytes`), all bagged trees will split on them first
→ correlated predictions → limited variance reduction. Random feature sampling forces
trees to explore other features → lower correlation → better ensemble.

**Settings:** 100 trees, `oob_score=True`, `n_jobs=-1`
            """)
        with c2:
            d = R["clf"]["Random Forest"]
            clf_metrics_row("Random Forest", d, yk_te)
            st.caption(f"OOB Score: {d['oob']:.4f}")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        cm = confusion_matrix(yk_te, R["clf"]["Random Forest"]["pred"])
        ConfusionMatrixDisplay(cm, display_labels=["Normal", "Attack"]).plot(
            ax=axes[0], cmap="Blues", colorbar=False)
        axes[0].set_title("Confusion Matrix — Random Forest")
        fpr, tpr, _ = roc_curve(yk_te, R["clf"]["Random Forest"]["prob"])
        axes[1].plot(fpr, tpr, color=PALETTE["red"], lw=2,
                     label=f"AUC={sklearn_auc(fpr,tpr):.4f}")
        axes[1].plot([0,1],[0,1],"k--",lw=1)
        axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
        axes[1].set_title("ROC Curve — Random Forest"); axes[1].legend()
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

        st.subheader("Feature Importances (Top 15)")
        st.image(F["rf_importance"], use_container_width=True)

    with tab3:
        st.markdown("### Gradient Boosting")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("""
**Concept:** Build an ensemble sequentially where each tree corrects the **residuals**
of the previous model (gradient descent in function space):
            """)
            st.latex(r"F_m(x) = F_{m-1}(x) + \nu \cdot h_m(x)")
            st.markdown("""
where $h_m$ minimises:

$$\\sum_i \\left[- \\frac{\\partial L(y_i, F_{m-1}(x_i))}{\\partial F_{m-1}(x_i)}\\right]^2$$

**Key hyperparameters:**
| Parameter | Value | Role |
|-----------|-------|------|
| `n_estimators` | 200 | Number of boosting rounds |
| `learning_rate` | 0.1 | Shrinkage — slows overfitting |
| `max_depth` | 4 | Limits tree complexity |
| `subsample` | 0.8 | Stochastic GB — samples 80% per round |
            """)
        with c2:
            d = R["clf"]["Gradient Boosting"]
            clf_metrics_row("Gradient Boosting", d, yk_te)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        cm = confusion_matrix(yk_te, R["clf"]["Gradient Boosting"]["pred"])
        ConfusionMatrixDisplay(cm, display_labels=["Normal", "Attack"]).plot(
            ax=axes[0], cmap="Blues", colorbar=False)
        axes[0].set_title("Confusion Matrix — Gradient Boosting")
        fpr, tpr, _ = roc_curve(yk_te, R["clf"]["Gradient Boosting"]["prob"])
        axes[1].plot(fpr, tpr, color=PALETTE["green"], lw=2,
                     label=f"AUC={sklearn_auc(fpr,tpr):.4f}")
        axes[1].plot([0,1],[0,1],"k--",lw=1)
        axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
        axes[1].set_title("ROC Curve — Gradient Boosting"); axes[1].legend()
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

        st.subheader("Learning Curve — Train vs Test Deviance")
        st.image(F["gb_deviance"], use_container_width=True)
        st.caption("The gap between train and test deviance closes quickly — model generalises well without overfitting.")


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE: SVM
# ─────────────────────────────────────────────────────────────────────────────
elif page == "svm":
    st.title("🔷 Support Vector Machines — Kernel Comparison")
    st.markdown("Applied to **KDD Cup 1999** · Trained on a **15,000-sample subset** (SVM O(n²) constraint)")
    st.divider()

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("""
### What is an SVM?

A Support Vector Machine finds the **maximum-margin hyperplane** separating classes.
The **kernel trick** implicitly maps data to a higher-dimensional space where
a linear boundary may exist, without computing the mapping explicitly:
        """)
        st.latex(r"f(x) = \text{sign}\!\left(\sum_{i \in \text{SVs}} \alpha_i y_i K(x_i, x) + b\right)")
        st.markdown("The decision is determined only by **support vectors** — the training points closest to the boundary.")
    with c2:
        st.markdown("""
### Kernel Functions

| Kernel | Formula | Hyperparams used |
|--------|---------|-----------------|
| **Linear** | $K(x,z) = x^\\top z$ | $C=1$ |
| **RBF** | $K(x,z) = e^{-\\gamma\\|x-z\\|^2}$ | $C=10$, $\\gamma$=scale |
| **Polynomial** | $K(x,z) = (\\gamma x^\\top z + 1)^d$ | $d=3$, $C=1$ |
| **Sigmoid** | $K(x,z) = \\tanh(\\gamma x^\\top z + c)$ | $C=1$ |

The **regularisation parameter $C$** controls the trade-off between margin width and
training error (higher $C$ = tighter fit, smaller margin).
        """)

    st.divider()

    # Interactive kernel selector
    kernel_select = st.selectbox("Select kernel to inspect:", ["Linear", "RBF", "Polynomial", "Sigmoid"])
    kname_full = f"SVM — {kernel_select}"
    d = R["clf"][kname_full]
    yk_te = R["yk_te"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",  f"{accuracy_score(yk_te, d['pred']):.4f}")
    c2.metric("F1 Score",  f"{f1_score(yk_te, d['pred']):.4f}")
    c3.metric("Precision", f"{precision_score(yk_te, d['pred'], zero_division=0):.4f}")
    c4.metric("AUC-ROC",   f"{roc_auc_score(yk_te, d['prob']):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    cm = confusion_matrix(yk_te, d["pred"])
    ConfusionMatrixDisplay(cm, display_labels=["Normal", "Attack"]).plot(
        ax=axes[0], cmap="Blues", colorbar=False)
    axes[0].set_title(f"Confusion Matrix — SVM {kernel_select}")
    fpr, tpr, _ = roc_curve(yk_te, d["prob"])
    axes[1].plot(fpr, tpr, color=PALETTE["blue"], lw=2, label=f"AUC={sklearn_auc(fpr,tpr):.4f}")
    axes[1].plot([0,1],[0,1],"k--",lw=1)
    axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
    axes[1].set_title(f"ROC Curve — SVM {kernel_select}"); axes[1].legend()
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    st.divider()
    st.subheader("All Kernels — Side-by-Side")
    st.image(F["clf_conf_all"], use_container_width=True)

    # ROC all SVM kernels
    st.subheader("ROC Curves — SVM Kernels")
    fig, ax = plt.subplots(figsize=(8, 5))
    svm_palette = [PALETTE["blue"], PALETTE["red"], PALETTE["green"], PALETTE["orange"]]
    for (kn, kd), col in zip(
        [(n, R["clf"][n]) for n in ["SVM — Linear","SVM — RBF","SVM — Polynomial","SVM — Sigmoid"]],
        svm_palette
    ):
        fpr, tpr, _ = roc_curve(yk_te, kd["prob"])
        ax.plot(fpr, tpr, lw=2, color=col, label=f"{kn} (AUC={sklearn_auc(fpr,tpr):.4f})")
    ax.plot([0,1],[0,1],"k--",lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All SVM Kernels")
    ax.legend()
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    st.info(
        "**Key insight:** RBF kernel outperforms all others because the network intrusion "
        "decision boundary is non-linear in the feature space. Sigmoid kernel underperforms "
        "(AUC ≈ 0.89) because tabular data rarely satisfies the assumptions underlying "
        "the sigmoid mapping. All SVMs are trained on a 15k-sample subset, while tree models "
        "use the full training set — this explains the performance gap."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE: FINAL COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
elif page == "comparison":
    st.title("🏆 Final Model Comparison")
    st.divider()

    tab_r, tab_c, tab_roc = st.tabs(["📈 Regression", "🔍 Classification", "📉 ROC Curves"])

    with tab_r:
        st.subheader("Regression — Seoul Bike Sharing Demand")
        rd = R["reg_df"].copy()
        rd.index = range(1, len(rd)+1)
        st.dataframe(
            rd.style
              .apply(highlight_min, subset=["RMSE","MAE"])
              .apply(highlight_max, subset=["R²"])
              .format({"RMSE":"{:.1f}","MAE":"{:.1f}","R²":"{:.4f}"}),
            use_container_width=True, height=320
        )
        st.image(F["reg_compare"], use_container_width=True)
        st.image(F["reg_avp"],     use_container_width=True)
        st.image(F["reg_resid"],   use_container_width=True)

    with tab_c:
        st.subheader("Classification — KDD Cup 1999")
        cd = R["clf_df"].copy()
        cd.index = range(1, len(cd)+1)
        st.dataframe(
            cd.style
              .apply(highlight_max, subset=["Accuracy","F1","Precision","Recall"])
              .format({"Accuracy":"{:.4f}","Precision":"{:.4f}",
                       "Recall":"{:.4f}","F1":"{:.4f}"}),
            use_container_width=True, height=280
        )
        st.image(F["clf_compare"],  use_container_width=True)
        st.image(F["clf_conf_all"], use_container_width=True)

    with tab_roc:
        st.subheader("ROC Curves — All Classification Models")
        st.markdown("""
The ROC (Receiver Operating Characteristic) curve plots **True Positive Rate** (Recall)
against **False Positive Rate** at all classification thresholds.

$$\\text{AUC} = \\int_0^1 TPR(FPR) \\, d(FPR) \\qquad \\text{(perfect classifier: AUC=1.0)}$$

All tree-based models achieve AUC ≈ 1.0 — the KDD Cup 1999 dataset has very strong
discriminative features. SVM sigmoid is the outlier (AUC ≈ 0.89).
        """)
        st.image(F["roc_all"], use_container_width=True)

        # Show AUC values table
        yk_te = R["yk_te"]
        auc_rows = []
        for name, d in R["clf"].items():
            fpr, tpr, _ = roc_curve(yk_te, d["prob"])
            auc_rows.append({"Model": name, "AUC-ROC": round(sklearn_auc(fpr, tpr), 5)})
        auc_df = pd.DataFrame(auc_rows).sort_values("AUC-ROC", ascending=False).reset_index(drop=True)
        auc_df.index = auc_df.index + 1
        st.dataframe(
            auc_df.style.apply(highlight_max, subset=["AUC-ROC"])
                        .format({"AUC-ROC": "{:.5f}"}),
            use_container_width=True, height=270
        )


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE: CONCLUSIONS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "conclusions":
    st.title("📝 Conclusions")
    st.divider()

    tab1, tab2, tab3 = st.tabs(["Regression Findings", "Classification Findings", "Team Contributions"])

    with tab1:
        st.subheader("Regression — Non-Linear Models on Seoul Bike Sharing")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
**Finding 1 — Non-linearity is essential**
The baseline linear model (R²=0.53) is significantly outperformed by every non-linear
method, confirming that bike demand is driven by complex, non-linear patterns:
bimodal hourly rhythm and a temperature sweet-spot around 20–25°C.

**Finding 2 — GAM is the best model**
`LinearGAM` achieves the highest R² and lowest RMSE by fitting independent spline
functions per feature with automatically tuned regularisation. Its partial dependence
plots make the model interpretable — a key advantage for reporting.

**Finding 3 — Polynomial degree must be regularised**
Without `StandardScaler` + `RidgeCV`, degree-4 polynomials overfit catastrophically
(R² = −15,000). With proper scaling and cross-validated regularisation, all degrees
produce sensible results. Degree 3 is optimal here.

**Finding 4 — Splines > Step Functions**
B-Splines and NCS produce smooth, continuous fits that outperform the piecewise-
constant step functions in both RMSE and visual quality.

**Finding 5 — LOESS is a diagnostic tool**
LOESS has no global model and operates in 1D; it is excluded from the RMSE comparison
but valuable for understanding the non-linear shape of individual predictors.
            """)
        with c2:
            rd = R["reg_df"].copy()
            rd.index = range(1, len(rd)+1)
            st.markdown("**Final Regression Ranking:**")
            st.dataframe(
                rd.style
                  .apply(highlight_max, subset=["R²"])
                  .format({"RMSE":"{:.1f}","MAE":"{:.1f}","R²":"{:.4f}"}),
                use_container_width=True, height=280
            )

    with tab2:
        st.subheader("Classification — Tree Methods & SVM on KDD Cup 1999")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
**Finding 1 — All models achieve near-perfect accuracy**
KDD Cup 1999 has extremely strong discriminative features (`src_bytes`, `dst_bytes`,
protocol type). This is expected and reflects the dataset's characteristics, not a
modelling artefact. The differences emerge in subtle metrics (AUC, computation time).

**Finding 2 — Random Forest ≈ Bagging (top performers)**
Both achieve F1 ≈ 0.9999 and AUC ≈ 1.000 on the full test set. RF's random feature
sampling decorrelates trees but provides minimal additional gain here because the
top features are so dominant that most splits land on them regardless.

**Finding 3 — Gradient Boosting is competitive but slower**
Sequential training (200 rounds) takes longer than parallel ensembles, with slightly
lower F1 (0.9998). The learning-curve plot confirms the model generalises well — test
deviance closely follows training deviance.

**Finding 4 — SVM RBF is the best kernel**
The non-linear RBF kernel (AUC=0.9999) outperforms linear (0.9977) and polynomial
(0.9978) kernels, confirming the decision boundary is non-linear.

**Finding 5 — SVM Sigmoid fails**
Sigmoid kernel achieves AUC=0.8861 and F1=0.9686 — significantly below all others.
The sigmoid mapping is not suited to this feature space and is sensitive to feature
scale even after standardisation.

**Finding 6 — SVM vs Ensembles: practical trade-off**
SVMs must be trained on a 15k-sample subset (O(n²) complexity), while tree ensembles
scale to the full 395k training records. In production intrusion detection, ensembles
win on both performance and scalability.
            """)
        with c2:
            cd = R["clf_df"].copy()
            cd.index = range(1, len(cd)+1)
            st.markdown("**Final Classification Ranking:**")
            st.dataframe(
                cd.style
                  .apply(highlight_max, subset=["F1"])
                  .format({"Accuracy":"{:.4f}","F1":"{:.4f}","Precision":"{:.4f}","Recall":"{:.4f}"}),
                use_container_width=True, height=280
            )
            st.markdown("**Practical Trade-offs:**")
            trade_df = pd.DataFrame([
                ["Random Forest",      "✅ Fast train", "✅ Scales to millions", "✅ Feature importance", "❌ Black box"],
                ["Gradient Boosting",  "⚠ Slower (sequential)", "✅ Scales well", "✅ Feature importance", "❌ Black box"],
                ["SVM RBF",            "❌ Slow O(n²)", "❌ Max ~50k practical", "❌ No importance", "✅ Max-margin guarantee"],
                ["Bagging",            "✅ Fast (parallel)", "✅ Scales well", "⚠ Correlated trees", "❌ Black box"],
            ], columns=["Model", "Training Speed", "Scalability", "Interpretability", "Unique Advantage"])
            st.dataframe(trade_df, use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("Team Contributions")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
**Gabil Gurbanov**
- Task 1 — Non-Linear Models:
  - Polynomial Regression (scaling + RidgeCV fix)
  - Step Functions (binning strategy)
  - B-Splines (SplineTransformer)
  - Natural Cubic Splines (patsy)
  - LOESS (bandwidth analysis)
  - GAMs (pygam, partial dependence)
- Regression dataset selection & EDA
- Regression model comparison and visualisations
- Streamlit app — regression sections
            """)
        with c2:
            st.markdown("""
**Hamida Hagverdiyeva**
- Task 2 — Tree-Based Methods:
  - Bagging (OOB scoring)
  - Random Forest (feature importance)
  - Gradient Boosting (deviance tracking)
- Task 3 — SVM:
  - Linear, RBF, Polynomial, Sigmoid kernels
  - Kernel comparison & ROC analysis
- Classification dataset selection & EDA
- Final model comparison report
- Streamlit app — classification sections
            """)
