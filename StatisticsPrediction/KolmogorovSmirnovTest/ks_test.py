#!/usr/bin/env python3
"""KS test + optimal threshold + re-evaluate XGBoost with that threshold."""
import sys, logging
from pathlib import Path
import numpy as np, pandas as pd
import xgboost as xgb
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from scipy.stats import ks_2samp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

logging.basicConfig(level=logging.INFO, format="%(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
L = logging.getLogger(__name__)

FINAL = Path(__file__).resolve().parent.parent.parent / "StatisticsIndex" / "output_stats_agri_final"
EVENTS = Path(__file__).resolve().parent.parent.parent / "StatisticsProcessing" / "DoubleDifference" / "did_events.csv"
OUT = Path(__file__).resolve().parent

def load():
    mapping = {
        "粮食单位面积产量 (公斤/公顷)": "yield",
        "农业机械总动力 (万千瓦)": "mach",
        "农用塑料薄膜使用量 (吨)": "film",
        "农药使用量 (万吨)": "pest",
        "农村用电量 (亿千瓦小时)": "elec",
        "农用化肥施用折纯量 (万吨)": "fert",
        "有效灌溉面积 (千公顷)": "irri",
        "农作物受灾面积 (千公顷)": "disa",
    }
    IVS = ["mach", "film", "pest", "elec", "fert", "irri", "disa"]
    rows = []
    for d in sorted(FINAL.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        for f in sorted(d.glob("*.csv")):
            df = pd.read_csv(f, encoding="utf-8-sig")
            if "指标名称" not in df.columns:
                continue
            m = df[df["指标名称"].str.strip().isin(mapping.keys())].copy()
            if len(m) == 0:
                continue
            m["s"] = m["指标名称"].str.strip().map(mapping)
            m["prov"] = d.name
            m.rename(columns={"年份": "year", "数值": "val"}, inplace=True)
            rows.append(m[["prov", "year", "s", "val"]])
    panel = pd.concat(rows).pivot_table(
        index=["prov", "year"], columns="s", values="val", aggfunc="first"
    ).sort_index()
    return panel.reset_index(), IVS


def main():
    panel, IVS = load()

    # Lags + dummies
    LAG_VARS = ["yield", "disa"]
    LAGS = [1, 2, 3]
    for prov, g in panel.groupby("prov"):
        g = g.sort_values("year")
        for v in LAG_VARS:
            for lag in LAGS:
                panel.loc[g.index, "{}_lag{}".format(v, lag)] = g[v].shift(lag)
    pdum = pd.get_dummies(panel["prov"], prefix="prv")
    panel = pd.concat([panel, pdum], axis=1)
    panel["year_col"] = panel["year"].astype(int)

    ev = pd.read_csv(EVENTS, encoding="utf-8-sig")
    ev_set = set(zip(ev["省份"], ev["年份"]))
    panel["treated"] = panel.apply(lambda r: 1 if (r["prov"], r["year"]) in ev_set else 0, axis=1)

    lag_cols = ["{}_lag{}".format(v, l) for v in LAG_VARS for l in LAGS]
    feat = ["year_col"] + IVS + lag_cols + list(pdum.columns)
    df = panel.dropna(subset=feat + ["treated"])

    X_train = df[df["year"] <= 2019][feat]
    y_train = df[df["year"] <= 2019]["treated"]
    X_test = df[df["year"] >= 2020][feat]
    y_test = df[df["year"] >= 2020]["treated"]

    scale = max(1, len(y_train[y_train == 0]) / max(1, len(y_train[y_train == 1])))
    model = xgb.XGBClassifier(
        scale_pos_weight=scale, n_estimators=100, max_depth=4,
        learning_rate=0.05, random_state=42,
    )
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]

    # ---- KS statistics ----
    prob_disaster = y_prob[y_test == 1]
    prob_normal = y_prob[y_test == 0]
    ks_stat, ks_p = ks_2samp(prob_disaster, prob_normal)

    L.info("K-S Test:")
    L.info("  KS statistic = %.4f", ks_stat)
    L.info("  p-value      = %.6f", ks_p)
    L.info("  Disaster probs: mean=%.4f median=%.4f", prob_disaster.mean(), np.median(prob_disaster))
    L.info("  Normal probs:   mean=%.4f median=%.4f", prob_normal.mean(), np.median(prob_normal))

    # ---- Find optimal threshold via KS ----
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    ks_vals = tpr - fpr
    best_idx = np.argmax(ks_vals)
    best_threshold = thresholds[best_idx]
    best_ks = ks_vals[best_idx]

    L.info("\nOptimal threshold (max KS):")
    L.info("  threshold = %.4f", best_threshold)
    L.info("  KS = %.4f  (TPR=%.4f FPR=%.4f)", best_ks, tpr[best_idx], fpr[best_idx])

    # ---- Evaluate with optimal threshold ----
    y_pred_opt = (y_prob >= best_threshold).astype(int)
    cm_opt = confusion_matrix(y_test, y_pred_opt)
    tn2, fp2, fn2, tp2 = cm_opt.ravel()

    # Default 0.5
    y_pred_default = (y_prob >= 0.5).astype(int)
    cm_def = confusion_matrix(y_test, y_pred_default)
    tn1, fp1, fn1, tp1 = cm_def.ravel()

    L.info("\nConfusion Matrix comparison:")
    L.info("  %-12s  Default(0.5)  Optimal(%.4f)", " ", best_threshold)
    L.info("  %-12s  %8d    %8d", "TN", tn1, tn2)
    L.info("  %-12s  %8d    %8d", "FP", fp1, fp2)
    L.info("  %-12s  %8d    %8d", "FN", fn1, fn2)
    L.info("  %-12s  %8d    %8d", "TP", tp1, tp2)
    L.info("  %-12s  %8.4f  %8.4f", "Recall", tp1/(tp1+fn1) if tp1+fn1>0 else 0, tp2/(tp2+fn2) if tp2+fn2>0 else 0)
    L.info("  %-12s  %8.4f  %8.4f", "Precision", tp1/(tp1+fp1) if tp1+fp1>0 else 0, tp2/(tp2+fp2) if tp2+fp2>0 else 0)

    # ---- KS curve plot ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, tpr, "b-", lw=2, label="TPR (Recall)")
    ax.plot(thresholds, fpr, "r-", lw=2, label="FPR")
    ax.plot(thresholds, ks_vals, "g--", lw=1.5, label="KS = TPR - FPR")
    ax.axvline(x=best_threshold, color="grey", ls="--", alpha=0.7,
               label="Best threshold=%.4f" % best_threshold)
    ax.axvline(x=0.5, color="orange", ls=":", alpha=0.7, label="Default=0.5")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Rate")
    ax.set_title("KS Curve (KS=%.4f)" % best_ks)
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.savefig(OUT / "ks_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    L.info("\nSaved: %s", OUT / "ks_curve.png")

    L.info("Done.")

if __name__ == "__main__":
    main()
