#!/usr/bin/env python3
"""ROC curve + AUC: same data/features as confusion matrix."""
import sys, logging
from pathlib import Path
import numpy as np, pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_curve, roc_auc_score
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

    # Lags
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

    auc = roc_auc_score(y_test, y_prob)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    L.info("AUC = %.4f", auc)
    L.info("Test samples: %d, positives: %d", len(y_test), y_test.sum())

    # Show a few threshold examples
    L.info("\nThreshold examples:")
    for t in [0.1, 0.2, 0.3, 0.5, 0.7]:
        pred_at_t = (y_prob >= t).astype(int)
        tp = ((pred_at_t == 1) & (y_test == 1)).sum()
        fp = ((pred_at_t == 1) & (y_test == 0)).sum()
        fn = ((pred_at_t == 0) & (y_test == 1)).sum()
        L.info("  threshold=%.1f -> TP=%d FP=%d FN=%d", t, tp, fp, fn)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, "b-", lw=2, label="XGBoost (AUC=%.3f)" % auc)
    ax.plot([0, 1], [0, 1], "grey", ls="--", lw=1, label="Random (AUC=0.5)")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)")
    ax.set_title("ROC Curve: Major Disaster Prediction")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.savefig(OUT / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    L.info("Saved: %s", OUT / "roc_curve.png")

    L.info("\nDone.")


if __name__ == "__main__":
    main()
