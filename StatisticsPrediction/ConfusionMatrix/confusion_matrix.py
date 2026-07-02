#!/usr/bin/env python3
"""Confusion Matrix: XGBoost binary classifier for major disaster events."""
import sys, logging
from pathlib import Path
import numpy as np, pandas as pd
import xgboost as xgb
from sklearn.metrics import confusion_matrix, roc_auc_score
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
            col = "指标名称"
            if col not in df.columns:
                continue
            m = df[df[col].str.strip().isin(mapping.keys())].copy()
            if len(m) == 0:
                continue
            m["s"] = m[col].str.strip().map(mapping)
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

    L.info("Data: %d rows (%d-%d) | treated=1: %d",
           len(df), df["year"].min(), df["year"].max(), df["treated"].sum())

    X_train = df[df["year"] <= 2019][feat]
    y_train = df[df["year"] <= 2019]["treated"]
    X_test = df[df["year"] >= 2020][feat]
    y_test = df[df["year"] >= 2020]["treated"]
    L.info("Train: %d | Test: %d", len(X_train), len(X_test))

    scale = max(1, len(y_train[y_train == 0]) / max(1, len(y_train[y_train == 1])))

    model = xgb.XGBClassifier(
        scale_pos_weight=scale, n_estimators=100, max_depth=4,
        learning_rate=0.05, random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)

    L.info("Confusion Matrix:")
    L.info("                  Pred=0  Pred=1")
    L.info("  Actual=0        %6d  %6d", cm[0, 0], cm[0, 1])
    L.info("  Actual=1        %6d  %6d", cm[1, 0], cm[1, 1])

    tn, fp, fn, tp = cm.ravel()
    total = tp + tn + fp + fn
    L.info("Accuracy : %.4f", (tp + tn) / total)
    L.info("Precision: %.4f", tp / max(1, tp + fp))
    L.info("Recall   : %.4f (disaster detection rate)", tp / max(1, tp + fn))
    L.info("AUC      : %.4f", roc_auc_score(y_test, y_prob))

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=20)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"]); ax.set_yticklabels(["Actual 0", "Actual 1"])
    ax.set_title("Confusion Matrix (AUC=%.3f)" % roc_auc_score(y_test, y_prob))
    fig.savefig(OUT / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    L.info("Saved: %s", OUT / "confusion_matrix.png")

    # Missed disasters
    test_df = df[df["year"] >= 2020].copy()
    test_df["pred"] = y_pred; test_df["prob"] = y_prob
    missed = test_df[(test_df["treated"] == 1) & (test_df["pred"] == 0)]
    if len(missed) > 0:
        L.info("Missed disasters:")
        for _, r in missed.iterrows():
            L.info("  %s %d prob=%.3f", r["prov"], r["year"], r["prob"])

    L.info("Done.")

if __name__ == "__main__":
    main()
