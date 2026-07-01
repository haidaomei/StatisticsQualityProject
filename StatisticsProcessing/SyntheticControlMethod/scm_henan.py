#!/usr/bin/env python3
"""SCM: Henan 2023 major disaster. Standardized + multi-start optimization."""
import sys, logging
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

FINAL = Path(__file__).resolve().parent.parent.parent / "StatisticsIndex" / "output_stats_agri_final"
EVENTS = Path(__file__).resolve().parent.parent / "DoubleDifference" / "did_events.csv"
OUT = Path(__file__).resolve().parent
TU = "河南"
TY = 2023
PS, PE = 2005, 2022  # pre-treatment
QS, QE = 2023, 2024  # post-treatment

NAME_MAP = {
    "nongzuowu danwei mianji chanliang (gongjin/gongqing)": "yield",
    "nongye jixie zong dongli (wan qianwa)": "mach",
    "nongyong suliao bomoshiyongliang (dun)": "film",
    "nongyao shiyongliang (wan dun)": "pest",
    "nongcun yongdianliang (yi qianwashixiaoshi)": "elec",
    "nongyong huafei shiyong zhechunliang (wan dun)": "fert",
    "youxiao guangai mianji (qian gongqing)": "irri",
    "nongzuowu shouhai mianji (qian gongqing)": "disa",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
L = logging.getLogger(__name__)

# ---------- data loading ----------
def load_panel():
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
    rows = []
    for d in sorted(FINAL.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        for f in sorted(d.glob("*.csv")):
            df = pd.read_csv(f, encoding="utf-8-sig")
            if "indic_name" not in df.columns and "指标名称" not in df.columns:
                continue
            col = "指标名称" if "指标名称" in df.columns else "indic_name"
            m = df[df[col].str.strip().isin(mapping.keys())].copy()
            if len(m) == 0:
                continue
            m["s"] = m[col].str.strip().map(mapping)
            m["p"] = d.name
            yr_col = "年份" if "年份" in df.columns else "year"
            m.rename(columns={yr_col: "y", "数值": "v"}, inplace=True)
            rows.append(m[["p", "y", "s", "v"]])
    return pd.concat(rows).pivot_table(index=["p", "y"], columns="s", values="v", aggfunc="first").sort_index()


def extract(panel, prov):
    dp = panel.loc[prov]
    m = (dp.index >= PS) & (dp.index <= PE)
    outcome_ts = dp.loc[m, "yield"].values.astype(float)
    cov_avg = dp.loc[m, COV].mean().values.astype(float)
    return np.concatenate([outcome_ts, cov_avg])


# ---------- main ----------
def main():
    L.info("Loading data...")
    panel = load_panel()
    global COV
    COV = [c for c in ["mach", "film", "pest", "elec", "fert", "irri", "disa"] if c in panel.columns]

    # Donor pool
    ev = pd.read_csv(EVENTS, encoding="utf-8-sig")
    exc = {TU}
    for _, r in ev.iterrows():
        if abs(int(r["year"] if "year" in r else r["年份"]) - TY) <= 2 and r.get("prov", r.get("省份")) != TU:
            exc.add(r.get("prov", r.get("省份")))
    donors = sorted(set(panel.index.get_level_values(0)) - exc)
    L.info("Donor pool: %d provinces (excluded: %s)", len(donors), exc)

    # Extract targets
    Yt = extract(panel, TU)
    Yd = np.column_stack([extract(panel, p) for p in donors])

    # Standardize each row
    for i in range(Yd.shape[0]):
        s = np.std(Yd[i])
        if s > 1e-8:
            Yd[i] /= s
            Yt[i] /= s

    # Multi-start optimization
    n = len(donors)
    best_w = None
    best_loss = 1e99
    for _ in range(30):
        w0 = np.random.dirichlet(np.ones(n))
        res = minimize(
            lambda w: np.sum((Yd @ w - Yt) ** 2),
            w0,
            bounds=[(0, 1)] * n,
            constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}],
            method="SLSQP",
            options={"maxiter": 10000, "ftol": 1e-15},
        )
        if res.fun < best_loss:
            best_loss = res.fun
            best_w = res.x

    w = best_w
    L.info("Top weights:")
    for p, wi in sorted(zip(donors, w), key=lambda x: -x[1]):
        if wi > 0.01:
            L.info("  %s  %.4f", p, wi)

    # Compute synthetic path
    yrs = list(range(PS, QE + 1))
    act = np.array([panel.loc[(TU, yr), "yield"] for yr in yrs])
    syn = sum(w[j] * np.array([panel.loc[(p, yr), "yield"] for yr in yrs]) for j, p in enumerate(donors))
    gap = act - syn
    rmsp = np.sqrt(np.mean(gap[: PE - PS + 1] ** 2))
    L.info("Pre-RMSPE: %.2f", rmsp)

    for yr in [2023, 2024]:
        i = yr - PS
        L.info("  %d actual=%.2f synth=%.2f ATT=%.2f", yr, act[i], syn[i], gap[i])

    # Path plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(yrs, act, "k-o", label=f"Actual {TU}", lw=2, ms=5)
    ax.plot(yrs, syn, "r--s", label="Synthetic", lw=2, ms=5)
    ax.axvline(x=TY - 0.5, color="grey", ls="--", alpha=0.7)
    ax.legend()
    ax.set_title(f"SCM: {TU} {TY} Disaster (Pre-RMSPE={rmsp:.1f})")
    ax.set_xlabel("Year")
    ax.set_ylabel("Yield (kg/ha)")
    fig.savefig(OUT / "scm_path.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Gap plot
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(yrs, gap, "b-o", lw=2, ms=5)
    ax.axvline(x=TY - 0.5, color="grey", ls="--", alpha=0.7)
    ax.axhline(y=0, color="black", lw=0.5)
    ax.set_title(f"Disaster Impact Gap: {TU} {TY}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Actual - Synthetic (kg/ha)")
    fig.savefig(OUT / "scm_gap.png", dpi=150, bbox_inches="tight")
    plt.close()

    L.info("Plots saved to %s", OUT)
    L.info("Done.")


if __name__ == "__main__":
    main()
