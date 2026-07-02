#!/usr/bin/env python3
"""
Placebo test: Fake treatment = Henan 2022 (good harvest year).
DID + SCM. If insignificant / no gap, methods are robust.
"""
import sys, logging
from pathlib import Path
import numpy as np, pandas as pd
from linearmodels.panel import PanelOLS
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

logging.basicConfig(level=logging.INFO, format="%(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
L = logging.getLogger(__name__)

FINAL = Path(__file__).resolve().parent.parent.parent / "StatisticsIndex" / "output_stats_agri_final"
OUT = Path(__file__).resolve().parent
TU, TY = "河南", 2022

MAP = {
    "粮食单位面积产量 (公斤/公顷)": "yield",
    "农业机械总动力 (万千瓦)": "mach",
    "农用塑料薄膜使用量 (吨)": "film",
    "农药使用量 (万吨)": "pest",
    "农村用电量 (亿千瓦小时)": "elec",
    "农用化肥施用折纯量 (万吨)": "fert",
    "有效灌溉面积 (千公顷)": "irri",
    "农作物受灾面积 (千公顷)": "disa",
}
OUTCOME = "yield"
IVS = ["mach", "film", "pest", "elec", "fert", "irri", "disa"]
COV = ["mach", "film", "pest", "elec", "fert", "irri", "disa"]

def load():
    rows = []
    for d in sorted(FINAL.iterdir()):
        if not d.is_dir() or d.name.startswith("."): continue
        for f in sorted(d.glob("*.csv")):
            df = pd.read_csv(f, encoding="utf-8-sig")
            if "指标名称" not in df.columns: continue
            m = df[df["指标名称"].str.strip().isin(MAP.keys())].copy()
            if len(m) == 0: continue
            m["s"] = m["指标名称"].str.strip().map(MAP); m["prov"] = d.name
            m.rename(columns={"年份": "year", "数值": "val"}, inplace=True)
            rows.append(m[["prov", "year", "s", "val"]])
    return pd.concat(rows).pivot_table(index=["prov", "year"], columns="s", values="val", aggfunc="first").sort_index()

def main():
    panel = load()

    # ---- 1. Placebo DID ----
    L.info("=" * 60)
    L.info("[Placebo DID] Fake treatment: %s %d", TU, TY)
    panel_df = panel.copy()
    panel_df["treated"] = 0
    for idx in panel_df.index:
        if idx[0] == TU and idx[1] == TY:
            panel_df.at[idx, "treated"] = 1.0
    df_did = panel_df[[OUTCOME, "treated"] + IVS].dropna()
    formula = "{} ~ treated + {} + EntityEffects + TimeEffects".format(OUTCOME, " + ".join(IVS))
    m = PanelOLS.from_formula(formula, data=df_did)
    r = m.fit(cov_type="clustered", cluster_entity=True)
    coef_t = r.params.get("treated", np.nan)
    pval_t = r.pvalues.get("treated", np.nan)
    L.info("  treated coefficient: %.4f  p=%.4f", coef_t, pval_t)
    if pval_t > 0.1:
        L.info("  => NOT significant (placebo passed)")
    else:
        L.info("  => Significant! (placebo FAILED)")

    # ---- 2. Placebo SCM ----
    L.info("\n[Placebo SCM] Fake treatment: %s %d", TU, TY)
    ev_csv = Path(__file__).resolve().parent.parent.parent / "StatisticsProcessing" / "DoubleDifference" / "did_events.csv"
    events = pd.read_csv(ev_csv, encoding="utf-8-sig")
    exc = {TU}
    for _, row in events.iterrows():
        if abs(int(row["年份"]) - TY) <= 2 and row["省份"] != TU:
            exc.add(row["省份"])
    donors = sorted(set(panel.index.get_level_values(0)) - exc)
    L.info("  Donor pool: %d provinces", len(donors))

    PRE_S, PRE_E = 2005, 2021
    def extract(prov):
        dp = panel.loc[prov]; m = (dp.index >= PRE_S) & (dp.index <= PRE_E)
        return np.concatenate([dp.loc[m, OUTCOME].values.astype(float),
                               dp.loc[m, COV].mean().values.astype(float)])
    Yt = extract(TU); Yd = np.column_stack([extract(p) for p in donors])
    for i in range(Yd.shape[0]):
        s = np.std(Yd[i]); Yd[i] /= (s if s > 1e-8 else 1); Yt[i] /= (s if s > 1e-8 else 1)

    n = len(donors); best_w, best_loss = None, 1e99
    for _ in range(30):
        w0 = np.random.dirichlet(np.ones(n))
        res = minimize(lambda w: np.sum((Yd @ w - Yt) ** 2), w0, bounds=[(0, 1)] * n,
                       constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}],
                       method="SLSQP", options={"maxiter": 10000, "ftol": 1e-15})
        if res.fun < best_loss: best_loss = res.fun; best_w = res.x
    w = best_w
    L.info("  Top weights:")
    for p, wi in sorted(zip(donors, w), key=lambda x: -x[1])[:3]:
        if wi > 0.01: L.info("    %s %.4f", p, wi)

    yrs = list(range(2005, 2025))
    act = np.array([panel.loc[(TU, yr), OUTCOME] for yr in yrs])
    syn = sum(w[j] * np.array([panel.loc[(p, yr), OUTCOME] for yr in yrs]) for j, p in enumerate(donors))
    gap = act - syn
    rmsp = np.sqrt(np.mean(gap[:PRE_E - PRE_S + 1] ** 2))
    L.info("  Pre-RMSPE: %.2f", rmsp)
    idx22 = TY - 2005
    L.info("  %d: actual=%.2f synth=%.2f ATT=%.2f", TY, act[idx22], syn[idx22], gap[idx22])
    if abs(gap[idx22]) < 1.5 * np.std(gap[:PRE_E - PRE_S + 1]):
        L.info("  => Gap within 1.5x pre-treatment std (placebo passed)")
    else:
        L.info("  => Gap outside range! (placebo FAILED)")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(yrs, act, "k-o", label="Actual {}".format(TU), lw=2, ms=5)
    ax.plot(yrs, syn, "r--s", label="Synthetic", lw=2, ms=5)
    ax.axvline(x=TY - 0.5, color="grey", ls="--", alpha=0.7)
    ax.legend(); ax.set_title("Placebo SCM: {} {} (RMSPE={:.1f})".format(TU, TY, rmsp))
    fig.savefig(OUT / "placebo_scm.png", dpi=150, bbox_inches="tight"); plt.close()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(yrs, gap, "b-o", lw=2, ms=5)
    ax.axvline(x=TY - 0.5, color="grey", ls="--", alpha=0.7)
    ax.axhline(y=0, color="black", lw=0.5)
    ax.set_title("Placebo Gap: {} {}".format(TU, TY))
    fig.savefig(OUT / "placebo_gap.png", dpi=150, bbox_inches="tight"); plt.close()
    L.info("  Plots saved: %s", OUT)

    L.info("\nDone.")

if __name__ == "__main__":
    main()
