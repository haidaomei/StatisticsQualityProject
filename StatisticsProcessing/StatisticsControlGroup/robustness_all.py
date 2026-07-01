#!/usr/bin/env python3
"""
稳健性分析：DV 替换为 农业总产值 (亿元)
依次运行：双向FE面板回归 → 主/非主产区 → 气候区 → 地形 → 分位数 → XGBoost → RF
"""
import sys, logging
from pathlib import Path
import numpy as np, pandas as pd
from linearmodels.panel import PanelOLS
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import statsmodels.api as sm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
L = logging.getLogger(__name__)

# ======================== CONFIG ========================
FINAL = Path(__file__).resolve().parent.parent.parent / "StatisticsIndex" / "output_stats_agri_final"
OUT = Path(__file__).resolve().parent

INDICATOR_MAP = {
    "农业总产值 (亿元)": "总产值",
    "农业机械总动力 (万千瓦)": "农机动力",
    "农用塑料薄膜使用量 (吨)": "薄膜",
    "农药使用量 (万吨)": "农药",
    "农村用电量 (亿千瓦小时)": "用电",
    "农用化肥施用折纯量 (万吨)": "化肥",
    "有效灌溉面积 (千公顷)": "灌溉",
    "农作物受灾面积 (千公顷)": "受灾",
}

DV = "总产值"
IVS = ["农机动力", "薄膜", "农药", "用电", "化肥", "灌溉", "受灾"]
INTERACTIONS = [("受灾", "灌溉"), ("受灾", "农机动力"), ("受灾", "农药"), ("受灾", "化肥"), ("受灾", "薄膜")]
INTER_COLS = [f"{a}_x_{b}" for a, b in INTERACTIONS]

MAIN_PRODUCERS = {"黑龙江","吉林","辽宁","内蒙古","河北","河南","山东","江苏","安徽","四川","湖南","湖北","江西"}

CLIMATE = {
    "温带季风": {"北京","天津","河北","山西","辽宁","吉林","黑龙江","山东","河南","陕西"},
    "亚热带季风": {"上海","江苏","浙江","安徽","福建","江西","湖北","湖南","广东","广西","重庆","四川","贵州","云南"},
    "温带大陆性": {"内蒙古","甘肃","宁夏","新疆"},
}

TERRAIN = {
    "第二级阶梯": {"新疆","内蒙古","甘肃","宁夏","陕西","山西","四川","重庆","贵州","云南"},
    "第三级阶梯": {"黑龙江","吉林","辽宁","北京","天津","河北","山东","河南","江苏","浙江","上海","安徽","江西","福建","广东","海南","广西","湖北","湖南"},
}

# ======================== DATA LOADING ========================
def load_panel():
    rows = []
    for d in sorted(FINAL.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        for f in sorted(d.glob("*.csv")):
            df = pd.read_csv(f, encoding="utf-8-sig")
            if "指标名称" not in df.columns:
                continue
            m = df[df["指标名称"].str.strip().isin(INDICATOR_MAP.keys())].copy()
            if len(m) == 0:
                continue
            m["s"] = m["指标名称"].str.strip().map(INDICATOR_MAP)
            m["prov"] = d.name
            m.rename(columns={"年份": "year", "数值": "val"}, inplace=True)
            rows.append(m[["prov", "year", "s", "val"]])
    return pd.concat(rows, ignore_index=True).pivot_table(
        index=["prov", "year"], columns="s", values="val", aggfunc="first"
    ).sort_index()

def add_inter(df):
    for a, b in INTERACTIONS:
        if a in df.columns and b in df.columns:
            df[f"{a}_x_{b}"] = df[a] * df[b]
    return df

def add_lags(df):
    df = df.reset_index()
    for prov, g in df.groupby("prov"):
        g = g.sort_values("year")
        for lag in [1, 2, 3]:
            g[f"{DV}_lag{lag}"] = g[DV].shift(lag)
            g[f"受灾_lag{lag}"] = g["受灾"].shift(lag)
        df.loc[g.index, g.columns] = g
    return df.set_index(["prov", "year"]).sort_index()

def run_panel_ols(df, formula, name, clustered=True):
    try:
        m = PanelOLS.from_formula(formula, data=df)
        r = m.fit(cov_type="clustered", cluster_entity=True) if clustered else m.fit()
    except Exception as e:
        L.error("  [%s] FAILED: %s", name, e)
        return None
    L.info("  [%s] R2_within=%.4f R2_overall=%.4f N=%d", name, r.rsquared_within, r.rsquared_overall, r.nobs)
    for v in r.params.index[:15]:
        coef, pv = r.params[v], r.pvalues[v]
        sig = "***" if pv < 0.01 else ("**" if pv < 0.05 else ("*" if pv < 0.1 else ""))
        if pv < 0.2:
            L.info("    %-20s %12.4f p=%.4f %s", v, coef, pv, sig)
    return r

# ======================== MAIN ========================
def main():
    L.info("=" * 70)
    L.info("ROBUSTNESS CHECK: DV = %s", DV)
    L.info("=" * 70)

    # ---- Load ----
    panel = load_panel()
    panel = add_inter(panel)
    df_full = panel.copy()
    L.info("Panel: %d rows x %d cols", len(df_full), len(df_full.columns))

    # ========== 1. Two-way FE Panel ==========
    L.info("\n[1] Panel FE (all provinces)")
    formula = f"{DV} ~ {' + '.join(IVS + INTER_COLS)} + EntityEffects + TimeEffects"
    run_panel_ols(df_full, formula, "FE")

    # ========== 2. Main vs Non-main ==========
    for grp_name, mask_fn in [
        ("主产区", lambda: df_full.index.get_level_values("prov").isin(MAIN_PRODUCERS)),
        ("非主产区", lambda: ~df_full.index.get_level_values("prov").isin(MAIN_PRODUCERS)),
    ]:
        m = mask_fn()
        df_g = df_full.loc[m].copy().dropna()
        if len(df_g) < 40:
            L.warning("  [%s] <40 obs, skip", grp_name)
            continue
        n_ent = df_g.index.get_level_values(0).nunique()
        fe = "EntityEffects + TimeEffects" if n_ent >= 3 else "TimeEffects"
        L.info("\n[2] %s (%d prov, %d obs)", grp_name, n_ent, len(df_g))
        formula = f"{DV} ~ {' + '.join(IVS + INTER_COLS)} + {fe}"
        run_panel_ols(df_g, formula, grp_name)

    # ========== 3. Climate ==========
    for cname, cprovinces in CLIMATE.items():
        df_c = df_full.loc[df_full.index.get_level_values("prov").isin(cprovinces)].copy().dropna()
        if len(df_c) < 40:
            continue
        n_ent = df_c.index.get_level_values(0).nunique()
        fe = "EntityEffects + TimeEffects" if n_ent >= 3 else "TimeEffects"
        L.info("\n[3] Climate: %s (%d prov, %d obs)", cname, n_ent, len(df_c))
        formula = f"{DV} ~ {' + '.join(IVS + INTER_COLS)} + {fe}"
        run_panel_ols(df_c, formula, cname)

    # ========== 4. Terrain ==========
    for tname, tprovinces in TERRAIN.items():
        df_t = df_full.loc[df_full.index.get_level_values("prov").isin(tprovinces)].copy().dropna()
        if len(df_t) < 40:
            continue
        n_ent = df_t.index.get_level_values(0).nunique()
        L.info("\n[4] Terrain: %s (%d prov, %d obs)", tname, n_ent, len(df_t))
        formula = f"{DV} ~ {' + '.join(IVS + INTER_COLS)} + EntityEffects + TimeEffects"
        run_panel_ols(df_t, formula, tname)

    # ========== 5. Quantile Regression ==========
    L.info("\n[5] Quantile Regression (full sample)")
    qdf = df_full[[DV] + IVS].dropna()
    X = sm.add_constant(qdf[IVS])
    y = qdf[DV]
    for tau in [0.25, 0.50, 0.75]:
        qr = sm.QuantReg(y, X).fit(q=tau, max_iter=100000)
        L.info("  tau=%.2f  PseudoR2=%.4f", tau, qr.prsquared)
        for v in ["受灾", "灌溉", "农机动力", "化肥", "农药"]:
            if v in qr.params:
                coef, pv = qr.params[v], qr.pvalues[v]
                sig = "***" if pv < 0.01 else ("**" if pv < 0.05 else ("*" if pv < 0.1 else ""))
                L.info("    %s %10.4f p=%.4f %s", v, coef, pv, sig)

    # ========== 6. XGBoost + RF ==========
    L.info("\n[6] ML Models (with lags, 2008-2019 train / 2020-2024 test)")
    ml_panel = load_panel()
    ml_panel = add_lags(ml_panel)

    ml_panel = ml_panel.reset_index()
    pdum = pd.get_dummies(ml_panel["prov"], prefix="prv").astype(int)
    ml_panel = pd.concat([ml_panel, pdum], axis=1)
    ml_panel = ml_panel.set_index(["prov", "year"]).sort_index()
    ml_panel["year"] = ml_panel.index.get_level_values("year").astype(int)

    lag_cols = [f"{DV}_lag{l}" for l in [1, 2, 3]] + [f"受灾_lag{l}" for l in [1, 2, 3]]
    feat = ["year"] + IVS + lag_cols + list(pdum.columns)
    dfm = ml_panel[[DV] + feat].dropna()

    L.info("  ML data: %d rows (%d-%d)", len(dfm),
           dfm.index.get_level_values(1).min(), dfm.index.get_level_values(1).max())

    train = dfm.index.get_level_values("year") <= 2019
    test = dfm.index.get_level_values("year") >= 2020
    Xt, yt = dfm.loc[train, feat], dfm.loc[train, DV]
    Xe, ye = dfm.loc[test, feat], dfm.loc[test, DV]

    for name, model in [
        ("XGBoost", xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                                      subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)),
        ("RF", RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=3,
                                     random_state=42, n_jobs=-1)),
    ]:
        model.fit(Xt, yt)
        yp = model.predict(Xe)
        L.info("  %s: test R2=%.4f MAE=%.2f", name, r2_score(ye, yp), mean_absolute_error(ye, yp))
        imp = pd.DataFrame({"f": feat, "i": model.feature_importances_}).sort_values("i", ascending=False)
        for _, r in imp.head(5).iterrows():
            L.info("    %-20s %.4f", r["f"], r["i"])

    L.info("\n" + "=" * 70)
    L.info("All robustness checks complete.")


if __name__ == "__main__":
    main()
