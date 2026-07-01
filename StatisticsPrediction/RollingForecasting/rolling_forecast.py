#!/usr/bin/env python3
"""Rolling forecast: features(year T) -> yield(year T+1). XGBoost + RF, default params."""
import sys, logging
from pathlib import Path
import numpy as np, pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler(sys.stdout)])
L = logging.getLogger(__name__)

FINAL = Path(__file__).resolve().parent.parent.parent / "StatisticsIndex" / "output_stats_agri_final"
MAP = {"粮食单位面积产量 (公斤/公顷)": "yield", "农业机械总动力 (万千瓦)": "mach",
       "农用塑料薄膜使用量 (吨)": "film", "农药使用量 (万吨)": "pest",
       "农村用电量 (亿千瓦小时)": "elec", "农用化肥施用折纯量 (万吨)": "fert",
       "有效灌溉面积 (千公顷)": "irri", "农作物受灾面积 (千公顷)": "disa"}
IVS = ["mach", "film", "pest", "elec", "fert", "irri", "disa"]

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

def build_df(panel, years):
    """Build DataFrame: features(year T) + DV(year T+1) for given feature years."""
    p = panel.reset_index(); all_rows = []
    for prov, g in p.groupby("prov"):
        g = g.sort_values("year"); g["lag1"] = g["yield"].shift(1)
        for y in years:
            rf = g[g["year"] == y]
            if len(rf) == 0: continue
            rf = rf.iloc[0]
            rd = g[g["year"] == y + 1]
            if len(rd) == 0: continue
            row = {"lag1": rf["lag1"], "year": y, "prov": prov, "dv": rd.iloc[0]["yield"]}
            for v in IVS: row[v] = rf[v] if v in rf.index else np.nan
            all_rows.append(row)
    return pd.DataFrame(all_rows).dropna()

def to_xy(df):
    X = df.drop(columns=["dv", "prov"])
    pdum = pd.get_dummies(df["prov"], prefix="prv")
    X = pd.concat([X, pdum], axis=1)
    return X, df["dv"].values, list(X.columns)

def main():
    panel = load()
    windows = [(range(2005, 2021), 2021), (range(2006, 2022), 2022), (range(2007, 2023), 2023)]
    for train_yrs, test_fy in windows:
        L.info("Window: feat %d-%d -> DV %d-%d | predict DV %d (feat %d)",
               train_yrs[0], train_yrs[-1], train_yrs[0]+1, train_yrs[-1]+1, test_fy+1, test_fy)
        tr = build_df(panel, list(train_yrs))
        te = build_df(panel, [test_fy])
        Xt, yt, cols = to_xy(tr)
        Xp, yp, _ = to_xy(te)
        Xp = Xp.reindex(columns=cols, fill_value=0).values
        Xt = Xt.values
        L.info("  Train=%d Test=%d features=%d", len(Xt), len(Xp), len(cols))
        for name, model in [
            ("XGBoost", xgb.XGBRegressor(random_state=42)),
            ("RF", RandomForestRegressor(random_state=42)),
        ]:
            model.fit(Xt, yt)
            ypred = model.predict(Xp)
            L.info("  %-10s R2=%.4f MAE=%.2f RMSE=%.2f",
                   name, r2_score(yp, ypred), mean_absolute_error(yp, ypred), np.sqrt(mean_squared_error(yp, ypred)))
    L.info("Done.")

if __name__ == "__main__":
    main()
