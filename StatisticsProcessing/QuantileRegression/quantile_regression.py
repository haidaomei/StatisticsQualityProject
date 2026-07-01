#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分位数回归：τ = 0.25 / 0.50 / 0.75，全量拟合看系数。
与 XGBoost/RF 相同的特征集，statsmodels.QuantReg。

用法：python quantile_regression.py
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

# -------------------- 配置 --------------------
FINAL_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "StatisticsIndex" / "output_stats_agri_final"
)
OUTPUT_DIR = Path(__file__).resolve().parent

INDICATOR_MAP = {
    "粮食单位面积产量 (公斤/公顷)": "单产",
    "农业机械总动力 (万千瓦)": "农机动力",
    "农用塑料薄膜使用量 (吨)": "薄膜",
    "农药使用量 (万吨)": "农药",
    "农村用电量 (亿千瓦小时)": "用电",
    "农用化肥施用折纯量 (万吨)": "化肥",
    "有效灌溉面积 (千公顷)": "灌溉",
    "农作物受灾面积 (千公顷)": "受灾",
}

DV = "单产"
IVS = ["农机动力", "薄膜", "农药", "用电", "化肥", "灌溉", "受灾"]
LAG_VARS = ["单产", "受灾"]
LAG_ORDERS = [1, 2, 3]
QUANTILES = [0.25, 0.50, 0.75]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_panel():
    rows = []
    for prov_dir in sorted(FINAL_DIR.iterdir()):
        if not prov_dir.is_dir() or prov_dir.name.startswith("."):
            continue
        prov_name = prov_dir.name
        for csv_path in sorted(prov_dir.glob("*.csv")):
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
            if "指标名称" not in df.columns:
                continue
            matched = df[df["指标名称"].str.strip().isin(INDICATOR_MAP.keys())].copy()
            if len(matched) == 0:
                continue
            matched["short_name"] = matched["指标名称"].str.strip().map(INDICATOR_MAP)
            matched["省份"] = prov_name
            matched.rename(columns={"年份": "年份", "数值": "值"}, inplace=True)
            rows.append(matched[["省份", "年份", "short_name", "值"]])
    full = pd.concat(rows, ignore_index=True)
    panel = full.pivot_table(
        index=["省份", "年份"], columns="short_name", values="值", aggfunc="first"
    )
    return panel.sort_index()


def add_lags(df):
    df = df.reset_index()
    for prov, g in df.groupby("省份"):
        g = g.sort_values("年份")
        for var in LAG_VARS:
            for lag in LAG_ORDERS:
                g[f"{var}_lag{lag}"] = g[var].shift(lag)
        df.loc[g.index, g.columns] = g
    return df.set_index(["省份", "年份"]).sort_index()


def main():
    logger.info("加载数据...")
    panel = load_panel()
    panel = add_lags(panel)

    # One-hot 省份
    panel = panel.reset_index()
    prov_dummies = pd.get_dummies(panel["省份"], prefix="prov").astype(int)
    panel = pd.concat([panel, prov_dummies], axis=1)
    panel = panel.set_index(["省份", "年份"]).sort_index()
    panel["年份"] = panel.index.get_level_values("年份").astype(int)

    # 全量数据（drop lag 造成的 2005-2007）
    lag_cols = [f"{v}_lag{l}" for v in LAG_VARS for l in LAG_ORDERS]
    feature_cols = ["年份"] + IVS + lag_cols + [c for c in prov_dummies.columns]
    df_model = panel[[DV] + feature_cols].dropna()

    logger.info("全量数据：%d 行 × %d 列 (%d-%d)",
                len(df_model), len(df_model.columns),
                df_model.index.get_level_values(1).min(),
                df_model.index.get_level_values(1).max())

    X = sm.add_constant(df_model[feature_cols])
    y = df_model[DV]

    # ---- 分位数回归 ----
    results = {}
    for tau in QUANTILES:
        logger.info("=" * 60)
        logger.info("分位数回归 τ=%.2f", tau)
        model = sm.QuantReg(y, X)
        result = model.fit(q=tau)
        results[tau] = result
        logger.info("Pseudo R² = %.4f", result.prsquared)

    # ---- 系数对比表 ----
    logger.info("=" * 60)
    header = f"{'Variable':<20s}  {'τ=0.25':>12s}  {'τ=0.50':>12s}  {'τ=0.75':>12s}"
    logger.info(header)
    logger.info("-" * 70)

    for var in ["const"] + feature_cols:
        if var not in X.columns:
            continue
        vals = []
        for tau in QUANTILES:
            coef = results[tau].params.get(var, np.nan)
            pval = results[tau].pvalues.get(var, np.nan)
            sig = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else ""))
            vals.append(f"{coef:>10.4f}{sig}")
        logger.info(f"  {var:<18s}  {vals[0]:>12s}  {vals[1]:>12s}  {vals[2]:>12s}")

    # ---- 保存 ----
    txt_path = OUTPUT_DIR / "quantile_regression_result.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for tau in QUANTILES:
            f.write(f"\n{'='*60}\n")
            f.write(f"Quantile τ={tau}\n")
            f.write(str(results[tau].summary()))
    logger.info("结果已保存：%s", txt_path)

    # ---- 系数对比图（核心7个IV） ----
    coef_data = {}
    for var in IVS:
        coef_data[var] = [results[tau].params.get(var, np.nan) for tau in QUANTILES]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(QUANTILES))
    width = 0.12
    colors = plt.cm.viridis(np.linspace(0, 1, len(IVS)))
    for i, var in enumerate(IVS):
        ax.bar(x + i * width, coef_data[var], width, label=var, color=colors[i])

    ax.set_xticks(x + width * len(IVS) / 2)
    ax.set_xticklabels([f"τ={t}" for t in QUANTILES])
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_ylabel("Coefficient")
    ax.set_title("Quantile Regression Coefficients (7 Core IVs)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "quantile_coefficients.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("系数对比图已保存：%s", OUTPUT_DIR / "quantile_coefficients.png")

    logger.info("=" * 60)
    logger.info("全部完成。")


if __name__ == "__main__":
    main()
