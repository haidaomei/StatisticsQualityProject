#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双重差分回归：
从 did_events.csv 构建 treated 列 → 双向 FE PanelOLS
DV: 单产 | Key: treated | Controls: 6 IV | 无 lag | 无连续受灾

用法：python did_regression.py
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

# -------------------- 配置 --------------------
FINAL_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "StatisticsIndex" / "output_stats_agri_final"
)
EVENTS_CSV = Path(__file__).resolve().parent / "did_events.csv"
OUTPUT_DIR = Path(__file__).resolve().parent

INDICATOR_MAP = {
    "粮食单位面积产量 (公斤/公顷)": "单产",
    "农业机械总动力 (万千瓦)": "农机动力",
    "农用塑料薄膜使用量 (吨)": "薄膜",
    "农药使用量 (万吨)": "农药",
    "农村用电量 (亿千瓦小时)": "用电",
    "农用化肥施用折纯量 (万吨)": "化肥",
    "有效灌溉面积 (千公顷)": "灌溉",
}

DV = "单产"
IVS = ["农机动力", "薄膜", "农药", "用电", "化肥", "灌溉"]

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


def main():
    logger.info("加载数据...")
    panel = load_panel()

    # 加载 DID 事件，构建 treated 列
    events = pd.read_csv(EVENTS_CSV, encoding="utf-8-sig")
    event_set = set(zip(events["省份"], events["年份"]))
    panel["treated"] = 0
    for idx in panel.index:
        if (idx[0], idx[1]) in event_set:
            panel.at[idx, "treated"] = 1.0

    logger.info("面板：%d 行 × %d 列", len(panel), len(panel.columns))
    logger.info("treated=1 的行数：%d", int(panel["treated"].sum()))

    # 全量数据，无 lag
    feature_cols = ["treated"] + IVS
    df_model = panel[[DV] + feature_cols].dropna()

    logger.info("建模数据：%d 行 (2005-2024)", len(df_model))

    # 双向 FE + clustered SE
    formula = f"{DV} ~ {' + '.join(feature_cols)} + EntityEffects + TimeEffects"
    logger.info("公式：%s", formula)

    model = PanelOLS.from_formula(formula, data=df_model)
    result = model.fit(cov_type="clustered", cluster_entity=True)

    # 手动构建摘要
    lines = []
    lines.append("=" * 70)
    lines.append("DID PanelOLS Two-way FE")
    lines.append(f"Observations: {len(df_model)} | Entities: {df_model.index.get_level_values(0).nunique()}")
    lines.append(f"Treated events: {int(panel['treated'].sum())}")
    lines.append(f"R² within: {result.rsquared_within:.4f} | R² overall: {result.rsquared_overall:.4f}")
    lines.append("=" * 70)
    lines.append(f"{'Variable':<16s} {'Coef':>12s} {'StdErr':>10s} {'T-stat':>10s} {'P-value':>10s}")
    lines.append("-" * 60)
    for var in result.params.index:
        lines.append(f"{var:<16s} {result.params[var]:>12.6f} {result.std_errors[var]:>10.6f} {result.tstats[var]:>10.4f} {result.pvalues[var]:>10.4f}")
    lines.append("")
    summary_str = "\n".join(lines)
    logger.info(summary_str)

    # 保存
    txt_path = OUTPUT_DIR / "did_result.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary_str)
    logger.info("结果已保存：%s", txt_path)

    # 关键摘要
    logger.info("-" * 50)
    logger.info("关键系数（treated = 特大灾害冲击）：")
    for var in ["treated"] + IVS:
        coef = result.params.get(var, np.nan)
        pval = result.pvalues.get(var, np.nan)
        sig = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else ""))
        logger.info("  %-12s  coef=%10.4f  p=%7.4f %s", var, coef, pval, sig)

    logger.info("=" * 70)
    logger.info("全部完成。")


if __name__ == "__main__":
    main()
