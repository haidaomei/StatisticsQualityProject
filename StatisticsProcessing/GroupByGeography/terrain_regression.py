#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三级阶梯分组交互项回归：与之前相同的双向 FE + 5 交互项。
第一级（2省）降级处理。

用法：python terrain_regression.py
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
OUTPUT_DIR = Path(__file__).resolve().parent

TERRAIN_ZONES = {
    "第一级阶梯": {"西藏", "青海"},
    "第二级阶梯": {
        "新疆", "内蒙古", "甘肃", "宁夏", "陕西",
        "山西", "四川", "重庆", "贵州", "云南",
    },
    "第三级阶梯": {
        "黑龙江", "吉林", "辽宁",
        "北京", "天津", "河北",
        "山东", "河南",
        "江苏", "浙江", "上海", "安徽", "江西",
        "福建", "广东", "海南", "广西",
        "湖北", "湖南",
    },
}

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
INTERACTION_PAIRS = [
    ("受灾", "灌溉"),
    ("受灾", "农机动力"),
    ("受灾", "农药"),
    ("受灾", "化肥"),
    ("受灾", "薄膜"),
]

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


def add_interactions(df):
    for iv1, iv2 in INTERACTION_PAIRS:
        col = f"{iv1}_x_{iv2}"
        if iv1 in df.columns and iv2 in df.columns:
            df[col] = df[iv1] * df[iv2]
    return df


def run_regression(df, zone_name):
    df = df.reset_index().set_index(["省份", "年份"])
    interaction_cols = [f"{a}_x_{b}" for a, b in INTERACTION_PAIRS]
    all_cols = [DV] + IVS + interaction_cols
    df = df[[c for c in all_cols if c in df.columns]].dropna()

    n_entities = df.index.get_level_values(0).nunique()
    n_obs = len(df)

    logger.info("=" * 70)
    logger.info("【%s】%d 省 × %d 观测", zone_name, n_entities, n_obs)

    if n_entities < 3:
        logger.warning("  ⚠ 省份 < 3，EntityEffects 不可靠，降级为仅 TimeEffects")
        formula = f"{DV} ~ {' + '.join(IVS + interaction_cols)} + TimeEffects"
    else:
        formula = f"{DV} ~ {' + '.join(IVS + interaction_cols)} + EntityEffects + TimeEffects"

    try:
        model = PanelOLS.from_formula(formula, data=df)
        result = model.fit(cov_type="clustered", cluster_entity=True)
    except Exception as e:
        logger.error("  回归失败：%s", e)
        return

    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"PanelOLS FE: {zone_name}")
    lines.append(f"Observations: {n_obs} | Entities: {n_entities}")
    lines.append(f"R² within: {result.rsquared_within:.4f} | R² overall: {result.rsquared_overall:.4f}")
    lines.append(f"{'='*60}")
    lines.append(f"{'Variable':<20s} {'Coef':>12s} {'StdErr':>10s} {'T-stat':>10s} {'P-value':>10s}")
    lines.append("-" * 60)
    for var in result.params.index:
        lines.append(f"{var:<20s} {result.params[var]:>12.6f} {result.std_errors[var]:>10.6f} {result.tstats[var]:>10.4f} {result.pvalues[var]:>10.4f}")
    lines.append("")
    summary_str = "\n".join(lines)
    logger.info(summary_str)

    safe = zone_name.replace("/", "_")
    txt_path = OUTPUT_DIR / f"regression_{safe}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"地形区: {zone_name}\n")
        f.write(f"省份: {df.index.get_level_values(0).unique().tolist()}\n")
        f.write(f"观测数: {n_obs}\n")
        f.write(summary_str)

    logger.info("-" * 50)
    logger.info("【%s】关键系数：", zone_name)
    for var in IVS + interaction_cols:
        if var not in result.params:
            continue
        coef = result.params[var]
        pval = result.pvalues[var]
        sig = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else ""))
        logger.info("  %-16s  coef=%12.6f  p=%7.4f %s", var, coef, pval, sig)
    logger.info("  R² within: %.4f | R² overall: %.4f", result.rsquared_within, result.rsquared_overall)


def main():
    logger.info("加载数据...")
    panel = load_panel()
    panel = add_interactions(panel)

    for zone_name, provinces in TERRAIN_ZONES.items():
        mask = panel.index.get_level_values("省份").isin(provinces)
        df_zone = panel.loc[mask].copy()
        if len(df_zone) == 0:
            logger.warning("【%s】无匹配省份，跳过", zone_name)
            continue
        run_regression(df_zone, zone_name)

    logger.info("=" * 70)
    logger.info("全部完成。")


if __name__ == "__main__":
    main()
