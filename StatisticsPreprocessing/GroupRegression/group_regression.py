#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分组交互项回归：主产区 vs 非主产区
每组独立跑双向固定效应面板回归，含 5 个 受灾×自变量 交互项。

用法：python group_regression.py
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

# 粮食主产区（13 省）
MAIN_PRODUCERS = {
    "黑龙江", "吉林", "辽宁", "内蒙古", "河北", "河南",
    "山东", "江苏", "安徽", "四川", "湖南", "湖北", "江西",
}

# 自变量 & 因变量 (指标名 → 短列名)
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

# 交互项：受灾 × ...
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


def load_panel() -> pd.DataFrame:
    """从 final 加载面板数据"""
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


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """构造交互项列，命名如 受灾×灌溉"""
    for iv1, iv2 in INTERACTION_PAIRS:
        col_name = f"{iv1}_x_{iv2}"
        if iv1 in df.columns and iv2 in df.columns:
            df[col_name] = df[iv1] * df[iv2]
    return df


def run_regression(df: pd.DataFrame, group_name: str):
    """对分组面板跑双向 FE 回归"""
    # 设定索引
    df = df.reset_index()
    df = df.set_index(["省份", "年份"])

    # 确保所有列都在
    interaction_cols = [f"{a}_x_{b}" for a, b in INTERACTION_PAIRS]
    all_cols = [DV] + IVS + interaction_cols
    missing = [c for c in all_cols if c not in df.columns]
    if missing:
        logger.error("[%s] 缺少列：%s", group_name, missing)
        return

    df = df[all_cols].dropna()

    # 构建公式
    main_part = " + ".join(IVS)
    inter_part = " + ".join(interaction_cols)
    formula = f"{DV} ~ {main_part} + {inter_part} + EntityEffects + TimeEffects"

    logger.info("")
    logger.info("=" * 70)
    logger.info("【%s】回归", group_name)
    logger.info("  省份数：%d | 观测数：%d", df.index.get_level_values(0).nunique(), len(df))
    logger.info("  公式：%s ~ 7 IV + 5 Interaction + Entity + Time", DV)

    model = PanelOLS.from_formula(formula, data=df)
    result = model.fit(cov_type="clustered", cluster_entity=True)

    logger.info(result.summary)

    # 保存
    safe_name = group_name.replace("/", "_")
    txt_path = OUTPUT_DIR / f"regression_{safe_name}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"分组: {group_name}\n")
        f.write(f"省份: {df.index.get_level_values(0).unique().tolist()}\n")
        f.write(str(result.summary))

    # 摘要
    logger.info("-" * 50)
    logger.info("【%s】关键系数：", group_name)
    for var in IVS + interaction_cols:
        coef = result.params.get(var, np.nan)
        pval = result.pvalues.get(var, np.nan)
        sig = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else ""))
        logger.info("  %-16s  coef=%12.6f  p=%7.4f %s", var, coef, pval, sig)

    logger.info("  R² within: %.4f | R² overall: %.4f", result.rsquared_within, result.rsquared_overall)
    logger.info("  结果已保存：%s", txt_path)


def main():
    logger.info("加载数据...")
    panel = load_panel()

    # 构造交互项
    panel = add_interactions(panel)

    # 分组
    main_mask = panel.index.get_level_values("省份").isin(MAIN_PRODUCERS)
    df_main = panel.loc[main_mask].copy()
    df_non = panel.loc[~main_mask].copy()

    logger.info("主产区：%d 省 × %d 年 = %d 行",
                df_main.index.get_level_values(0).nunique(),
                df_main.index.get_level_values(1).nunique(),
                len(df_main))
    logger.info("非主产区：%d 省 × %d 年 = %d 行",
                df_non.index.get_level_values(0).nunique(),
                df_non.index.get_level_values(1).nunique(),
                len(df_non))

    # 分别回归
    run_regression(df_main, "主产区")
    run_regression(df_non, "非主产区")

    logger.info("=" * 70)
    logger.info("全部完成。")


if __name__ == "__main__":
    main()
