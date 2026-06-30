#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双向固定效应面板回归：
从 ../StatisticsIndex/output_stats_agri_final/ 读取数据，
构建面板 → Two-way FE (省份 + 年份) → clustered SE

用法：python panel_regression.py
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

# -------------------- 日志 --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_panel() -> pd.DataFrame:
    """从 final 目录加载所有省份的指定指标，合并为面板数据框"""
    rows = []

    for prov_dir in sorted(FINAL_DIR.iterdir()):
        if not prov_dir.is_dir() or prov_dir.name.startswith("."):
            continue
        prov_name = prov_dir.name

        for csv_path in sorted(prov_dir.glob("*.csv")):
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
            if "指标名称" not in df.columns:
                continue

            # 匹配目标指标
            matched = df[df["指标名称"].str.strip().isin(INDICATOR_MAP.keys())].copy()
            if len(matched) == 0:
                continue

            matched["short_name"] = matched["指标名称"].str.strip().map(INDICATOR_MAP)
            matched["省份"] = prov_name
            matched.rename(columns={"年份": "年份", "数值": "值"}, inplace=True)
            rows.append(matched[["省份", "年份", "short_name", "值"]])

    full = pd.concat(rows, ignore_index=True)

    # 透视：行 = (省份, 年份)，列 = short_name
    panel = full.pivot_table(
        index=["省份", "年份"], columns="short_name", values="值", aggfunc="first"
    )
    panel = panel.sort_index()
    return panel


def main():
    logger.info("加载数据...")
    panel = load_panel()

    # 检查
    missing_cols = [c for c in [DV] + IVS if c not in panel.columns]
    if missing_cols:
        logger.error("缺少列：%s", missing_cols)
        sys.exit(1)

    panel = panel[[DV] + IVS].dropna()
    logger.info("面板：%d 行 × %d 列", len(panel), len(panel.columns))
    logger.info("变量：DV=%s, IVs=%s", DV, IVS)

    # 设定 PanelOLS 所需的 MultiIndex: (省份, 年份)
    if not isinstance(panel.index, pd.MultiIndex):
        logger.error("索引不是 MultiIndex")
        sys.exit(1)

    # -------------------- 回归 --------------------
    logger.info("=" * 60)
    logger.info("运行双向固定效应面板回归...")

    # 公式: DV ~ IV1 + IV2 + ... + EntityEffects + TimeEffects
    formula = f"{DV} ~ {' + '.join(IVS)} + EntityEffects + TimeEffects"
    logger.info("公式：%s", formula)

    model = PanelOLS.from_formula(
        formula,
        data=panel,
    )

    # clustered SE (按省份聚类)
    result = model.fit(cov_type="clustered", cluster_entity=True)

    # -------------------- 输出 --------------------
    logger.info("=" * 60)
    logger.info("回归结果：\n%s", result.summary)

    # 保存文本结果
    txt_path = OUTPUT_DIR / "regression_result.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(str(result.summary))
    logger.info("结果已保存：%s", txt_path)

    # 简要输出
    logger.info("=" * 60)
    logger.info("关键系数摘要：")
    for var in IVS:
        coef = result.params.get(var, np.nan)
        pval = result.pvalues.get(var, np.nan)
        sig = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else ""))
        logger.info("  %-10s  coef=%10.6f  p=%7.4f %s", var, coef, pval, sig)

    logger.info("R² (within): %.4f", result.rsquared_within)
    logger.info("R² (overall): %.4f", result.rsquared_overall)
    logger.info("观测数: %d", result.nobs)


if __name__ == "__main__":
    main()
