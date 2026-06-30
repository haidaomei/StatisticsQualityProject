#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
农药使用量补救脚本：
从 screened 读取 农药使用量 (万吨) ，
按省份拆分 → 线性插值+外推 → 3σ 去噪 → 再线性插值+外推，
追加到 final 各省的 农用柴油和农药使用量 CSV 中。

用法：python add_pesticide.py
"""

import logging
import sys
import math
from pathlib import Path

import numpy as np
import pandas as pd

# -------------------- 配置 --------------------
SCREENED_FILE = (
    Path(__file__).resolve().parent.parent.parent
    / "StatisticsIndex"
    / "output_stats_agri_screened"
    / "农用柴油和农药使用量_分省_2005YY-2024YY.csv"
)
FINAL_DIR = Path(__file__).resolve().parent

YEAR_START = 2005
YEAR_END = 2024
ALL_YEARS = np.arange(YEAR_START, YEAR_END + 1, dtype=float)
SIGMA = 3

# -------------------- 日志 --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def fill_group(years, values):
    """线性插值（闭区间）+ OLS 外推（首尾），返回填充后数组"""
    filled = values.copy()
    n = len(values)
    mask = ~np.isnan(values)
    if not mask.any():
        return np.zeros(n)
    idx = np.where(mask)[0]
    first, last = idx[0], idx[-1]

    # 闭区间插值
    if first < last:
        x_v = years[first:last+1][mask[first:last+1]]
        y_v = values[first:last+1][mask[first:last+1]]
        filled[first:last+1] = np.interp(years[first:last+1], x_v, y_v)

    # OLS 外推
    if first > 0 or last < n - 1:
        x_train = years[first:last+1]
        y_train = filled[first:last+1]
        if len(x_train) >= 2:
            a = np.cov(x_train, y_train)[0, 1] / np.var(x_train)
            b = np.mean(y_train) - a * np.mean(x_train)
            if first > 0:
                filled[:first] = a * years[:first] + b
            if last < n - 1:
                filled[last+1:] = a * years[last+1:] + b
        else:
            v = filled[first]
            if first > 0:
                filled[:first] = v
            if last < n - 1:
                filled[last+1:] = v
    return filled


def denoise_and_fill(values):
    """3σ 去噪 → 抹 NaN → 重新线性填充"""
    std = float(np.nanstd(values, ddof=1))
    mean = float(np.nanmean(values))
    if std == 0:
        return values  # 无噪值

    lower, upper = mean - SIGMA * std, mean + SIGMA * std
    cleaned = values.copy()
    mask_outlier = (cleaned < lower) | (cleaned > upper)
    if mask_outlier.sum() == 0:
        return values

    cleaned[mask_outlier] = np.nan
    return fill_group(ALL_YEARS, cleaned)


def main():
    if not SCREENED_FILE.exists():
        logger.error("screened 文件不存在：%s", SCREENED_FILE)
        sys.exit(1)

    df_all = pd.read_csv(SCREENED_FILE, encoding="utf-8-sig")
    df_pest = df_all[df_all["指标名称"].str.contains("农药")].copy()

    if len(df_pest) == 0:
        logger.error("未找到农药使用量数据")
        sys.exit(1)

    logger.info("读取农药使用量：%d 行，%d 省", len(df_pest), df_pest["地区"].nunique())

    # 按省份处理
    total_outliers = 0
    for prov, gdf in df_pest.groupby("地区"):
        gdf = gdf.sort_values("年份").reset_index(drop=True)

        # Step 1: 线性填充
        values = gdf["数值"].to_numpy(dtype=float)
        filled1 = fill_group(gdf["年份"].to_numpy(dtype=float), values)

        # Step 2: 3σ 去噪 + 再填充
        final_values = denoise_and_fill(filled1)

        # 统计
        outliers = int((np.abs(final_values - filled1) > 1e-6).sum())
        total_outliers += outliers

        gdf["数值"] = final_values

        # 追加到 final 目录该省对应的 CSV
        prov_dir = FINAL_DIR / str(prov).strip()
        prov_dir.mkdir(parents=True, exist_ok=True)
        target = prov_dir / "农用柴油和农药使用量_分省_2005YY-2024YY.csv"

        if target.exists():
            existing = pd.read_csv(target, encoding="utf-8-sig")
            combined = pd.concat([existing, gdf], ignore_index=True)
            combined = combined.sort_values(
                ["数据集名称", "指标名称", "年份"], ignore_index=True
            )
        else:
            combined = gdf

        combined.to_csv(target, index=False, encoding="utf-8-sig")
        logger.info("  %s：追加农药 %d 行，噪值修正 %d 个", str(prov).strip(), len(gdf), outliers)

    logger.info("=" * 50)
    logger.info("全部完成：31 省，噪值修正 %d 个", total_outliers)


if __name__ == "__main__":
    main()
