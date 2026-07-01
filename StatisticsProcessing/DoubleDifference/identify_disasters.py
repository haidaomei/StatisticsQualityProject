#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特大灾害识别脚本：
1. 每省计算 农作物受灾面积 的 mean + 1.5σ（无偏），标记特大灾害年份
2. 统计每年受灾省份数，≥ ceil(31/2)=16 → 全国性灾害，排除
3. 输出保留的单省单年特大灾害列表，供后续 DID 分析

用法：python identify_disasters.py
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# -------------------- 配置 --------------------
FINAL_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "StatisticsIndex" / "output_stats_agri_final"
)
OUTPUT_DIR = Path(__file__).resolve().parent

SIGMA = 1.5
NATIONAL_THRESHOLD = 16  # ceil(31/2)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    # ---- 1. 加载所有省份的受灾面积 ----
    records = []
    for prov_dir in sorted(FINAL_DIR.iterdir()):
        if not prov_dir.is_dir() or prov_dir.name.startswith("."):
            continue
        csv_path = prov_dir / "自然灾害_分省_2005YY-2024YY.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        disaster = df[df["指标名称"].str.strip() == "农作物受灾面积 (千公顷)"].copy()
        if len(disaster) == 0:
            continue
        disaster["省份"] = prov_dir.name
        records.append(disaster[["省份", "年份", "数值"]])

    full = pd.concat(records, ignore_index=True)
    logger.info("加载 %d 省 × %d 年 = %d 条纪录", full["省份"].nunique(), full["年份"].nunique(), len(full))

    # ---- 2. 每省识别特大灾害 ----
    province_disasters: dict[str, list[int]] = {}
    for prov, gdf in full.groupby("省份"):
        gdf = gdf.sort_values("年份")
        values = gdf["数值"].to_numpy(dtype=float)
        mean_val = float(np.nanmean(values))
        std_val = float(np.nanstd(values, ddof=1))
        threshold = mean_val + SIGMA * std_val

        disaster_years = sorted(
            gdf.loc[values > threshold, "年份"].astype(int).tolist()
        )
        province_disasters[prov] = disaster_years

        if disaster_years:
            logger.info(
                "  %s  mean=%.1f  σ=%.1f  threshold=%.1f  → 特大灾害年: %s",
                prov, mean_val, std_val, threshold, disaster_years,
            )

    # ---- 3. 统计每年受灾省份数 ----
    year_counts: dict[int, int] = {}
    for prov, years in province_disasters.items():
        for yr in years:
            year_counts[yr] = year_counts.get(yr, 0) + 1

    logger.info("=" * 60)
    logger.info("每年特大灾害省份数：")
    excluded_years = []
    for yr in sorted(year_counts.keys()):
        cnt = year_counts[yr]
        flag = "  ← 全国性灾害，排除" if cnt >= NATIONAL_THRESHOLD else ""
        logger.info("  %d 年：%d 省受灾%s", yr, cnt, flag)
        if cnt >= NATIONAL_THRESHOLD:
            excluded_years.append(yr)

    # ---- 4. 输出保留的 DID 事件 ----
    did_events = []
    for prov, years in province_disasters.items():
        for yr in years:
            if yr not in excluded_years:
                did_events.append((prov, yr))

    logger.info("=" * 60)
    logger.info("排除的全国性灾害年份：%s", excluded_years if excluded_years else "无")
    logger.info("保留的单省特大灾害事件：%d 个", len(did_events))
    for prov, yr in sorted(did_events):
        logger.info("  %s %d", prov, yr)

    # ---- 保存 ----
    events_df = pd.DataFrame(did_events, columns=["省份", "年份"])
    events_df.to_csv(OUTPUT_DIR / "did_events.csv", index=False, encoding="utf-8-sig")
    logger.info("DID 事件列表已保存：%s", OUTPUT_DIR / "did_events.csv")

    if excluded_years:
        pd.DataFrame({"年份": excluded_years}).to_csv(
            OUTPUT_DIR / "excluded_years.csv", index=False, encoding="utf-8-sig"
        )
        logger.info("排除年份已保存：%s", OUTPUT_DIR / "excluded_years.csv")

    logger.info("全部完成。")


if __name__ == "__main__":
    main()
