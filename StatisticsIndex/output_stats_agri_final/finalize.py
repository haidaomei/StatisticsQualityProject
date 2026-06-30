#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终处理脚本：
从 ../output_stats_agri_剑锋噪值/ 读取分省数据，
1. 所有负数值 → 改为 0
2. 单位为 "个" / "台" / "部" 的 → 向下取整 (floor)
输出到当前目录，保持省份文件夹结构。

用法：python finalize.py
"""

import logging
import sys
import math
from pathlib import Path

import pandas as pd

# -------------------- 配置 --------------------
SOURCE_DIR = Path(__file__).resolve().parent.parent / "output_stats_agri_剑锋噪值"
OUTPUT_DIR = Path(__file__).resolve().parent

FLOOR_UNITS = {"个", "台", "部"}

# -------------------- 日志 --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def finalize_value(val, unit: str):
    """对单个数值做最终处理：负数→0，计数单位→floor"""
    if pd.isna(val):
        return val
    if val < 0:
        return 0.0
    if unit in FLOOR_UNITS:
        return float(math.floor(val))
    return val


def main():
    if not SOURCE_DIR.exists():
        logger.error("数据源目录不存在：%s", SOURCE_DIR)
        sys.exit(1)

    province_dirs = sorted([
        d for d in SOURCE_DIR.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])
    if not province_dirs:
        logger.error("未找到省份文件夹")
        sys.exit(1)

    total_neg = 0
    total_floored = 0

    for prov_dir in province_dirs:
        prov_out = OUTPUT_DIR / prov_dir.name
        prov_out.mkdir(parents=True, exist_ok=True)

        prov_neg = 0
        prov_floored = 0

        for csv_path in sorted(prov_dir.glob("*.csv")):
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
            if "数值" not in df.columns or "单位" not in df.columns:
                df.to_csv(prov_out / csv_path.name, index=False, encoding="utf-8-sig")
                continue

            # 统计负数
            prov_neg += int((df["数值"] < 0).sum())

            # 统计需 floor 的行
            prov_floored += int(
                df.apply(
                    lambda r: str(r["单位"]).strip() in FLOOR_UNITS
                    and not pd.isna(r["数值"]),
                    axis=1,
                ).sum()
            )

            # 应用修正
            df["数值"] = df.apply(
                lambda r: finalize_value(r["数值"], str(r["单位"]).strip()),
                axis=1,
            )

            df.to_csv(prov_out / csv_path.name, index=False, encoding="utf-8-sig")

        total_neg += prov_neg
        total_floored += prov_floored
        logger.info("  %s：负数→0  %d 个 | floor  %d 个", prov_dir.name, prov_neg, prov_floored)

    logger.info("=" * 50)
    logger.info("全部完成：负数修正 %d 个，向下取整 %d 行", total_neg, total_floored)


if __name__ == "__main__":
    main()
