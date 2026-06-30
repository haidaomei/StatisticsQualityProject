#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 ../output_stats_agri_screened/ 读取所有 CSV，
剔除「指标名称」含指定关键字的行，原样输出到当前目录。

关键字：麻 棉 油料 油菜籽 烟 甘蔗 糖 花生 葵花籽 菜 药 禽 猪 畜 茶 蛋 牧 渔 林 果 采矿业
"""

import logging
import sys
from pathlib import Path
import pandas as pd

# -------------------- 配置 --------------------
SCREENED_DIR = Path(__file__).resolve().parent.parent / "output_stats_agri_screened"
OUTPUT_DIR = Path(__file__).resolve().parent

# 剔除关键字列表
EXCLUDE_KEYWORDS = [
    "麻", "棉", "油料", "油菜籽", "烟", "甘蔗", "糖",
    "花生", "葵花籽", "菜", "药", "禽", "猪", "畜",
    "茶", "蛋", "牧", "渔", "林", "果", "采矿业",
]

# -------------------- 日志 --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def should_exclude(indicator_name: str) -> bool:
    """指标名称含任一关键字则返回 True（应被剔除）"""
    if not isinstance(indicator_name, str):
        return False
    for kw in EXCLUDE_KEYWORDS:
        if kw in indicator_name:
            return True
    return False


def process_file(csv_path: Path) -> tuple[int, int]:
    """处理单个 CSV 文件，返回 (保留行数, 剔除行数)"""
    logger.info("读取：%s", csv_path.name)
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    if "指标名称" not in df.columns:
        logger.warning("跳过（缺少「指标名称」列）：%s", csv_path.name)
        return 0, 0

    total_rows = len(df)
    mask = ~df["指标名称"].apply(should_exclude)
    filtered_df = df[mask]
    removed_rows = total_rows - len(filtered_df)

    out_path = OUTPUT_DIR / csv_path.name
    filtered_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("已输出：%s（保留 %d 行，剔除 %d 行）", out_path.name, len(filtered_df), removed_rows)
    return len(filtered_df), removed_rows


def main():
    if not SCREENED_DIR.exists():
        logger.error("screened 目录不存在：%s", SCREENED_DIR)
        sys.exit(1)

    csv_files = sorted(SCREENED_DIR.glob("*.csv"))
    if not csv_files:
        logger.error("screened 目录下无 CSV 文件")
        sys.exit(1)

    logger.info("找到 %d 个 CSV 文件，开始过滤...", len(csv_files))
    total_kept = 0
    total_removed = 0

    for csv_path in csv_files:
        kept, removed = process_file(csv_path)
        total_kept += kept
        total_removed += removed

    logger.info("全部完成：共保留 %d 行，剔除 %d 行", total_kept, total_removed)


if __name__ == "__main__":
    main()
