#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将合并的 agri_indicators_all_*.csv 按 `数据集名称` 拆分为单独 CSV。
输出目录：output_stats_agri/splits/

用法：
    python split_export.py
可选参数：
    --per-indicator  同时为每个指标生成单独 CSV（保存在子目录中）
"""
import argparse
import logging
from pathlib import Path
import sys
import time
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output_stats_agri")
SPLITS_DIR = OUTPUT_DIR / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_filename(s: str) -> str:
    invalid = '<>:"/\\|?*\n\r\t'
    res = "".join(c for c in s if c not in invalid)
    return res.strip().replace(" ", "_")[:240]


def find_latest_combined_csv(base_dir: Path) -> Path:
    files = list(base_dir.glob("agri_indicators_all_*.csv"))
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


def split_by_dataset(input_csv: Path, per_indicator: bool = False):
    logger.info("读取合并文件：%s", input_csv)
    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    if "数据集名称" not in df.columns:
        logger.error("CSV 缺少列：数据集名称。无法拆分。")
        return

    groups = df.groupby("数据集名称")
    total_files = 0
    total_rows = 0
    for name, g in groups:
        safe = sanitize_filename(name)
        out_path = SPLITS_DIR / f"{safe}.csv"
        g.to_csv(out_path, index=False, encoding="utf-8-sig")
        logger.info("已保存数据集：%s -> %s （%d 行）", name, out_path, len(g))
        total_files += 1
        total_rows += len(g)

        if per_indicator:
            subdir = SPLITS_DIR / safe
            subdir.mkdir(parents=True, exist_ok=True)
            for iname, ig in g.groupby("指标名称"):
                safe_i = sanitize_filename(iname)
                i_path = subdir / f"{safe}_{safe_i}.csv"
                ig.to_csv(i_path, index=False, encoding="utf-8-sig")
    logger.info("拆分完成：%d 个数据集文件，共 %d 行（含表头）", total_files, total_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-indicator", action="store_true", help="同时为每个指标生成单独 CSV")
    args = parser.parse_args()

    csv_path = find_latest_combined_csv(OUTPUT_DIR)
    if not csv_path:
        logger.error("未找到合并 CSV（agri_indicators_all_*.csv）在 %s 下", OUTPUT_DIR)
        sys.exit(1)

    split_by_dataset(csv_path, per_indicator=args.per_indicator)


if __name__ == "__main__":
    main()
