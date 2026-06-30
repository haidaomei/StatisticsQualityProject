#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 ../output_stats_agri_filtered/ 读取所有 CSV，
按「地区」列拆分为省份文件夹，每个文件夹内存该省份专属的 CSV。
"""

import logging
import sys
from pathlib import Path
import pandas as pd

# -------------------- 配置 --------------------
SOURCE_DIR = Path(__file__).resolve().parent.parent / "output_stats_agri_filtered"
OUTPUT_DIR = Path(__file__).resolve().parent

# -------------------- 日志 --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def sanitize_dirname(name: str) -> str:
    """清理省份名，使其可作为文件夹名"""
    invalid = '<>:"/\\|?*\n\r\t'
    return "".join(c for c in str(name) if c not in invalid).strip()


def main():
    if not SOURCE_DIR.exists():
        logger.error("数据源目录不存在：%s", SOURCE_DIR)
        sys.exit(1)

    csv_files = [p for p in sorted(SOURCE_DIR.glob("*.csv")) if p.name != "README.txt"]
    if not csv_files:
        logger.error("数据源目录下无 CSV 文件")
        sys.exit(1)

    logger.info("数据源：%d 个 CSV", len(csv_files))

    province_counts: dict[str, int] = {}
    file_count = 0

    for csv_path in csv_files:
        logger.info("处理：%s", csv_path.name)
        df = pd.read_csv(csv_path, encoding="utf-8-sig")

        if "地区" not in df.columns:
            logger.warning("  跳过（缺少「地区」列）")
            continue

        # 按地区分组
        for province, group in df.groupby("地区"):
            province_dir_name = sanitize_dirname(str(province))
            if not province_dir_name:
                continue

            province_dir = OUTPUT_DIR / province_dir_name
            province_dir.mkdir(parents=True, exist_ok=True)

            out_path = province_dir / csv_path.name
            group.to_csv(out_path, index=False, encoding="utf-8-sig")

            province_counts[province_dir_name] = province_counts.get(province_dir_name, 0) + 1

        file_count += 1

    logger.info("完成：%d 个 CSV → %d 个省份文件夹", file_count, len(province_counts))
    for prov in sorted(province_counts.keys()):
        logger.info("  %s：%d 个表", prov, province_counts[prov])


if __name__ == "__main__":
    main()
