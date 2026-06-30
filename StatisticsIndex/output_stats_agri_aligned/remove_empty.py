#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理 aligned 目录下各省份文件夹中的空 CSV（仅有表头无数据行）。
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ALIGNED_DIR = Path(__file__).resolve().parent

total_removed = 0

for prov_dir in sorted(ALIGNED_DIR.iterdir()):
    if not prov_dir.is_dir() or prov_dir.name.startswith("."):
        continue

    for csv_path in sorted(prov_dir.glob("*.csv")):
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            lines = f.readlines()

        if len(lines) <= 1:  # 只有 header 或更少
            csv_path.unlink()
            logger.info("已删除：%s/%s", prov_dir.name, csv_path.name)
            total_removed += 1

logger.info("完成：共删除 %d 个空 CSV 文件", total_removed)
