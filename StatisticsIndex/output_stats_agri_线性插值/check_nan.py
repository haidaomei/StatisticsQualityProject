#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证脚本：遍历当前目录下所有省份文件夹的 CSV，
检查是否还存在 NaN，并给出统计报告。
"""

import pandas as pd
import numpy as np
from pathlib import Path

ALIGNED_DIR = Path(__file__).resolve().parent

total_files = 0
total_cells = 0
total_nan = 0
bad_files: list[tuple[str, str, int]] = []  # (省份, 文件名, NaN数)

for prov_dir in sorted(ALIGNED_DIR.iterdir()):
    if not prov_dir.is_dir() or prov_dir.name.startswith("."):
        continue

    for csv_path in sorted(prov_dir.glob("*.csv")):
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        total_files += 1

        if len(df) == 0:
            continue

        if "数值" in df.columns:
            nan_count = int(df["数值"].isna().sum())
            total_cells += len(df)
            total_nan += nan_count
            if nan_count > 0:
                bad_files.append((prov_dir.name, csv_path.name, nan_count))

if total_nan == 0:
    print(f"✅ 全部通过：{total_files} 个文件，{total_cells} 个数值单元格，0 个 NaN")
else:
    print(f"❌ 发现 NaN：{total_files} 个文件中 {len(bad_files)} 个有问题")
    print(f"   总 NaN 数：{total_nan} / {total_cells} 个数值单元格")
    print()
    for prov, fname, cnt in bad_files[:30]:
        print(f"   {prov}/{fname}  →  {cnt} NaN")
    if len(bad_files) > 30:
        print(f"   ... 共 {len(bad_files)} 个文件有问题")
