#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证脚本：打印每个省份有多少个 unique (数据集名称, 指标名称) key。
"""

import pandas as pd
from pathlib import Path
from collections import defaultdict

ALIGNED_DIR = Path(__file__).resolve().parent

prov_keys: dict[str, set[tuple]] = defaultdict(set)
all_keys: set[tuple] = set()

for prov_dir in sorted(ALIGNED_DIR.iterdir()):
    if not prov_dir.is_dir() or prov_dir.name.startswith("."):
        continue

    for csv_path in sorted(prov_dir.glob("*.csv")):
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        if len(df) == 0:
            continue
        for _, row in df.iterrows():
            key = (str(row["数据集名称"]), str(row["指标名称"]))
            prov_keys[prov_dir.name].add(key)
            all_keys.add(key)

print(f"省份数：{len(prov_keys)}")
print(f"全局唯一 key 数：{len(all_keys)}")
print()

max_name = max(len(k) for k in prov_keys)
for prov in sorted(prov_keys):
    n = len(prov_keys[prov])
    diff = all_keys - prov_keys[prov]
    bar = "█" * n + ("░" * (len(all_keys) - n)) if len(all_keys) > 0 else ""
    print(f"  {prov:{max_name}s}  {n:3d} keys  {bar}")

# 检查是否所有省份 key 集合一致
first_keys = None
consistent = True
for prov in sorted(prov_keys):
    if first_keys is None:
        first_keys = prov_keys[prov]
    elif prov_keys[prov] != first_keys:
        consistent = False
        missing = first_keys - prov_keys[prov]
        extra = prov_keys[prov] - first_keys
        if missing:
            print(f"\n  ⚠ {prov} 缺少 {len(missing)} 个 key: {list(missing)[:5]}...")
        if extra:
            print(f"\n  ⚠ {prov} 多出 {len(extra)} 个 key: {list(extra)[:5]}...")

if consistent:
    print(f"\n✅ 所有 {len(prov_keys)} 个省份 key 集合完全一致（已统一口径）")
else:
    print(f"\n❌ 省份间 key 集合不一致！")
