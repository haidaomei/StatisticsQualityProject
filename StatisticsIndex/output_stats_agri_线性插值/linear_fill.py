#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
线性插值 + 线性回归外推脚本：
从 ../output_stats_agri_aligned/ 读取分省数据，
1. 对每个 group 的闭区间（首尾非NaN之间）进行线性插值
2. 对闭区间外的首尾 NaN 用 OLS 线性回归外推
3. 输出到当前目录，保持省份文件夹结构

用法：python linear_fill.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# -------------------- 配置 --------------------
SOURCE_DIR = Path(__file__).resolve().parent.parent / "output_stats_agri_aligned"
OUTPUT_DIR = Path(__file__).resolve().parent

YEAR_START = 2005
YEAR_END = 2024
ALL_YEARS: np.ndarray = np.arange(YEAR_START, YEAR_END + 1, dtype=float)  # shape (20,)

# -------------------- 日志 --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# -------------------- 工具函数 --------------------
def collect_province_dirs(source: Path) -> list[Path]:
    dirs = sorted([
        d for d in source.iterdir()
        if d.is_dir() and not d.name.startswith(".") and d.name != "__pycache__"
    ])
    if not dirs:
        logger.error("未找到省份文件夹：%s", source)
        sys.exit(1)
    return dirs


def collect_csv_names(province_dirs: list[Path]) -> list[str]:
    names: set[str] = set()
    for prov_dir in province_dirs:
        for f in prov_dir.glob("*.csv"):
            names.add(f.name)
    return sorted(names)


def fill_group(years: np.ndarray, values: np.ndarray) -> np.ndarray:
    """
    对单个 group 的时间序列做线性插值 + 外推。
    years: 年份数组 (已排序, float)
    values: 对应数值数组 (含 NaN)
    返回: 填充后的数值数组 (无 NaN)
    """
    filled = values.copy()
    n = len(values)

    # --- 找到首尾非 NaN 位置 ---
    valid_mask = ~np.isnan(values)
    if not valid_mask.any():
        # 全部 NaN —— 理论上不应出现，但兜底：全填 0
        return np.zeros(n)

    valid_idx = np.where(valid_mask)[0]
    first = valid_idx[0]
    last = valid_idx[-1]

    # --- Step 1: 闭区间内线性插值 ---
    if first < last:
        x_valid = years[first : last + 1][valid_mask[first : last + 1]]
        y_valid = values[first : last + 1][valid_mask[first : last + 1]]
        x_interp = years[first : last + 1]
        filled[first : last + 1] = np.interp(x_interp, x_valid, y_valid)

    # --- Step 2 & 3: 首尾外推（用闭区间全部数据做 OLS） ---
    need_extrapolate_head = first > 0
    need_extrapolate_tail = last < n - 1

    if need_extrapolate_head or need_extrapolate_tail:
        # 用闭区间全部数据（含插值后）做 y = a*x + b 回归（纯 numpy OLS）
        x_train = years[first : last + 1]
        y_train = filled[first : last + 1]

        if len(x_train) >= 2:
            # OLS: a = cov(x,y)/var(x), b = mean(y) - a*mean(x)
            a = np.cov(x_train, y_train)[0, 1] / np.var(x_train)
            b = np.mean(y_train) - a * np.mean(x_train)

            if need_extrapolate_head:
                filled[:first] = a * years[:first] + b

            if need_extrapolate_tail:
                filled[last + 1:] = a * years[last + 1:] + b
        else:
            # 只有 1 个点，用该值填充首尾
            fill_val = filled[first]
            if need_extrapolate_head:
                filled[:first] = fill_val
            if need_extrapolate_tail:
                filled[last + 1:] = fill_val

    return filled


def process_province(
    prov_dir: Path,
    csv_names: list[str],
    prov_name: str,
):
    """处理单个省份的所有 CSV"""
    prov_out = OUTPUT_DIR / prov_name
    prov_out.mkdir(parents=True, exist_ok=True)

    for csv_name in csv_names:
        csv_path = prov_dir / csv_name
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        if len(df) == 0:
            df.to_csv(prov_out / csv_name, index=False, encoding="utf-8-sig")
            continue

        # 确保年份为数值类型
        if "年份" in df.columns:
            df["年份"] = pd.to_numeric(df["年份"], errors="coerce")

        # 按 group 处理
        if "数据集名称" in df.columns and "指标名称" in df.columns:
            result_parts = []
            for gk, gdf in df.groupby(["数据集名称", "指标名称"], dropna=False):
                gdf = gdf.sort_values("年份").reset_index(drop=True)
                years_arr = gdf["年份"].to_numpy(dtype=float)
                values_arr = gdf["数值"].to_numpy(dtype=float)
                filled_values = fill_group(years_arr, values_arr)
                gdf["数值"] = filled_values
                result_parts.append(gdf)

            if result_parts:
                df = pd.concat(result_parts, ignore_index=True)
                # 排回 group→年份 顺序
                df = df.sort_values(["数据集名称", "指标名称", "年份"], ignore_index=True)

        df.to_csv(prov_out / csv_name, index=False, encoding="utf-8-sig")

    logger.info("  已处理：%s（%d 个 CSV）", prov_name, len(csv_names))


def main():
    if not SOURCE_DIR.exists():
        logger.error("数据源目录不存在：%s", SOURCE_DIR)
        sys.exit(1)

    province_dirs = collect_province_dirs(SOURCE_DIR)
    csv_names = collect_csv_names(province_dirs)

    logger.info("省份数：%d，CSV 种类：%d", len(province_dirs), len(csv_names))
    logger.info("-" * 50)

    for prov_dir in province_dirs:
        process_province(prov_dir, csv_names, prov_dir.name)

    logger.info("=" * 50)
    logger.info("全部完成：%d 个省份，%d 种 CSV", len(province_dirs), len(csv_names))


if __name__ == "__main__":
    main()
