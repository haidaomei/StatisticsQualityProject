#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
剑锋噪值去除脚本：
从 ../output_stats_agri_线性插值/ 读取分省数据，
1. 对每个 group 按 3σ 原则（双侧，无偏 std）标记异常值 → 抹为 NaN
2. 重新执行线性插值 + OLS 外推填充
3. 输出到当前目录，保持省份文件夹结构

用法：python denoise.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# -------------------- 配置 --------------------
SOURCE_DIR = Path(__file__).resolve().parent.parent / "output_stats_agri_线性插值"
OUTPUT_DIR = Path(__file__).resolve().parent

YEAR_START = 2005
YEAR_END = 2024
ALL_YEARS: np.ndarray = np.arange(YEAR_START, YEAR_END + 1, dtype=float)

SIGMA_MULTIPLIER = 3  # 3σ 原则

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
    线性插值 + OLS 外推，与 linear_fill.py 相同逻辑。
    """
    filled = values.copy()
    n = len(values)

    valid_mask = ~np.isnan(values)
    if not valid_mask.any():
        return np.zeros(n)

    valid_idx = np.where(valid_mask)[0]
    first = valid_idx[0]
    last = valid_idx[-1]

    # 闭区间内线性插值
    if first < last:
        x_valid = years[first:last + 1][valid_mask[first:last + 1]]
        y_valid = values[first:last + 1][valid_mask[first:last + 1]]
        x_interp = years[first:last + 1]
        filled[first:last + 1] = np.interp(x_interp, x_valid, y_valid)

    # 首尾 OLS 外推
    need_head = first > 0
    need_tail = last < n - 1

    if need_head or need_tail:
        x_train = years[first:last + 1]
        y_train = filled[first:last + 1]

        if len(x_train) >= 2:
            a = np.cov(x_train, y_train)[0, 1] / np.var(x_train)
            b = np.mean(y_train) - a * np.mean(x_train)

            if need_head:
                filled[:first] = a * years[:first] + b
            if need_tail:
                filled[last + 1:] = a * years[last + 1:] + b
        else:
            fill_val = filled[first]
            if need_head:
                filled[:first] = fill_val
            if need_tail:
                filled[last + 1:] = fill_val

    return filled


def process_province(prov_dir: Path, csv_names: list[str], prov_name: str):
    """处理单个省份的所有 CSV"""
    prov_out = OUTPUT_DIR / prov_name
    prov_out.mkdir(parents=True, exist_ok=True)

    total_outliers = 0

    for csv_name in csv_names:
        csv_path = prov_dir / csv_name
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        if len(df) == 0:
            df.to_csv(prov_out / csv_name, index=False, encoding="utf-8-sig")
            continue

        if "年份" in df.columns:
            df["年份"] = pd.to_numeric(df["年份"], errors="coerce")

        if "数据集名称" in df.columns and "指标名称" in df.columns:
            result_parts = []
            for gk, gdf in df.groupby(["数据集名称", "指标名称"], dropna=False):
                gdf = gdf.sort_values("年份").reset_index(drop=True)
                values_arr = gdf["数值"].to_numpy(dtype=float)

                # --- 计算无偏标准差 (ddof=1) ---
                std_val = float(np.nanstd(values_arr, ddof=1))
                mean_val = float(np.nanmean(values_arr))

                if std_val > 0:
                    # 双侧 3σ 标记异常值
                    lower = mean_val - SIGMA_MULTIPLIER * std_val
                    upper = mean_val + SIGMA_MULTIPLIER * std_val
                    mask_outlier = (values_arr < lower) | (values_arr > upper)
                    n_outliers = int(mask_outlier.sum())

                    if n_outliers > 0:
                        total_outliers += n_outliers
                        # 抹为 NaN 后重新填充
                        cleaned = values_arr.copy()
                        cleaned[mask_outlier] = np.nan
                        filled = fill_group(gdf["年份"].to_numpy(dtype=float), cleaned)
                        gdf["数值"] = filled
                    # std=0 或 无异常值 → 原样保留
                # std == 0 → 保留原样

                result_parts.append(gdf)

            if result_parts:
                df = pd.concat(result_parts, ignore_index=True)
                df = df.sort_values(["数据集名称", "指标名称", "年份"], ignore_index=True)

        df.to_csv(prov_out / csv_name, index=False, encoding="utf-8-sig")

    logger.info("  已处理：%s（移除噪值 %d 个）", prov_name, total_outliers)


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
