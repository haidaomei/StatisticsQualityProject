#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
口径对齐脚本：从 ../output_stats_agri_divided/ 读取分省数据，
1. 按 (数据集名称, 指标名称) 分组，补全 2005-2024 缺失年份（数值填 NaN）
2. 识别系统性缺失：任一省份该 group NaN >= 10 → 从所有省份剔除该 group
3. 兜底：某省完全不存在该 group → 直接标记剔除
4. 输出到当前目录，保持省份文件夹结构

用法：python align_provinces.py
"""

import logging
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

# -------------------- 配置 --------------------
SOURCE_DIR = Path(__file__).resolve().parent.parent / "output_stats_agri_divided"
OUTPUT_DIR = Path(__file__).resolve().parent

YEAR_START = 2005
YEAR_END = 2024
ALL_YEARS: list[int] = list(range(YEAR_START, YEAR_END + 1))  # 2005..2024

NAN_THRESHOLD = 10  # 任一省份 NaN >= 此值则剔除

# -------------------- 日志 --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# -------------------- 工具函数 --------------------
def get_unit_for_group(gdf: pd.DataFrame) -> str:
    """从 group 子表中获取单位值（取第一个非空非 NaN）"""
    if "单位" not in gdf.columns:
        return ""
    units = gdf["单位"].dropna()
    if len(units) > 0:
        val = units.iloc[0]
        return str(val) if not pd.isna(val) else ""
    return ""


def collect_province_dirs(source: Path) -> list[Path]:
    """收集所有省份文件夹（排除特殊目录）"""
    dirs = sorted([
        d for d in source.iterdir()
        if d.is_dir() and not d.name.startswith(".") and d.name != "__pycache__"
    ])
    if not dirs:
        logger.error("未找到省份文件夹：%s", source)
        sys.exit(1)
    return dirs


def collect_csv_names(province_dirs: list[Path]) -> list[str]:
    """收集所有省份文件夹中出现的 CSV 文件名（取并集）"""
    names: set[str] = set()
    for prov_dir in province_dirs:
        for f in prov_dir.glob("*.csv"):
            names.add(f.name)
    return sorted(names)


def load_province_data(
    province_dirs: list[Path], csv_name: str
) -> dict[str, pd.DataFrame]:
    """
    为所有省份加载同一个 CSV 文件。
    返回值：{ 省份名: DataFrame }
    若某省无该文件，则返回空 DataFrame（含完整列名）。
    """
    COLUMNS = ["数据集名称", "指标名称", "地区", "年份", "数值", "单位"]
    result: dict[str, pd.DataFrame] = {}

    for prov_dir in province_dirs:
        csv_path = prov_dir / csv_name
        if csv_path.exists():
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
            # 统一年份为数值类型
            if "年份" in df.columns:
                df["年份"] = pd.to_numeric(df["年份"], errors="coerce")
            result[prov_dir.name] = df
        else:
            result[prov_dir.name] = pd.DataFrame(columns=COLUMNS)

    return result


def build_group_index(
    province_data: dict[str, pd.DataFrame],
) -> dict[str, dict[tuple, pd.DataFrame]]:
    """
    将每个省份的 DataFrame 按 (数据集名称, 指标名称) 拆分为子表字典。
    返回值：{ 省份名: { group_key: sub-DataFrame } }
    """
    index: dict[str, dict[tuple, pd.DataFrame]] = defaultdict(dict)

    for prov, df in province_data.items():
        if len(df) == 0:
            continue
        if "数据集名称" not in df.columns or "指标名称" not in df.columns:
            logger.warning("  [%s] 缺少关键列，跳过", prov)
            continue
        for gk, gdf in df.groupby(["数据集名称", "指标名称"], dropna=False):
            # gk 是 (数据集名称值, 指标名称值) 的元组
            index[prov][gk] = gdf

    return index


def collect_all_groups(index: dict[str, dict[tuple, pd.DataFrame]]) -> set[tuple]:
    """收集所有省份中出现过的全部 group key"""
    all_groups: set[tuple] = set()
    for prov_groups in index.values():
        all_groups.update(prov_groups.keys())
    return all_groups


def count_nan_per_group(
    all_groups: set[tuple],
    group_index: dict[str, dict[tuple, pd.DataFrame]],
    province_names: list[str],
) -> dict[tuple, dict[str, int]]:
    """
    对每个 group，统计每个省份的 NaN 数量（基于补全 2005-2024 年份后的数据）。
    返回值：{ group_key: { 省份名: NaN数量 } }
    """
    nan_counts: dict[tuple, dict[str, int]] = {}

    for gk in all_groups:
        nan_counts[gk] = {}
        for prov in province_names:
            gdf = group_index[prov].get(gk)

            if gdf is None or len(gdf) == 0:
                # 该省完全缺失此 group → 20 个 NaN
                nan_counts[gk][prov] = len(ALL_YEARS)
            else:
                existing_years: set[int] = set(
                    gdf["年份"].dropna().astype(int).tolist()
                )
                missing_years = set(ALL_YEARS) - existing_years
                existing_nan = int(gdf["数值"].isna().sum())
                nan_counts[gk][prov] = existing_nan + len(missing_years)

    return nan_counts


def find_systematic_missing(
    nan_counts: dict[tuple, dict[str, int]],
) -> set[tuple]:
    """
    识别系统性缺失：任一省份 NaN >= NAN_THRESHOLD。
    返回需要被从所有省份剔除的 group key 集合。
    """
    to_remove: set[tuple] = set()

    for gk, prov_counts in nan_counts.items():
        for prov, cnt in prov_counts.items():
            if cnt >= NAN_THRESHOLD:
                to_remove.add(gk)
                logger.warning(
                    "  ✕ 剔除 group：%s | %s（触发省：%s，NaN=%d）",
                    gk[0], gk[1], prov, cnt,
                )
                break  # 触发即标记，无需继续检查其他省

    return to_remove


def pad_and_write(
    province_data: dict[str, pd.DataFrame],
    groups_to_remove: set[tuple],
    province_names: list[str],
    csv_name: str,
):
    """
    对每个省份：过滤剔除 group、补全年份、排序、写出。
    """
    for prov in province_names:
        df = province_data[prov]
        if len(df) == 0:
            # 空 DataFrame 也写出空文件，保持省份文件夹结构一致
            prov_out_dir = OUTPUT_DIR / prov
            prov_out_dir.mkdir(parents=True, exist_ok=True)
            out_path = prov_out_dir / csv_name
            df.to_csv(out_path, index=False, encoding="utf-8-sig")
            continue

        has_key_cols = "数据集名称" in df.columns and "指标名称" in df.columns

        # Step 1: 剔除系统性缺失的 group
        if has_key_cols and groups_to_remove:
            mask = ~df.apply(
                lambda row, _remove=groups_to_remove:
                    (row["数据集名称"], row["指标名称"]) in _remove,
                axis=1,
            )
            df = df.loc[mask].copy()

        # Step 2: 补全保留 group 的缺失年份 (2005-2024)
        if has_key_cols and len(df) > 0:
            padded_rows: list[dict] = []
            for gk, gdf in df.groupby(["数据集名称", "指标名称"], dropna=False):
                existing_years: set[int] = set(
                    gdf["年份"].dropna().astype(int).tolist()
                )
                unit = get_unit_for_group(gdf)

                for yr in ALL_YEARS:
                    if yr not in existing_years:
                        padded_rows.append({
                            "数据集名称": gk[0] if isinstance(gk, tuple) else gk,
                            "指标名称": gk[1] if isinstance(gk, tuple) else gk,
                            "地区": prov,
                            "年份": yr,
                            "数值": np.nan,
                            "单位": unit,
                        })

            if padded_rows:
                pad_df = pd.DataFrame(padded_rows)
                df = pd.concat([df, pad_df], ignore_index=True)

        # Step 3: 排序
        if len(df) > 0:
            sort_cols = ["数据集名称", "指标名称", "年份"]
            available = [c for c in sort_cols if c in df.columns]
            if available:
                df = df.sort_values(available, ignore_index=True)

        # Step 4: 写出
        prov_out_dir = OUTPUT_DIR / prov
        prov_out_dir.mkdir(parents=True, exist_ok=True)
        out_path = prov_out_dir / csv_name
        df.to_csv(out_path, index=False, encoding="utf-8-sig")

    logger.info("  已写出 %d 个省份", len(province_names))


# -------------------- 主流程 --------------------
def main():
    if not SOURCE_DIR.exists():
        logger.error("数据源目录不存在：%s", SOURCE_DIR)
        sys.exit(1)

    # 1. 收集省份与 CSV 文件
    province_dirs = collect_province_dirs(SOURCE_DIR)
    province_names = [d.name for d in province_dirs]
    logger.info(
        "找到 %d 个省份：%s ...",
        len(province_names),
        ", ".join(province_names[:6]),
    )

    csv_names = collect_csv_names(province_dirs)
    logger.info("共 %d 种 CSV 文件（跨省去重）", len(csv_names))

    total_groups_all = 0
    total_removed_all = 0

    # 2. 逐个 CSV 处理
    for csv_name in csv_names:
        logger.info("=" * 60)
        logger.info("处理：%s", csv_name)

        # 2a. 加载所有省份的该 CSV
        province_data = load_province_data(province_dirs, csv_name)

        # 2b. 按 group key 建立索引
        group_index = build_group_index(province_data)

        # 2c. 收集全部 group key
        all_groups = collect_all_groups(group_index)
        if not all_groups:
            logger.info("  无有效 group，跳过")
            continue

        # 2d. 统计每个 group 在每个省的 NaN 数量
        nan_counts = count_nan_per_group(all_groups, group_index, province_names)

        # 2e. 识别系统性缺失
        groups_to_remove = find_systematic_missing(nan_counts)

        kept = len(all_groups) - len(groups_to_remove)
        logger.info(
            "  汇总：%d 个 group → 剔除 %d，保留 %d",
            len(all_groups), len(groups_to_remove), kept,
        )

        total_groups_all += len(all_groups)
        total_removed_all += len(groups_to_remove)

        # 2f. 输出对齐后的文件
        pad_and_write(province_data, groups_to_remove, province_names, csv_name)

    # 3. 收尾统计
    logger.info("=" * 60)
    logger.info(
        "全部完成：%d 个 CSV，共 %d 个 group，剔除 %d 个，保留 %d 个",
        len(csv_names),
        total_groups_all,
        total_removed_all,
        total_groups_all - total_removed_all,
    )


if __name__ == "__main__":
    main()
