#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全国指标热力图：
从 ../StatisticsIndex/output_stats_agri_final/ 读取所有省份数据，
每个 (指标名称, 年份) 取全国平均，绘制热力图。

用法：python heatmap.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from pathlib import Path

# -------------------- 配置 --------------------
SOURCE_DIR = Path(__file__).resolve().parent.parent.parent / "StatisticsIndex" / "output_stats_agri_final"
OUTPUT_DIR = Path(__file__).resolve().parent

# 中文字体
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

# -------------------- 主流程 --------------------
def main():
    # 1. 收集所有省份数据
    all_rows = []
    for prov_dir in sorted(SOURCE_DIR.iterdir()):
        if not prov_dir.is_dir() or prov_dir.name.startswith("."):
            continue
        for csv_path in sorted(prov_dir.glob("*.csv")):
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
            all_rows.append(df)

    full_df = pd.concat(all_rows, ignore_index=True)
    print(f"总行数：{len(full_df)}")

    # 2. 全国平均：groupby (指标名称, 年份) → mean(数值)
    national_avg = (
        full_df.groupby(["指标名称", "年份"])["数值"]
        .mean()
        .reset_index()
    )
    print(f"聚合后 (指标, 年份) 组合数：{len(national_avg)}")

    # 3. 透视成热力图矩阵：行=指标, 列=年份
    heatmap_data = national_avg.pivot(
        index="指标名称", columns="年份", values="数值"
    )
    print(f"热力图矩阵：{heatmap_data.shape[0]} 指标 × {heatmap_data.shape[1]} 年")

    # 4. 标准化（按行 z-score，方便看出相对高低）
    # 保留原始值用于标注，热力图颜色用 z-score
    heatmap_z = heatmap_data.subtract(heatmap_data.mean(axis=1), axis=0).div(
        heatmap_data.std(axis=1), axis=0
    )

    # 5. 绘图
    n_indicators = heatmap_data.shape[0]
    fig_height = max(6, n_indicators * 0.45)
    fig, ax = plt.subplots(figsize=(18, fig_height))

    sns.heatmap(
        heatmap_z,
        annot=heatmap_data.round(1),   # 标注原始均值
        fmt=".1f",
        cmap="RdYlGn",
        center=0,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Z-score (行标准化)"},
        ax=ax,
        annot_kws={"fontsize": 6},
    )

    ax.set_title("全国农业指标热力图（2005-2024 各省均值）", fontsize=14, pad=12)
    ax.set_xlabel("年份")
    ax.set_ylabel("指标名称")
    ax.tick_params(axis="both", labelsize=7)

    plt.tight_layout()

    out_path = OUTPUT_DIR / "national_heatmap.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\n热力图已保存：{out_path}")
    plt.close()


if __name__ == "__main__":
    main()
