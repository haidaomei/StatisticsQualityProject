#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相关性热力图：
从 ../StatisticsIndex/output_stats_agri_final/ 读取所有省份数据，
每 (指标, 年份) 全国平均 → 58×20 矩阵，
计算 Pearson 和 Spearman 相关系数（58×58），左右并排一张图输出。

用法：python correlation_heatmap.py
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

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def main():
    # 1. 加载 + 聚合全国均值
    all_rows = []
    for prov_dir in sorted(SOURCE_DIR.iterdir()):
        if not prov_dir.is_dir() or prov_dir.name.startswith("."):
            continue
        for csv_path in sorted(prov_dir.glob("*.csv")):
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
            all_rows.append(df)

    full_df = pd.concat(all_rows, ignore_index=True)

    national_avg = (
        full_df.groupby(["指标名称", "年份"])["数值"]
        .mean()
        .reset_index()
    )

    # 透视：行=指标, 列=年份 → 58×20
    mat = national_avg.pivot(index="指标名称", columns="年份", values="数值")
    print(f"矩阵：{mat.shape[0]} 指标 × {mat.shape[1]} 年")

    # 2. 计算相关系数矩阵
    # 注意：corr 在 pandas 中是列与列之间的相关，我们要指标间的 → 先转置
    # 指标在行，年份在列 → corr 默认算列相关 = 年份间相关，不是我们要的
    # 所以我们要算行之间的相关 → 用 mat.T.corr()? 不对
    # mat: index=指标, columns=年份 → df.corr() 是列(年份)之间相关
    # 我们需要行(指标)之间相关 → 直接用 numpy: corrcoef 或 mat.T 然后 corr
    # 简单方法：指标间相关 = np.corrcoef(mat.values) → 58×58

    pearson_mat = np.corrcoef(mat.values)  # 行间 Pearson
    spearman_mat = mat.T.corr(method="spearman").values  # 转置后列间 = 原行间

    # 3. 绘图：左右并排
    n = mat.shape[0]
    fig_height = max(12, n * 0.28)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2 * fig_height + 2, fig_height))

    labels = mat.index.tolist()

    for ax, data, title in [
        (ax1, pearson_mat, "Pearson 相关系数"),
        (ax2, spearman_mat, "Spearman 秩相关系数"),
    ]:
        mask = np.triu(np.ones_like(data, dtype=bool), k=1)
        sns.heatmap(
            data,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            center=0,
            linewidths=0.3,
            linecolor="white",
            square=True,
            cbar_kws={"shrink": 0.7, "label": "相关系数"},
            ax=ax,
            annot_kws={"fontsize": 5},
            xticklabels=labels,
            yticklabels=labels,
        )
        ax.set_title(title, fontsize=13, pad=8)
        ax.tick_params(axis="both", labelsize=5)

    fig.suptitle("全国农业指标相关性热力图（2005-2024 各省均值）", fontsize=15, y=1.01)
    plt.tight_layout()

    out_path = OUTPUT_DIR / "correlation_heatmap.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"已保存：{out_path}")
    plt.close()


if __name__ == "__main__":
    main()
