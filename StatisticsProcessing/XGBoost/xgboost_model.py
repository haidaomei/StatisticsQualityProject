#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost 时序预测：7 IV + 单产lag1-3 + 受灾lag1-3 → 预测单产
训练: 2008-2019 | 测试: 2020-2024
省份 One-hot 编码。

用法：python xgboost_model.py
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

# -------------------- 配置 --------------------
FINAL_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "StatisticsIndex" / "output_stats_agri_final"
)
OUTPUT_DIR = Path(__file__).resolve().parent

INDICATOR_MAP = {
    "粮食单位面积产量 (公斤/公顷)": "单产",
    "农业机械总动力 (万千瓦)": "农机动力",
    "农用塑料薄膜使用量 (吨)": "薄膜",
    "农药使用量 (万吨)": "农药",
    "农村用电量 (亿千瓦小时)": "用电",
    "农用化肥施用折纯量 (万吨)": "化肥",
    "有效灌溉面积 (千公顷)": "灌溉",
    "农作物受灾面积 (千公顷)": "受灾",
}

DV = "单产"
IVS = ["农机动力", "薄膜", "农药", "用电", "化肥", "灌溉", "受灾"]
LAG_VARS = ["单产", "受灾"]  # 做 lag 的变量
LAG_ORDERS = [1, 2, 3]

TRAIN_END = 2019
TEST_START = 2020
TEST_END = 2024

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_panel():
    rows = []
    for prov_dir in sorted(FINAL_DIR.iterdir()):
        if not prov_dir.is_dir() or prov_dir.name.startswith("."):
            continue
        prov_name = prov_dir.name
        for csv_path in sorted(prov_dir.glob("*.csv")):
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
            if "指标名称" not in df.columns:
                continue
            matched = df[df["指标名称"].str.strip().isin(INDICATOR_MAP.keys())].copy()
            if len(matched) == 0:
                continue
            matched["short_name"] = matched["指标名称"].str.strip().map(INDICATOR_MAP)
            matched["省份"] = prov_name
            matched.rename(columns={"年份": "年份", "数值": "值"}, inplace=True)
            rows.append(matched[["省份", "年份", "short_name", "值"]])
    full = pd.concat(rows, ignore_index=True)
    panel = full.pivot_table(
        index=["省份", "年份"], columns="short_name", values="值", aggfunc="first"
    )
    return panel.sort_index()


def add_lags(df):
    """按省份分组，添加 lag 特征"""
    df = df.reset_index()
    lag_cols = []
    for prov, g in df.groupby("省份"):
        g = g.sort_values("年份")
        for var in LAG_VARS:
            for lag in LAG_ORDERS:
                col_name = f"{var}_lag{lag}"
                g[col_name] = g[var].shift(lag)
        lag_cols.extend(g.columns.difference(df.columns).tolist())
        df.loc[g.index, g.columns] = g
    df = df.set_index(["省份", "年份"]).sort_index()
    return df


def main():
    logger.info("加载数据...")
    panel = load_panel()
    panel = add_lags(panel)

    # One-hot 省份
    panel = pd.get_dummies(panel, columns=[], prefix="", prefix_sep="")
    panel = panel.reset_index()
    prov_dummies = pd.get_dummies(panel["省份"], prefix="prov")
    panel = pd.concat([panel, prov_dummies], axis=1)
    panel = panel.set_index(["省份", "年份"]).sort_index()

    # 年份同时作为列加入（索引保留用于时间切分）
    panel["年份"] = panel.index.get_level_values("年份").astype(int)

    # Drop 缺失（lag 造成的 2005-2007）
    lag_cols = [f"{v}_lag{l}" for v in LAG_VARS for l in LAG_ORDERS]
    feature_cols = ["年份"] + IVS + lag_cols + [c for c in prov_dummies.columns]
    all_needed = [DV] + feature_cols
    df_model = panel[all_needed].dropna()

    logger.info("建模数据：%d 行 × %d 列", len(df_model), len(df_model.columns))
    logger.info("年份范围：%d - %d", df_model.index.get_level_values(1).min(), df_model.index.get_level_values(1).max())

    # 时间切分
    train_mask = df_model.index.get_level_values("年份") <= TRAIN_END
    test_mask = df_model.index.get_level_values("年份") >= TEST_START

    X_train = df_model.loc[train_mask, feature_cols]
    y_train = df_model.loc[train_mask, DV]
    X_test = df_model.loc[test_mask, feature_cols]
    y_test = df_model.loc[test_mask, DV]

    logger.info("训练集：%d 行 (%d-%d)", len(X_train),
                df_model.loc[train_mask].index.get_level_values("年份").min(),
                df_model.loc[train_mask].index.get_level_values("年份").max())
    logger.info("测试集：%d 行 (%d-%d)", len(X_test),
                df_model.loc[test_mask].index.get_level_values("年份").min(),
                df_model.loc[test_mask].index.get_level_values("年份").max())

    # 训练 XGBoost
    logger.info("=" * 60)
    logger.info("训练 XGBoost...")

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # 预测
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # 评估
    logger.info("=" * 60)
    logger.info("训练集评估：")
    logger.info("  R² = %.4f | MAE = %.4f | RMSE = %.4f",
                r2_score(y_train, y_pred_train),
                mean_absolute_error(y_train, y_pred_train),
                np.sqrt(mean_squared_error(y_train, y_pred_train)))

    logger.info("测试集评估：")
    logger.info("  R² = %.4f | MAE = %.4f | RMSE = %.4f",
                r2_score(y_test, y_pred_test),
                mean_absolute_error(y_test, y_pred_test),
                np.sqrt(mean_squared_error(y_test, y_pred_test)))

    # 特征重要性
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    logger.info("=" * 60)
    logger.info("Top 15 特征重要性：")
    for _, row in importance.head(15).iterrows():
        bar = "█" * int(row["importance"] * 200)
        logger.info("  %-24s  %.4f  %s", row["feature"], row["importance"], bar)

    # 保存特征重要性图
    top20 = importance.head(20)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top20["feature"], top20["importance"], color="steelblue")
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance")
    ax.set_title("XGBoost Feature Importance (Top 20)")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("特征重要性图已保存：%s", OUTPUT_DIR / "feature_importance.png")

    logger.info("=" * 60)
    logger.info("全部完成。")


if __name__ == "__main__":
    main()
