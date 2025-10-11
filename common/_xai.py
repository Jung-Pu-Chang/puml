import os
import pandas as pd
import numpy as np
import re
import shap
from typing import List, Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
import warnings

from ._base import BaseWithSeed

# 微軟正黑體, 黑體
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Heiti TC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # 解決負號顯示問題
warnings.filterwarnings("ignore")


class XAI(BaseWithSeed):

    def shap_tree(
        self, model: Any, train_X: pd.DataFrame, class_cat: Optional[int] = None
    ) -> Tuple[Optional[np.ndarray], Optional[Any]]:
        """
        計算 SHAP 值
        - 回歸模型: 直接回傳 shap_values (2D numpy array)
        - 分類模型: 需要指定 class_cat (類別索引)，若未指定，預設使用第 0 類
        """
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(train_X)
            baseline = explainer.expected_value

            if isinstance(shap_values, list):  # 多分類 = list 格式
                if class_cat is None:
                    print(
                        "[shap_tree] Warning: class_cat not specified, using class 0 by default"
                    )
                    class_cat = 0  # 預設以 0 為解釋目標
                shap_values = shap_values[class_cat]

            return shap_values, baseline
        except Exception as e:
            print("shap_tree has error : " + str(e))
            return None, None

    def qcut_bin_shap_summary(
        self, X: pd.DataFrame, shap_values: np.ndarray, n_bins: int
    ) -> pd.DataFrame:
        """分位距分箱，n_bins = 4 = 四分位距"""
        results = []
        for i, col in enumerate(X.columns):
            try:
                bins = pd.qcut(X[col], q=n_bins, duplicates="drop")
            except ValueError:
                continue  # 若分箱失敗，繼續換下一個特徵

            bin_df = pd.DataFrame(
                {"feature_value": X[col], "bin": bins, "shap_value": shap_values[:, i]}
            )

            summary = (
                bin_df.groupby("bin")
                .agg(mean_shap=("shap_value", "mean"), count=("shap_value", "size"))
                .reset_index()
            )

            summary["feature"] = col
            results.append(summary)

        final_df = pd.concat(results, ignore_index=True)
        return final_df

    def plot_shap_bin_effects(self, df: pd.DataFrame, save_dir: Optional[str] = None):
        """
        畫每個變數的 SHAP 值區間影響曲線
        df 必須包含欄位: feature, bin, mean_shap
        """
        if not all(col in df.columns for col in ["feature", "bin", "mean_shap"]):
            raise ValueError(
                "輸入的 DataFrame 缺少必要欄位 ('feature', 'bin', 'mean_shap')。"
            )

        features = df["feature"].unique()

        for feat in features:
            sub_df = df[df["feature"] == feat].copy()

            # pandas.mid 抓中間值，確保 'bin' 欄位是 Interval
            if pd.api.types.is_interval_dtype(sub_df["bin"]):
                sub_df["bin_mid"] = sub_df["bin"].apply(lambda b: b.mid)
            else:
                # 如果不是 Interval，即類別特徵，直接使用
                sub_df["bin_mid"] = sub_df["bin"]

            sub_df = sub_df.sort_values("bin_mid")

            plt.figure(figsize=(8, 5))
            plt.plot(
                sub_df["bin_mid"],
                sub_df["mean_shap"],
                marker="o",
                linestyle="-",
                color="royalblue",
            )
            plt.axhline(0, color="gray", linestyle="--", linewidth=1)

            plt.title(f"特徵 '{feat}' 的 SHAP 影響曲線", fontsize=16)
            plt.xlabel(f"特徵值 '{feat}' (區間中點)", fontsize=12)
            plt.ylabel("對模型輸出的平均影響 (SHAP 值)", fontsize=12)
            plt.grid(True, linestyle=":", alpha=0.6)
            plt.tight_layout()  # 自動調整邊距

            if save_dir:
                try:
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"{feat}_shap_curve.png")
                    plt.savefig(save_path, bbox_inches="tight", dpi=150)
                    plt.close()
                    print(f"圖表已儲存至: {save_path}")
                except Exception as e:
                    print(f"儲存圖表 '{feat}' 時發生錯誤: {e}")
            else:
                plt.show()
