import pandas as pd
import re
import shap
import matplotlib.pyplot as plt
import warnings

from ._base import BaseWithSeed

warnings.filterwarnings("ignore")


class XAI(BaseWithSeed):

    def shap_tree(self, model, train_X, class_cat=None):
        """
        計算 SHAP 值
        - 回歸模型: 直接回傳 shap_values (2D numpy array)
        - 分類模型: 需要指定 class_cat (類別索引)，若未指定，預設使用第 0 類
        """
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(train_X)

            if isinstance(shap_values, list):
                if class_cat is None:
                    print(
                        "[shap_tree] Warning: class_cat not specified, using class 0 by default"
                    )
                    class_cat = 0
                shap_values = shap_values[class_cat]

            baseline = explainer.expected_value
            return shap_values, baseline
        except Exception as e:
            print("shap_tree has error : " + str(e))
            return None

    def qcut_bin_shap_summary(self, X, shap_values, n_bins):
        """分位距分箱，n_bins = 4 = 四分位距"""
        results = []
        for i, col in enumerate(X.columns):
            try:
                bins = pd.qcut(X[col], q=n_bins, duplicates="drop")
            except ValueError:
                continue

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

    def plot_shap_bin_effects(self, df, save_dir=None):
        """
        畫每個變數的 SHAP 值區間影響曲線
        df 必須包含欄位: feature, bin, mean_shap
        """
        features = df["feature"].unique()

        for feat in features:
            sub_df = df[df["feature"] == feat].copy()

            def get_bin_mid(b):
                nums = re.findall(r"[-+]?\d*\.\d+|\d+", str(b))
                if len(nums) >= 2:
                    return (float(nums[0]) + float(nums[1])) / 2
                return None

            sub_df["bin_mid"] = sub_df["bin"].apply(get_bin_mid)
            sub_df = sub_df.sort_values("bin_mid")

            plt.figure(figsize=(6, 4))
            plt.plot(sub_df["bin_mid"], sub_df["mean_shap"], marker="o")
            plt.axhline(0, color="gray", linestyle="--", linewidth=1)
            plt.title(f"SHAP effect of {feat}")
            plt.xlabel(f"{feat} (bin mid)")
            plt.ylabel("Mean SHAP value (L)")
            plt.grid(True, linestyle="--", alpha=0.5)

            if save_dir:
                plt.savefig(
                    f"{save_dir}/{feat}_shap_curve.png", bbox_inches="tight", dpi=150
                )
                plt.close()
            else:
                plt.show()
