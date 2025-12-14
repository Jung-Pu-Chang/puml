from scipy import stats
import pandas as pd
import numpy as np


class DataValidator:
    SIGNIFICANCE_LEVEL = 0.05

    @staticmethod
    def check_drift(X_train: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
        """
        資料漂移檢查（KS：數值型、Chi²：類別型）
        p-value < α → 拒絕 H0 → 分佈顯著不同 → 有 drift
        """

        results = []

        print("\n🔎 Checking data drift (KS-Test & Chi-Squared Test)...")

        # ---------- Numerical (KS-Test) ----------
        for col in X_train.select_dtypes(include="number"):
            if col not in X_test:
                continue

            train, test = X_train[col].dropna(), X_test[col].dropna()
            if len(train) < 2 or len(test) < 2:
                continue

            stat, p = stats.ks_2samp(train, test)
            results.append(
                {
                    "Feature": col,
                    "Test_Type": "KS-Test",
                    "Statistic": round(stat, 4),
                    "P_Value": round(p, 4),
                    "Drift_Detected": int(p < DataValidator.SIGNIFICANCE_LEVEL),
                }
            )

        # ---------- Categorical (Chi-Squared) ----------
        for col in X_train.select_dtypes(include=["object", "category"]):
            if col not in X_test:
                continue

            train_cnt = X_train[col].value_counts()
            test_cnt = X_test[col].value_counts()
            idx = train_cnt.index.union(test_cnt.index)

            train_f = train_cnt.reindex(idx, fill_value=0)
            test_f = test_cnt.reindex(idx, fill_value=0)

            if train_f.sum() == 0 or test_f.sum() == 0:
                continue

            stat, p, _, _ = stats.chi2_contingency([train_f, test_f])
            results.append(
                {
                    "Feature": col,
                    "Test_Type": "Chi2-Test",
                    "Statistic": round(stat, 4),
                    "P_Value": round(p, 4),
                    "Drift_Detected": int(p < DataValidator.SIGNIFICANCE_LEVEL),
                }
            )

        return pd.DataFrame(results)

    @staticmethod
    def check_normality_and_correlation(df: pd.DataFrame) -> pd.DataFrame:
        """
        執行統計特徵分析：
        - 常態性：AD Test (Anderson-Darling Test)
        - 相關性：Pearson 連續型、
        """
        results = []
        num_cols = df.select_dtypes(include=["number"]).columns

        print("\n📊 正在分析統計特徵 (Normality & Correlation)...")

        # *** 效率優化：只計算一次相關矩陣 ***
        # 使用 .fillna(0) 處理可能產生的 NaN (例如常數欄位)
        try:
            corr_matrix = df[num_cols].corr().abs().fillna(0)
        except ValueError:
            # 如果欄位太少，無法計算相關性，則跳過
            corr_matrix = pd.DataFrame(0, index=num_cols, columns=num_cols)

        for col in num_cols:

            # 1. Normality (Anderson-Darling)
            is_normal = 0
            try:
                data = df[col].dropna().to_numpy()
                # AD Test 需要至少 8 個樣本
                if len(data) >= 8 and np.std(data) > 0:
                    res = stats.anderson(data, dist="norm")
                    # 與 5% 顯著水準的 Critical Value 比較
                    if (
                        res.statistic
                        < res.critical_values[DataValidator.AD_CRITICAL_INDEX]
                    ):
                        is_normal = 1
            except:
                pass

            # 2. Correlation (Average absolute correlation with other features)
            avg_corr = 0
            try:
                # 從已計算的矩陣中取出該欄位的平均絕對相關性
                avg_corr = corr_matrix[col].mean()
            except:
                pass

            results.append(
                {
                    "Feature": col,
                    "Is_Normal": is_normal,
                    "Avg_Correlation": round(avg_corr, 4),
                }
            )

        return pd.DataFrame(results)
