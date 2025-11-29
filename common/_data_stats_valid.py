from scipy import stats


class DataValidator:
    """
    data drift：KS-Test
    常態性：AD Test
    相關性：Pearson
    """

    @staticmethod
    def check_drift(X_train, X_test):
        """執行 KS-Test 檢查資料漂移"""
        results = []
        # 只檢查數值型欄位
        num_cols = X_train.select_dtypes(include=["number"]).columns

        print("\n🔎 正在檢查資料漂移 (KS-Test)...")
        for col in num_cols:
            if col in X_test.columns:
                try:
                    stat, p_val = stats.ks_2samp(
                        X_train[col].dropna(), X_test[col].dropna()
                    )
                    # p < 0.05 代表分佈顯著不同 (有漂移)
                    drift_detected = 1 if p_val < 0.05 else 0
                    results.append(
                        {
                            "Feature": col,
                            "KS_Stat": round(stat, 4),
                            "P_Value": round(p_val, 4),
                            "Drift_Detected": drift_detected,
                        }
                    )
                except Exception as e:
                    pass

        return pd.DataFrame(results)

    @staticmethod
    def check_normality_and_correlation(df):
        """執行 AD Test (常態性) 與 Pearson (相關性)"""
        results = []
        num_cols = df.select_dtypes(include=["number"]).columns

        print("\n📊 正在分析統計特徵 (Normality & Correlation)...")
        for col in num_cols:
            # 1. Normality (Anderson-Darling)
            is_normal = 0
            try:
                # 注意：樣本數過大時，AD Test 幾乎都會拒絕常態假設
                res = stats.anderson(df[col].dropna(), dist="norm")
                if res.statistic < res.critical_values[2]:  # 使用 5% 顯著水準
                    is_normal = 1
            except:
                pass

            # 2. Correlation (Average absolute correlation with other features)
            avg_corr = 0
            try:
                corr_matrix = df[num_cols].corr().abs()
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
