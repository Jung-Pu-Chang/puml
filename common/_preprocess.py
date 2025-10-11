import datetime
import numpy as np
import pandas as pd
import sklearn
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    RobustScaler,
)
from scipy import stats
import warnings
import math

from ._base import BaseWithSeed

warnings.filterwarnings("ignore")


class PreProcess(BaseWithSeed):
    """
    資料前處理主類別，負責依序執行：
    1. 型別轉換
    2. 類別編碼
    3. 缺失值補值 (KNN/MICE)
    4. 特徵縮放 (RobustScaler)
    5. 異常值偵測 (IsolationForest)
    6. 資料漂移
    7. 統計檢驗
    """

    def __init__(self, train, test, target_col, drop_col=None, seed: int = 17):
        super().__init__(seed)
        self.target_col = (
            [target_col] if isinstance(target_col, str) else list(target_col)
        )
        self.drop_col = (
            []
            if drop_col is None
            else ([drop_col] if isinstance(drop_col, str) else list(drop_col))
        )

        # train test 切分
        self.train = train.drop(columns=self.drop_col, errors="ignore")
        self.test = test.drop(columns=self.drop_col, errors="ignore")
        self.trainY = self.train[self.target_col]  # 取出 target 欄位
        self.trainX = self.train.drop(columns=self.target_col, errors="ignore")
        self.testX = self.test[self.trainX.columns]

        # train test 合併
        self.df_all = pd.concat([self.train, self.test], axis=0, ignore_index=True)

    # ---------- 1. 自動型別轉換 ----------
    def auto_type_convert(self):
        df_type = self.df_all.copy()
        for col in self.df_all.columns:
            if df_type[col].dtype == "object":
                try:
                    df_type[col] = pd.to_datetime(df_type[col])
                    print(f"{col}: 轉為 datetime")
                except Exception:
                    df_type[col] = df_type[col].astype("category")
                    print(f"{col}: 轉為 category")
            elif pd.api.types.is_numeric_dtype(df_type[col]):
                df_type[col] = df_type[col].astype(float)
        print("🔍 自動轉換特徵型態完畢")
        return df_type

    # ---------- 2. 編碼（One-hot / Label / target） ----------
    def cat_encode(self, df_type, strategy="target", cat_cols=None):
        df_encoded = df.copy()

        if cat_cols is None:  # 若未指定，自動找 category dtype
            cat_cols = df_encoded.select_dtypes(
                include=["category", "object"]
            ).columns.tolist()

        if not cat_cols:
            print("⚠️ 未偵測到類別欄位，略過編碼步驟。")
            # return df_encoded

        for col in cat_cols:
            if strategy == "onehot":
                df_encoded = pd.get_dummies(df_encoded, columns=[col], prefix=col)
            elif strategy == "label":
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            elif strategy == "target":  # 注意 Data Leakage
                train_mapping_df = self.train  # 只抓 train
                target_mean = (
                    train_mapping_df.groupby(col)[self.target_col].mean().reset_index()
                )
                target_mean.columns = [col, f"{col}_Target_Mean"]
                df_encoded = pd.merge(df_encoded, target_mean, on=col, how="inner")
                df_encoded = df_encoded.drop(col, axis=1)
            else:
                raise ValueError(f"未知的編碼策略：{strategy}")
        print(f"🎨 使用 {strategy} 轉換類別特徵完畢")
        return df_encoded

    # ---------- 3. 缺失值補值（缺失率 na_rate < 0.1 補值，否則刪除該欄位） ----------
    def sparse_fill_na(
        self, df_encoded, na_rate=0.1, algorithm="knn", n_neighbors=None
    ):
        chk = df_encoded.isnull().mean().reset_index()
        chk.columns = ["欄位名稱", "缺失率"]
        fillna_col = chk.loc[chk["缺失率"] <= na_rate, "欄位名稱"].tolist()

        if not fillna_col:
            raise ValueError("沒有欄位符合 na_rate 條件可進行補值。")

        fillna_df = df_encoded[fillna_col].copy()

        if algorithm == "mice":
            if n_neighbors is not None:
                print("⚠️ n_neighbors 只在 KNN 模式有效，MICE 模式會忽略此參數")
            imputer = IterativeImputer(random_state=self.seed)
        elif algorithm == "knn":
            imputer = KNNImputer(n_neighbors=n_neighbors)
        else:
            raise ValueError("algorithm 必須是 'knn' 或 'mice'")

        # 刪除預測目標，再補值
        cols = [c for c in fillna_df.columns if c not in self.target_col]
        fillna_df = pd.DataFrame(imputer.fit_transform(fillna_df[cols]), columns=cols)

        fillna_df["type"] = np.repeat(
            ["train", "test"], [len(self.trainX), len(self.testX)]
        )
        print(f"🧩 使用 {algorithm.upper()} 補值完畢")
        return pd.concat([fillna_df, df_encoded[self.target_col]], axis=1)

    # ---------- 4. 特徵轉換 robust 消離群值 ----------
    def robust_scaling(self, fillna_df):
        fillna_df = fillna_df.drop(self.target_col, axis=1)
        df_train = fillna_df.loc[fillna_df["type"] == "train"].reset_index(drop=True)
        df_test = fillna_df.loc[fillna_df["type"] == "test"].reset_index(drop=True)
        num_cols = df_train.select_dtypes(include="number").columns
        scaler = RobustScaler()
        scaled_train = pd.DataFrame(
            scaler.fit_transform(df_train[num_cols]), columns=num_cols
        )

        scaled_test = pd.DataFrame(
            scaler.transform(df_test[num_cols]), columns=num_cols
        )

        df_train_scaled = pd.concat(
            [scaled_train, df_train.drop(columns=num_cols)], axis=1
        )
        df_test_scaled = pd.concat(
            [scaled_test, df_test.drop(columns=num_cols)], axis=1
        )
        scaled_df = pd.concat([df_train_scaled, df_test_scaled])

        print("✅ Robust scaling 完成")
        return pd.concat([scaled_df, self.trainY], axis=1)

    # ---------- 5. anomaly detect (IsolationForest)，並回傳去除 anomaly 後 train data ----------
    def anomaly_detect(self, scaled_df, max_features=1.0, contamination="auto"):
        # 注意：預測目標不能放入
        df_train = (
            scaled_df.loc[scaled_df["type"] == "train"]
            .drop(target_col + ["type"], axis=1)
            .reset_index(drop=True)
        )
        n_estimators = max(
            100, min(2000, 200 * max(1, int(round(math.log2(max(2, len(df_train)))))))
        )  # 建議樹量
        model = IsolationForest(
            n_estimators=n_estimators,
            max_samples="auto",
            contamination=contamination,  # 丟掉多少比例的異常值
            max_features=max_features,  # 每棵樹看多少特徵，1.0 = 100%
            random_state=self.seed,
        )
        model.fit(df_train)
        preds = model.predict(df_train)
        df_train["anomaly"] = preds
        n_removed = int((preds == -1).sum())
        percentage = n_removed / len(df_train) * 100
        df_train[self.target_col] = scaled_df.loc[
            scaled_df["type"] == "train", self.target_col
        ].reset_index(drop=True)
        df_train = df_train.loc[df_train["anomaly"] == 1].reset_index(drop=True)

        print(f"移除訓練資料 {n_removed} 異常值，約 {percentage:.1f}%")

        df_test = fillna_df.loc[fillna_df["type"] == "test"].reset_index(drop=True)
        df_clean = pd.concat([df_train, df_test], ignore_index=True)
        df_clean["type"] = ["train"] * len(df_train) + ["test"] * len(df_test)
        return df_clean.drop(["anomaly"], axis=1)

    # ---------- 6. train vs test similarity (KS test) ----------
    def data_drift(self, df_clean):
        results = []
        train_df = df_clean.loc[df_clean["type"] == "train"].drop(["type"], axis=1)
        test_df = df_clean.loc[df_clean["type"] == "test"].drop(["type"], axis=1)

        for col in train_df.select_dtypes(include=["number"]).columns:
            try:
                stat, p_val = stats.ks_2samp(
                    train_df[col].fillna(0), test_df[col].fillna(0)
                )
            except Exception:
                stat, p_val = np.nan, np.nan
            results.append(
                [col, stat, p_val, 0 if (not np.isnan(p_val) and p_val < 0.05) else 1]
            )

        print("🔎 check data_drift with KS-test")
        return pd.DataFrame(
            results, columns=["特徵", "KS統計量", "p-value", "是否相似"]
        )

    # ---------- 7. hypothesis tests (AD + Pearson) ----------
    def stat_summary_ad_pearson(self, df_clean):
        chk = pd.DataFrame(
            columns=["欄位名稱", "ad_stat", "是否常態", "獨立數", "不獨立數", "獨立率"]
        )
        df_train = df_clean.loc[df_clean["type"] == "train"]
        df_train = df_train.drop(columns=["type"] + self.target_col, errors="ignore")

        for col in df_train.columns:
            try:
                result = stats.anderson(df_train[col].dropna(), dist="norm")
                stat_value = result.statistic
                critical_value = (
                    result.critical_values[2]
                    if len(result.critical_values) > 2
                    else result.critical_values[-1]
                )
                is_normal = 1 if stat_value < critical_value else 0
            except Exception:
                stat_value = np.nan
                is_normal = 0

            independent_count = 0
            dependent_count = 0
            for other_col in df_train.columns:
                if col == other_col:
                    continue
                try:
                    corr, p_val = stats.pearsonr(
                        df_train[col].fillna(0), df_train[other_col].fillna(0)
                    )
                    if p_val < 0.05:
                        dependent_count += 1
                    else:
                        independent_count += 1
                except Exception:
                    pass
            total_comparisons = max(1, len(df_train.columns) - 1)
            independent_rate = independent_count / total_comparisons
            chk = pd.concat(
                [
                    chk,
                    pd.DataFrame(
                        {
                            "欄位名稱": [col],
                            "ad_stat": [stat_value],
                            "是否常態": [is_normal],
                            "獨立數": [independent_count],
                            "不獨立數": [dependent_count],
                            "獨立率": [round(independent_rate, 2)],
                        }
                    ),
                ],
                ignore_index=True,
            )
        print("📊 check train_x and train_y by ad、pearson test ")
        return chk
