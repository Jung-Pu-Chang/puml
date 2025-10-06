import datetime
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import sklearn
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from scipy import stats  # boxcox & yeojohnson
import warnings

from ._base import BaseWithSeed

warnings.filterwarnings("ignore")


class PreProcess(BaseWithSeed):

    def __init__(self, train, test, target_col, drop_col, seed: int = 17):
        super().__init__(seed)

        if isinstance(target_col, str):  # str -> list
            target_col = [target_col]

        if isinstance(drop_col, str):
            drop_col = [drop_col]

        self.train = train.drop(columns=drop_col, axis=1)
        self.test = test.drop(columns=drop_col, axis=1)
        self.target_col = target_col
        self.drop_col = drop_col
        self.trainY = train[target_col]  # 取出 target 欄位
        self.trainX = train.drop(columns=target_col, axis=1)
        self.testX = test.drop(columns=target_col, axis=1)
        self.encoding_maps = []
        self.df = pd.concat([self.trainX, self.testX], axis=0).reset_index(drop=True)
        self.df_all = pd.concat([self.train, self.test], axis=0, ignore_index=True)

    def EDA(self):
        print("train + test EDA")  # na 不影響型態
        chk = pd.DataFrame(
            {
                "欄位名稱": self.df_all.columns,
                "缺失樣本數": self.df_all.isnull().sum().values,
                "總樣本數": len(self.df_all),
                "缺失率": self.df_all.isnull().mean().values,
                "資料型態": self.df_all.dtypes.values,
            }
        )
        return chk

    def target_encode(self, cat_cols):
        print("train + test target mean")
        for cat_col in cat_cols:
            target_mean = (
                self.df_all.groupby(cat_col)[self.target_col].mean().reset_index()
            )
            target_mean.columns = [cat_col, "Target_Mean"]
            self.encoding_maps.append(target_mean)
            self.df_all[cat_col] = self.df_all[cat_col].map(
                dict(zip(target_mean[cat_col], target_mean["Target_Mean"]))
            )
            self.df_all[cat_col] = self.df_all[cat_col].fillna(
                self.df_all[self.target_col].mean()
            )
        return self.df_all, self.encoding_maps

    def sparse_fill_na(self, chk, na_rate, algorithm="knn"):
        print("train + test fill_na")
        fillna_col = chk.loc[chk["缺失率"] <= na_rate]["欄位名稱"].tolist()
        fillna_df = self.df_all[fillna_col]

        if algorithm == "mice":
            print("使用 MICE 進行補值")
            mice_imputer = IterativeImputer(random_state=self.seed)
            fillna_df = pd.DataFrame(
                mice_imputer.fit_transform(fillna_df),
                columns=fillna_df.columns,
            )
        else:
            print("使用 KNN 進行補值")
            knn_imputer = KNNImputer(n_neighbors=3)
            fillna_df = pd.DataFrame(
                knn_imputer.fit_transform(fillna_df), columns=fillna_df.columns
            )

        fillna_df["type"] = np.repeat(
            ["train", "test"], [len(self.trainX), len(self.testX)]
        )
        return fillna_df

    def scaling(self, fillna_df, min_, max_):
        print("train scaling & fit test")
        standard_scale = sklearn.preprocessing.StandardScaler()
        minmax_scale = sklearn.preprocessing.MinMaxScaler(feature_range=(min_, max_))
        robust_scale = sklearn.preprocessing.RobustScaler()

        numeric_cols = fillna_df.drop(columns=["type"]).columns
        fillna_df_numeric = fillna_df[numeric_cols]

        standard_trans = pd.DataFrame(
            standard_scale.fit_transform(fillna_df_numeric),
            columns=numeric_cols,
        )
        minmax_trans = pd.DataFrame(
            minmax_scale.fit_transform(fillna_df_numeric), columns=numeric_cols
        )
        robust_trans = pd.DataFrame(
            robust_scale.fit_transform(fillna_df_numeric), columns=numeric_cols
        )

        standard_trans["type"] = fillna_df["type"].values
        minmax_trans["type"] = fillna_df["type"].values
        robust_trans["type"] = fillna_df["type"].values
        return standard_trans, minmax_trans, robust_trans

    def train_test_similarity(self, df):
        print("train vs test similarity")
        results = []
        train_df = df.loc[df["type"] == "train"].drop(["type"], axis=1)
        test_df = df.loc[df["type"] == "test"].drop(["type"], axis=1)

        for col in train_df.select_dtypes(include=["number"]).columns:
            stat, p_val = stats.ks_2samp(train_df[col], test_df[col])
            results.append([col, stat, p_val, 0 if p_val < 0.05 else 1])

        return pd.DataFrame(
            results, columns=["特徵", "KS統計量", "p-value", "是否相似"]
        )

    def clustering(self, df):
        print("train + test GMM clustering")
        df = df.drop(["type"], axis=1)
        n_components_range = range(1, 10)
        bic_scores = []

        for n in n_components_range:
            gmm = GaussianMixture(n_components=n, random_state=self.seed)
            gmm.fit(df)
            bic_scores.append(gmm.bic(df))

        optimal_n_components = n_components_range[np.argmin(bic_scores)]
        gmm = GaussianMixture(
            n_components=optimal_n_components,
            covariance_type="full",
            random_state=self.seed,
        )
        gmm.fit(df)

        probs = gmm.predict_proba(df)
        labels = gmm.predict(df)
        labels = pd.DataFrame(labels).rename(columns={0: "硬分群"})

        df_probs = pd.DataFrame(
            probs,
            columns=[f"Cluster_{i+1}_Prob" for i in range(optimal_n_components)],
        )
        df_result = pd.concat([df, labels, df_probs], axis=1)
        bic_scores = pd.DataFrame(bic_scores).rename(columns={0: "bic_score"})
        return df_result, bic_scores

    def anomaly_detect(self, df):
        print("train anomaly detection by IsolationForest")
        df = df.loc[df["type"] == "train"].drop(["type"], axis=1)
        n_estimators = 200 * round(math.log2(len(df)))
        model = IsolationForest(
            n_estimators=n_estimators,
            max_samples="auto",
            contamination=0.1,
            max_features=1.0,
            random_state=self.seed,
        )
        model.fit(df)
        df["anomaly"] = model.predict(df)
        df = pd.concat([df, self.trainY], axis=1).reset_index(drop=True)
        print("drop " + str(len(df.loc[df["anomaly"] == -1])) + " anomaly from train")
        df = df.loc[df["anomaly"] == 1]
        return df

    def hypothesis_test(self, df):
        np.random.seed(self.seed)
        print("train + test ad & pearson test")
        chk = pd.DataFrame(
            columns=["欄位名稱", "ad臨界值", "是否常態", "獨立數", "不獨立數", "獨立率"]
        )
        df = df.drop(["type"], axis=1)
        df = df.drop(self.target_col, axis=1)

        for col in df.columns:
            result = stats.anderson(df[col], dist="norm")
            stat_value = result.statistic
            critical_value = result.critical_values[2]
            is_normal = 1 if stat_value < critical_value else 0
            independent_count = 0
            dependent_count = 0

            for other_col in df.columns:
                if col != other_col:
                    corr, p_val = stats.pearsonr(df[col], df[other_col])
                    if p_val < 0.05:
                        dependent_count += 1
                    else:
                        independent_count += 1

            total_comparisons = len(df.columns) - 1
            independent_rate = independent_count / total_comparisons
            chk = pd.concat(
                [
                    chk,
                    pd.DataFrame(
                        {
                            "欄位名稱": [col],
                            "ad臨界值": [stat_value],
                            "是否常態": [is_normal],
                            "獨立數": [independent_count],
                            "不獨立數": [dependent_count],
                            "獨立率": [round(independent_rate, 2)],
                        }
                    ),
                ],
                ignore_index=True,
            )
        return chk
