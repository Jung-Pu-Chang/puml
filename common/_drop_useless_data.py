import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from feature_engine.wrappers import SklearnTransformerWrapper

from ._base import BaseWithSeed


# --- 依據缺失率刪除欄位 ---
class DropHighNaNFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.1):
        self.threshold = threshold
        self.drop_cols_ = []

    def fit(self, X, y=None):
        # 計算每個欄位的缺失率
        na_rate = X.isnull().mean()
        self.drop_cols_ = na_rate[na_rate > self.threshold].index.tolist()
        if self.drop_cols_:
            print(f"⚠️ DropHighNaNFeatures: 將移除以下高缺失率欄位: {self.drop_cols_}")
        return self

    def transform(self, X):
        return X.drop(columns=self.drop_cols_, errors="ignore")


# --- IsolationForest 異常值移除 (僅在訓練時作用) ---
class IsolationForestCleaner(BaseEstimator, BaseWithSeed):
    """
    相容於 imblearn pipeline 的採樣器 (Sampler)。
    fit_resample 只在訓練時執行，transform 在推論時不執行 (保留原樣)。
    """

    def __init__(self, contamination="auto", seed: int = 17):
        super().__init__(seed)
        self.contamination = contamination
        self.random_state = self.seed
        self.model_ = None

    def fit_resample(self, X, y):
        # 1. 訓練 IF 模型
        self.model_ = IsolationForest(
            contamination=self.contamination, random_state=self.random_state, n_jobs=-1
        )
        preds = self.model_.fit_predict(X)

        # 2. 篩選非異常值 (preds == 1)
        mask = preds != -1
        n_removed = (~mask).sum()

        if n_removed > 0:
            print(
                f"🗑️ IsolationForest: 移除 {n_removed} 筆異常樣本 (佔 {n_removed/len(X):.1%})"
            )

        return X[mask], y[mask]

    # 為了相容一般 fit
    def fit(self, X, y):
        return self
