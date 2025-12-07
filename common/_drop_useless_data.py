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
class IsolationForestCleaner(BaseEstimator, TransformerMixin, BaseWithSeed):
    def __init__(self, contamination="auto", seed=17):
        super().__init__(seed)
        self.contamination = contamination

    def fit(self, X, y=None):
        self.model_ = IsolationForest(
            contamination=self.contamination, random_state=self.seed, n_jobs=-1
        )
        self.model_.fit(X)
        return self

    def transform(self, X):
        preds = self.model_.predict(X)
        # 1 表示正常，-1 表示異常
        X_out = X.copy()
        X_out["__is_outlier"] = (preds == -1).astype(int)
        mask = preds == 1
        return X_out.loc[mask].drop(columns="__is_outlier")
