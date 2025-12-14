import pandas as pd
import numpy as np
import warnings
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split

from common import BaseWithSeed, PreprocessPipeline, DataValidator

warnings.filterwarnings("ignore")

"""
1994 年人口普查資料
預測：年收入是否 > 5萬美金
https://archive.ics.uci.edu/dataset/2/adult
"""

print("Loading UCI Adult dataset...")
data = fetch_ucirepo(id=2)
X = data.data.features
y = data.data.targets
y["income"] = y["income"].replace(to_replace=[">50K", ">50K."], value=1, regex=True)
y["income"] = y["income"].replace(to_replace=["<=50K", "<=50K."], value=0, regex=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y["income"], test_size=0.2, random_state=BaseWithSeed().seed
)
# y series 兼容所有 sklearn / feature-engine，效能幾乎沒差

"""
1. 原始 train test check drift：ks_test + chi-squared
2. preprocess pipeline
3. train_pre test_pre check drift：ks_test + chi-squared（可略）
4. train_pre test_pre check_normality_and_correlation
"""

# 型態轉換屬於客製化，如日期等，需丟到pipeline 前客製化處理
chk = DataValidator.check_drift(X_train, X_test)

pipeline = PreprocessPipeline(
    drop_na_rate=0.1,  # A: 移除高缺失欄位 (> 10%)
    imputer_strategy="knn",  # B: 補值方法 ('mice' or 'knn')
    cat_combine_rate=0.02,  # C-1: 類別依樣本數合併 (< 2%)
    encoding_strategy="target",  # C-2: 類別編碼方法 ('target' or 'onehot' or 'label')
    norm_method="yeo-johnson",  # D: 常態化方法
    outlier_contamination="auto",  # F: 異常值移除比例
    # seed = 2266,
)

X_train_processed = pipeline.fit_transform(X_train, y_train)
X_test_processed = pipeline.transform(X_test)

chk = DataValidator.check_drift(X_train_processed, X_test_processed)
chk = DataValidator.check_normality_and_correlation(X_train_processed)

# %%
from imblearn.pipeline import Pipeline

steps = pipeline.pipeline.steps

step_names = [name for name, _ in steps]
idx = step_names.index("F_outlier_remover")  # 不包含
partial_pipeline = Pipeline(steps[:idx])
X_tmp = partial_pipeline.fit_transform(X_train, y_train)

step = pipeline.pipeline.named_steps["C2_label_enc"]
X_tmp = step.fit_transform(X_tmp, y_train)

X_tmp.isna().sum()
