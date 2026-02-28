from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from imblearn.pipeline import Pipeline as ImbPipeline

from feature_engine.encoding import (
    MeanEncoder,
    OneHotEncoder,
    RareLabelEncoder,
    OrdinalEncoder,
)
from feature_engine.imputation import AddMissingIndicator, CategoricalImputer
from feature_engine.wrappers import SklearnTransformerWrapper

from ._base import BaseWithSeed
from ._drop_useless_data import DropHighNaNFeatures, IsolationForestCleaner


# DropConstantFeatures、共線性未新增
class PreprocessPipeline(BaseWithSeed):
    def __init__(
        self,
        drop_na_rate=0.1,
        imputer_strategy="knn",
        cat_combine_rate=0.02,
        encoding_strategy="target",
        norm_method="yeo-johnson",
        outlier_contamination="auto",
        seed: int = 17,
    ):
        super().__init__(seed)
        self.drop_na_rate = drop_na_rate
        self.imputer_strategy = imputer_strategy
        self.norm_method = norm_method
        self.cat_combine_rate = cat_combine_rate
        self.encoding_strategy = encoding_strategy
        self.outlier_contamination = outlier_contamination

        self.pipeline = None

    def _build_pipeline(self, X, remove_outlier=True):
        # 1. 定義補值器
        if self.imputer_strategy == "mice":
            imputer_algo = IterativeImputer(random_state=self.seed)
        else:
            imputer_algo = KNNImputer()

        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

        # 2. 定義步驟
        steps = []

        # Step A: 移除高缺失
        steps.append(
            ("A_drop_high_nan", DropHighNaNFeatures(threshold=self.drop_na_rate))
        )

        # Step B: 補值 (Indicator -> Cat -> Num)
        steps.append(("B1_missing_ind", AddMissingIndicator()))
        if cat_cols:
            steps.append(
                ("B2_impute_cat", CategoricalImputer(imputation_method="missing"))
            )

        # Wrapper 預設 variables=None 會自動鎖定數值欄位，不用手動指定
        steps.append(
            ("B3_imputer_num", SklearnTransformerWrapper(transformer=imputer_algo))
        )

        # Step C: 編碼 (Rare -> Encode)
        if cat_cols:
            steps.append(
                (
                    "C1_rare_label",
                    RareLabelEncoder(
                        tol=self.cat_combine_rate, n_categories=10, replace_with="Other"
                    ),
                )
            )  # 類別數 >= 10 啟動，把樣本數 < 2% 的合併成 Other

            if self.encoding_strategy == "target":
                steps.append(("C2_target_enc", MeanEncoder(smoothing=10)))
            elif self.encoding_strategy == "onehot":
                steps.append(("C2_onehot_enc", OneHotEncoder(drop_last=True)))
            elif self.encoding_strategy == "label":
                steps.append(
                    ("C2_label_enc", OrdinalEncoder(encoding_method="arbitrary"))
                )  # arbitrary 以出現順序編碼，ordered 以 y 單調增

        # Step D: 常態化 (PowerTransformer)
        steps.append(
            (
                "D_norm_transform",
                SklearnTransformerWrapper(
                    transformer=PowerTransformer(method=self.norm_method)
                ),
            )
        )

        # Step E: 縮放
        steps.append(("E_scaler", SklearnTransformerWrapper(RobustScaler())))

        # Step F: 移除異常值
        if remove_outlier:
            steps.append(
                (
                    "F_outlier_remover",
                    IsolationForestCleaner(
                        contamination=self.outlier_contamination,
                        seed=self.seed,
                        indicator_suffix="_na",
                    ),
                )
            )

        return ImbPipeline(steps)

    # pipeline：sklearn（只動 x） vs imblearn（x y 都動）
    def fit(self, train_x, train_y=None):  # 學習規則
        try:
            self.pipeline = self._build_pipeline(train_x)
            return self.pipeline.fit(train_x, train_y)
        except Exception as e:
            raise e

    def transform(self, test_x):  # 應用規則，新資料套用 fit 的規則
        try:
            print("test_x 不刪除 row，跳過異常偵測步驟")
            pipeline_no_outlier = self._build_pipeline(test_x, remove_outlier=False)
            pipeline_no_outlier.steps = [
                step for step in self.pipeline.steps if step[0] != "F_outlier_remover"
            ]

            return pipeline_no_outlier.transform(test_x)
        except Exception as e:
            raise e

    def fit_transform(self, train_x, train_y=None):  # 訓練並產出結果
        try:
            self.pipeline = self._build_pipeline(train_x, remove_outlier=True)
            X_processed = self.pipeline.fit_transform(train_x, train_y)

            if train_y is None:
                return X_processed

            # pandas Series / DataFrame 都安全
            if hasattr(train_y, "loc"):
                y_processed = train_y.loc[X_processed.index]
            else:
                # numpy array fallback
                mask = X_processed.index
                y_processed = train_y[mask]

            return X_processed, y_processed

        except Exception as e:
            raise e
