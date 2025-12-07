from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from imblearn.pipeline import Pipeline as ImbPipeline

from feature_engine.encoding import MeanEncoder, OneHotEncoder, RareLabelEncoder
from feature_engine.imputation import AddMissingIndicator, CategoricalImputer
from feature_engine.wrappers import SklearnTransformerWrapper

from ._base import BaseWithSeed
from ._drop_useless_data import DropHighNaNFeatures, IsolationForestCleaner


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

        self.pipeline = self._build_pipeline()

    def _build_pipeline(self):
        # 1. 定義補值器
        if self.imputer_strategy == "mice":
            imputer_algo = IterativeImputer(random_state=self.seed)
        else:
            imputer_algo = KNNImputer()

        # 2. 定義步驟
        steps = []

        # Step A: 移除高缺失
        steps.append(
            ("A_drop_high_nan", DropHighNaNFeatures(threshold=self.drop_na_rate))
        )

        # Step B: 補值 (Indicator -> Cat -> Num)
        steps.append(("B1_missing_ind", AddMissingIndicator()))
        steps.append(("B2_impute_cat", CategoricalImputer(imputation_method="missing")))
        # Wrapper 預設 variables=None 會自動鎖定數值欄位，不用手動指定
        steps.append(
            ("B3_imputer_num", SklearnTransformerWrapper(transformer=imputer_algo))
        )

        # Step C: 編碼 (Rare -> Encode)
        steps.append(
            (
                "C1_rare_label",
                RareLabelEncoder(
                    tol=self.cat_combine_rate, n_categories=10, replace_with="Other"
                ),
            )
        )  # 類別數 > 10 啟動，把樣本數 < 2% 的合併成 Other

        if self.encoding_strategy == "target":
            # smoothing=10: 樣本少於10的類別，其編碼值會被拉向總平均，避免過擬合
            # 交叉驗證需要另外用 OOF
            steps.append(("C2_target_enc", MeanEncoder(smoothing=10)))
        elif self.encoding_strategy == "onehot":
            steps.append(("C2_onehot_enc", OneHotEncoder(drop_last=True)))
        elif self.encoding_strategy == "label":
            steps.append(("C2_label_enc", OrdinalEncoder(encoding_method="arbitrary")))
            # arbitrary 以出現順序編碼，ordered 以 y 單調增

        # Step D: 常態化 (PowerTransformer)
        # 此時所有欄位(包含編碼後的類別)都是數值，Wrapper 會自動處理全部
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

        # Step F: 移除異常值 (這會改變樣本數，必須用 ImbPipeline)
        steps.append(
            (
                "F_outlier_remover",
                IsolationForestCleaner(
                    contamination=self.outlier_contamination, seed=self.seed
                ),
            )
        )

        return ImbPipeline(steps)

    # 依照 sklearn
    def fit(self, train_x, train_y=None):  # 擬合所有步驟
        try:
            return self.pipeline.fit(train_x, train_y)
        except Exception as e:
            raise e

    def transform(self, test_x):  # 在新資料上套用 fit 內容做轉換
        try:
            return self.pipeline.transform(test_x)
        except Exception as e:
            raise e

    def fit_transform(self, train_x, train_y=None):  # 擬合並轉換
        try:
            return self.pipeline.fit_transform(train_x, train_y)
        except Exception as e:
            raise e
