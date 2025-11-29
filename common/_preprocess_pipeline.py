from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from imblearn.pipeline import Pipeline

from feature_engine.encoding import MeanEncoder, OneHotEncoder, RareLabelEncoder
from feature_engine.imputation import AddMissingIndicator
from feature_engine.wrappers import SklearnTransformerWrapper

from ._base import BaseWithSeed
from ._drop_useless_data import DropHighNaNFeatures, IsolationForestCleaner


# drop → impute → normalize → encode → scale → outlier
class PreprocessPipeline(BaseWithSeed):
    def __init__(
        self,
        # --- 配置參數 (Configuration Parameters) ---
        drop_na_rate=0.1,  # A: 移除高缺失欄位 (> 10%)
        imputer_strategy="knn",  # B: 補值方法 ('mice' or 'knn')
        norm_method="yeo-johnson",  # C: 常態化方法
        cat_combine_rate=0.02,  # D-1: 類別依樣本數合併 (< 2%)
        encoding_strategy="target",  # D-2: 類別編碼方法 ('target' or 'onehot')
        outlier_contamination="auto",  # E: 異常值移除比例（0.05 等）
        seed: int = 17,
    ):
        super().__init__(seed)
        self.drop_na_rate = drop_na_rate
        self.imputer_strategy = imputer_strategy
        self.norm_method = norm_method
        self.cat_combine_rate = cat_combine_rate
        self.encoding_strategy = encoding_strategy
        self.outlier_contamination = outlier_contamination

        # 初始化 pipeline
        self.pipeline = self._build_pipeline()

    # --------------------------------------------------------
    # 建立前處理 pipeline
    # --------------------------------------------------------
    def _build_pipeline(self):

        # ---------- 1. 補值演算法 ----------
        if self.imputer_strategy == "mice":
            imputer_step = SklearnTransformerWrapper(
                transformer=IterativeImputer(random_state=self.seed),
                variables=None,
            )
        else:
            imputer_step = SklearnTransformerWrapper(
                transformer=KNNImputer(),
                variables=None,
            )

        # ---------- 2. pipeline steps ----------
        steps = []

        # Step A: 移除高缺失欄位
        steps.append(
            ("A_drop_high_nan", DropHighNaNFeatures(threshold=self.drop_na_rate))
        )

        # Step B: 缺失值處理 (B-1: 標記, B-2: 補值)
        steps.append(("C1_missing_ind", AddMissingIndicator()))
        steps.append(("C2_imputer", imputer_step))

        # Step C: 分佈轉換 (常態化)
        steps.append(
            (
                "D_norm_transform",
                SklearnTransformerWrapper(
                    transformer=PowerTransformer(method=self.norm_method),
                    variables=None,  # 對所有數值欄位執行
                ),
            )
        )

        # Step D: 類別編碼 (D-1: 稀有標籤, D-2: 數值編碼)
        steps.append(
            (
                "E1_rare_label",
                RareLabelEncoder(
                    tol=self.cat_combine_rate, n_categories=2, replace_with="Other"
                ),
            )
        )

        if self.encoding_strategy == "target":
            steps.append(("E2_target_enc", MeanEncoder(smoothing=10)))
        elif self.encoding_strategy == "onehot":
            steps.append(("E2_onehot_enc", OneHotEncoder(drop_last=True)))

        # Step E: Scaling (特徵縮放)
        steps.append(("F_scaler", SklearnTransformerWrapper(RobustScaler())))

        # Step F: Outlier removal (樣本刪除)
        steps.append(
            (
                "G_outlier_remover",
                IsolationForestCleaner(
                    contamination=self.outlier_contamination,
                    seed=self.seed,
                ),
            )
        )

        return Pipeline(steps)

    # 依照 sklearn
    def fit(self, train_x, train_y=None):  # 擬合所有步驟
        return self.pipeline.fit(train_x, train_y)

    def transform(self, test_x):  # 在新資料上套用 fit 內容做轉換
        return self.pipeline.transform(test_x)

    def fit_transform(self, train_x, train_y=None):  # 擬合並轉換
        return self.pipeline.fit_transform(train_x, train_y)
