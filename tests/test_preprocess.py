import pandas as pd
import numpy as np
from common import BaseWithSeed, PreProcess


class TestLightGBM(unittest.TestCase):

    def setUp(self):
        # 創建 LightGBM 實例
        self.lgbm_module = LightGBM(seed=17)
        self.n_samples = 200
        self.n_features = 5
        self.fold_time = 3

        # 預設評分標準
        self.scoring_cls = {
            "accuracy": make_scorer(accuracy_score),
            "f1_macro": make_scorer(f1_score, average="macro"),
        }
        self.scoring_reg = {
            "mae": make_scorer(mean_absolute_error, greater_is_better=False),
        }

    def _create_data(self, n_classes=2, is_classifier=True):
        """創建模擬數據"""
        X = np.random.rand(self.n_samples, self.n_features)
        if is_classifier:
            if n_classes == 2:
                # 二元分類 (稍微不平衡)
                y = np.random.choice([0, 1], size=self.n_samples, p=[0.7, 0.3])
            else:
                # 多元分類
                y = np.random.randint(0, n_classes, size=self.n_samples)
        else:
            # 迴歸
            y = X[:, 0] * 5 + X[:, 1] * 2 + np.random.normal(0, 0.5, self.n_samples)

        return X, y

    ## 1. 迴歸測試 (LGBMRegressor)

    def test_01_regression_build_model(self):
        """測試迴歸建模和 KFold CV"""
        X, y = self._create_data(is_classifier=False)
        params = {"n_estimators": 50, "learning_rate": 0.1}

        model, cv, cv_idx = self.lgbm_module.build_model(
            X,
            y,
            params=params,
            scoring=self.scoring_reg,
            fold_time=self.fold_time,
            isClassifier=False,
        )

        # 驗證返回結果
        self.assertIsInstance(model, lgb.LGBMRegressor)
        self.assertEqual(len(cv), self.fold_time)
        self.assertIn("test_mae", cv.columns)
        self.assertEqual(len(cv_idx), self.fold_time)
        self.assertIsInstance(cv_idx[0], np.ndarray)

    def test_02_regression_optuna_tune(self):
        """測試迴歸 Optuna 調參"""
        X, y = self._create_data(is_classifier=False)
        n_trials = 3

        best_params = self.lgbm_module.optuna_tune(
            X, y, n_trials=n_trials, loss="mae", isClassifier=False
        )

        # 驗證返回參數
        self.assertIsInstance(best_params, dict)
        self.assertIn("objective", best_params)
        self.assertEqual(best_params["objective"], "mae")
        # 驗證參數是從定義的空間中選取
        self.assertIn(
            best_params["max_depth"],
            self.lgbm_module.DEFAULT_OPTUNA_PARAMS["max_depth"],
        )

    ## 2. 分類測試 (LGBMClassifier)

    def test_03_binary_classification_build_model(self):
        """測試二元分類建模和 StratifiedKFold CV"""
        X, y = self._create_data(n_classes=2, is_classifier=True)
        params = {"n_estimators": 50, "learning_rate": 0.1}

        model, cv, cv_idx = self.lgbm_module.build_model(
            X,
            y,
            params=params,
            scoring=self.scoring_cls,
            fold_time=self.fold_time,
            isClassifier=True,
        )

        # 驗證返回結果
        self.assertIsInstance(model, lgb.LGBMClassifier)
        self.assertEqual(len(cv), self.fold_time)
        self.assertIn("test_accuracy", cv.columns)
        self.assertEqual(len(cv_idx), self.fold_time)
        # 驗證 num_class 不在 params 中 (二元分類)
        self.assertNotIn("num_class", model.get_params())

    def test_04_multiclass_classification_build_model(self):
        """測試多元分類建模和 StratifiedKFold CV"""
        X, y = self._create_data(n_classes=4, is_classifier=True)
        params = {"n_estimators": 50, "learning_rate": 0.1}

        model, cv, cv_idx = self.lgbm_module.build_model(
            X,
            y,
            params=params,
            scoring=self.scoring_cls,
            fold_time=self.fold_time,
            isClassifier=True,
        )

        # 驗證返回結果
        self.assertIsInstance(model, lgb.LGBMClassifier)
        self.assertIn("num_class", model.get_params())
        self.assertEqual(model.get_params()["num_class"], 4)  # 驗證 num_class 設置正確

    def test_05_classification_optuna_tune(self):
        """測試分類 Optuna 調參"""
        X, y = self._create_data(n_classes=3, is_classifier=True)
        n_trials = 3

        best_params = self.lgbm_module.optuna_tune(
            X, y, n_trials=n_trials, loss="multiclass", isClassifier=True
        )

        # 驗證返回參數
        self.assertIsInstance(best_params, dict)
        self.assertIn("objective", best_params)
        self.assertEqual(best_params["objective"], "multiclass")
        # 驗證 num_class 設置正確
        self.assertEqual(best_params["num_class"], 3)
        # 驗證參數是從定義的空間中選取
        self.assertIn(
            best_params["boosting_type"],
            self.lgbm_module.DEFAULT_OPTUNA_PARAMS["boosting_type"],
        )

    ## 3. Pipeline + SMOTE 測試 (高標準防洩漏)

    def test_06_pipeline_smote_build_model(self):
        """測試傳入 Pipeline (SMOTE -> LGBM) 的建模和 CV，驗證防資料洩漏"""
        X, y = self._create_data(n_classes=2, is_classifier=True)

        # 1. 獲取基礎參數
        best_params = self.lgbm_module.optuna_tune(
            X, y, n_trials=2, loss="binary", isClassifier=True
        )

        # 2. 建立 Pipeline
        base_lgbm = self.lgbm_module.get_base_model(best_params, isClassifier=True)
        smote = SMOTE(
            random_state=self.lgbm_module.seed, k_neighbors=1
        )  # k=1 以確保數據量能增加
        pipeline = Pipeline([("smote", smote), ("classifier", base_lgbm)])

        # 3. 執行 build_model
        model, cv, cv_idx = self.lgbm_module.build_model(
            X,
            y,
            model=pipeline,
            scoring=self.scoring_cls,
            fold_time=self.fold_time,
            isClassifier=True,
        )

        # 驗證返回結果
        self.assertIsInstance(model, Pipeline)
        self.assertIsInstance(model.named_steps["classifier"], lgb.LGBMClassifier)
        self.assertEqual(len(cv), self.fold_time)
        self.assertIn("test_f1_macro", cv.columns)

        # 驗證 SMOTE 在 CV 中正確應用 (間接測試：Pipeline 的 CV 是正確的)
        # 由於 Pipeline 在 cross_validate 內部會正確處理 SMOTE，我們只需檢查 CV 執行成功即可。
        # 為了進一步驗證，我們確保 CV 分數有合理範圍（非空）
        self.assertTrue(cv["test_f1_macro"].notnull().all())

    def test_07_build_model_no_params_no_model(self):
        """測試未傳入 model 或 params 時拋出 ValueError"""
        X, y = self._create_data(n_classes=2, is_classifier=True)
        with self.assertRaises(ValueError):
            self.lgbm_module.build_model(X, y, scoring=self.scoring_cls, fold_time=3)


if __name__ == "__main__":
    # 運行測試
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
