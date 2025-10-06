import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_validate, cross_val_score
from sklearn.metrics import (
    classification_report,
    make_scorer,
    mean_absolute_error,
)
import lightgbm as lgb
import optuna
import warnings
from collections import Counter

from ._base import BaseWithSeed

warnings.filterwarnings("ignore")


class LightGBM(BaseWithSeed):

    def build_model(
        self, train_X, train_Y, params, scoring, fold_time, isClassifier=True
    ):
        """建模 + kfold(驗證資料間互斥)，np.array計算"""
        train_X = train_X.to_numpy()
        train_Y = train_Y.to_numpy()

        try:
            if isClassifier:
                num_classes = len(np.unique(train_Y))
                if num_classes > 2:
                    params["num_class"] = num_classes
                else:
                    params.pop("num_class", None)

                model = lgb.LGBMClassifier(**params, random_state=self.seed)
            else:
                model = lgb.LGBMRegressor(**params, random_state=self.seed)

            kf = KFold(fold_time, random_state=self.seed, shuffle=True)
            cv = pd.DataFrame(
                cross_validate(
                    model,
                    train_X,
                    train_Y,
                    cv=kf,
                    scoring=scoring,
                )
            )
            cv_idx = [test_index for train_index, test_index in kf.split(train_X)]

            model.fit(train_X, train_Y)
            return model, cv, cv_idx
        except Exception as e:
            print("build_model has error : " + str(e))
            return None, None, None

    def optuna_tune(self, train_X, train_Y, n_trials, loss, isClassifier=True):
        """
        調參，輸出最佳模型 + 參數結果，np.array計算
        metric 預設 = objective，評估指標，不影響模型訓練
        """
        train_X = train_X.to_numpy()
        train_Y = train_Y.to_numpy()

        def objective(trial):
            kf = KFold(n_splits=5, shuffle=True, random_state=self.seed)
            params_tuned = {
                "learning_rate": trial.suggest_categorical(
                    "learning_rate", [0.1, 0.05, 0.01]
                ),
                "max_depth": trial.suggest_categorical("max_depth", [3, 5, 7]),
                "subsample": trial.suggest_categorical("subsample", [0.1, 0.5, 0.7]),
                "n_estimators": trial.suggest_categorical(
                    "n_estimators", [1000, 1500, 2000]
                ),
                "boosting_type": trial.suggest_categorical(
                    "boosting_type", ["gbdt", "dart"]
                ),
                "random_state": self.seed,
                "verbose": -1,
            }
            if isClassifier:
                model = lgb.LGBMClassifier(**params_tuned)
                score = cross_val_score(
                    model, train_X, train_Y, cv=kf, scoring="f1_macro"
                ).mean()
            else:
                model = lgb.LGBMRegressor(**params_tuned)
                mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
                score = -cross_val_score(
                    model, train_X, train_Y, cv=kf, scoring=mae_scorer
                ).mean()
            return score

        try:
            if isClassifier:
                study = optuna.create_study(
                    direction="maximize",
                    sampler=optuna.samplers.TPESampler(seed=self.seed),
                )
                study.optimize(objective, n_trials=n_trials)
                best_params = study.best_params
                best_params["objective"] = loss
                best_params["verbose"] = -1
                num_classes = len(np.unique(train_Y))
                best_params["num_class"] = num_classes  # binary 不影響
                best_model = lgb.LGBMClassifier(**best_params)
            else:
                study = optuna.create_study(
                    direction="minimize",
                    sampler=optuna.samplers.TPESampler(seed=self.seed),
                )
                study.optimize(objective, n_trials=n_trials)
                best_params = study.best_params
                best_params["objective"] = loss
                best_params["verbose"] = -1

            return best_params
        except Exception as e:
            print("optuna_tune has error : " + str(e))
            return None, None

    def cv_error_analysis(model, train, cv_idx):

        all_y_true = []
        all_y_pred = []

        train_x = train[FEATURE_COLUMNS].fillna(0).to_numpy()
        train_y = train[["風險分類"]].to_numpy()

        for test_idx in cv_idx:
            x_val = train_x[test_idx]
            y_val = train_y[test_idx]

            # 用原模型對 fold 驗證集預測
            y_pred = model.predict(x_val)

            all_y_true.append(y_val)
            all_y_pred.append(y_pred)

        # 合併所有 fold
        y_true_cv = np.concatenate(all_y_true)
        y_pred_cv = np.concatenate(all_y_pred)

        return classification_report(y_true_cv, y_pred_cv, digits=4)
