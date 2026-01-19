import pandas as pd
import numpy as np
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_validate,
    cross_val_score,
)
from sklearn.metrics import make_scorer, mean_absolute_error
import lightgbm as lgb
import optuna
import warnings

from ._base import BaseWithSeed

warnings.filterwarnings("ignore")


class LightGBM(BaseWithSeed):
    # 注意 python 3.7 以上，參數有序，不要動
    DEFAULT_OPTUNA_PARAMS = {
        "max_depth": [3, 5, 7, 9],
        "num_leaves": [15, 31, 63, 127],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "n_estimators": [500, 1000, 1500, 2000],
        "min_data_in_leaf": [20, 50, 100],
        "lambda_l1": [0, 0.1, 0.5, 1.0],
        "lambda_l2": [0, 0.1, 0.5, 1.0],
        "subsample": [0.6, 0.7, 0.8, 0.9],
        "feature_fraction": [0.6, 0.7, 0.8, 0.9],
        "bagging_freq": [1, 5],
        "path_smooth": [0, 0.1, 1.0],
    }

    def _get_kfold(self, n_splits, isClassifier, y):
        """回傳適合的 KFold 或 StratifiedKFold"""
        if isClassifier:
            try:
                return StratifiedKFold(
                    n_splits=n_splits, shuffle=True, random_state=self.seed
                )
            except ValueError:
                # 若分類樣本太少導致 stratify 失敗，退回 KFold
                return KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        else:
            return KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)

    def build_model(
        self, train_X, train_Y, params, scoring, n_splits, isClassifier=True
    ):
        train_X = train_X.to_numpy()
        train_Y = train_Y.to_numpy()

        try:
            if isClassifier:
                num_classes = len(np.unique(train_Y))
                params["num_class"] = num_classes if num_classes > 2 else None
                model = lgb.LGBMClassifier(**params, random_state=self.seed)
            else:
                model = lgb.LGBMRegressor(**params, random_state=self.seed)

            kf = self._get_kfold(n_splits, isClassifier, train_Y)

            cv = pd.DataFrame(
                cross_validate(
                    model,
                    train_X,
                    train_Y,
                    cv=kf,
                    scoring=scoring,
                )
            )

            cv_idx = [test_index for _, test_index in kf.split(train_X, train_Y)]
            model.fit(train_X, train_Y)

            return model, cv, cv_idx

        except Exception as e:
            print(f"build_model has error: {e}")
            return None, None, None

    def optuna_tune(
        self,
        train_X,
        train_Y,
        n_trials,
        n_splits,
        loss,
        isClassifier=True,
        param_space=None,
    ):
        train_X = train_X.to_numpy()
        train_Y = train_Y.to_numpy()

        current_param_space = param_space or self.DEFAULT_OPTUNA_PARAMS

        def objective(trial):
            kf = self._get_kfold(n_splits, isClassifier, train_Y)
            params_tuned = {}
            for key, values in current_param_space.items():
                if key == "num_leaves":
                    # def：num_leaves < 2^max_depth
                    max_depth = params_tuned.get("max_depth", 7)
                    params_tuned[key] = trial.suggest_int(
                        key, 2, min(values[-1], 2**max_depth - 1)
                    )
                else:
                    params_tuned[key] = trial.suggest_categorical(key, values)

            params_tuned.update({"verbose": -1, "random_state": self.seed})

            if isClassifier:
                unique_classes = np.unique(train_Y)
                if len(unique_classes) > 2:
                    params_tuned["num_class"] = len(unique_classes)
                    # 不平衡處理：動態搜權重 = 0固定為1，剩餘找 1.0~10.0 倍
                    class_weight = {0: 1.0}
                    for c in unique_classes:
                        if c != 0:
                            class_weight[c] = trial.suggest_float(
                                f"weight_class_{c}", 1.0, 10.0
                            )

                    model = lgb.LGBMClassifier(
                        **params_tuned, class_weight=class_weight
                    )
                else:
                    # 二元分類：scale_pos_weight 正(1)是負(0)的幾倍
                    s_weight = trial.suggest_float("scale_pos_weight", 1.0, 10.0)
                    model = lgb.LGBMClassifier(**params_tuned)

                score = cross_val_score(
                    model, train_X, train_Y, cv=kf, scoring="f1_macro"
                ).mean()
                return score
            else:
                model = lgb.LGBMRegressor(**params_tuned)
                mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
                score = -cross_val_score(
                    model, train_X, train_Y, cv=kf, scoring=mae_scorer
                ).mean()
                return score

        try:
            direction = "maximize" if isClassifier else "minimize"
            study = optuna.create_study(
                direction=direction,
                sampler=optuna.samplers.TPESampler(seed=self.seed),
            )
            study.optimize(objective, n_trials=n_trials)

            best_params = study.best_params
            best_params["objective"] = loss
            best_params["verbose"] = -1

            if isClassifier:
                weights = {0: 1.0}
                for c in np.unique(train_Y):
                    if c != 0:
                        key = f"weight_class_{c}"
                        weights[c] = best_params.pop(key)
                best_params["class_weight"] = weights
                best_params["num_class"] = len(np.unique(train_Y))
                # scale_pos_weight 會直接在 best_params 中，不需額外處理
            return best_params

        except Exception as e:
            print(f"optuna_tune has error: {e}")
            return None
