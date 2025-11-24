# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 10:17:30 2025

@author: user
"""

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
    DEFAULT_OPTUNA_PARAMS = {
        "learning_rate": [0.1, 0.05, 0.01],
        "max_depth": [3, 5, 7],
        "subsample": [0.5, 0.7, 0.9],
        "n_estimators": [500, 1000, 1500],
        "boosting_type": ["gbdt", "dart"],
        "bagging_freq": [0, 1, 5],
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
            params_tuned = {
                key: trial.suggest_categorical(key, values)
                for key, values in current_param_space.items()
            }
            params_tuned.update({"verbose": -1, "random_state": self.seed})

            if isClassifier:
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
                num_classes = len(np.unique(train_Y))
                best_params["num_class"] = num_classes

            return best_params

        except Exception as e:
            print(f"optuna_tune has error: {e}")
            return None
