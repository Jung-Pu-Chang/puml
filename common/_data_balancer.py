import pandas as pd
from typing import Type, Tuple, Dict, Any
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.base import BaseSampler

from ._base import BaseWithSeed


class Balancer(BaseWithSeed):

    def __init__(self, seed: int = 17):
        super().__init__(seed)
        self._samplers: Dict[str, Type[BaseSampler]] = {
            "smote": SMOTE,
            "adasyn": ADASYN,
            "under_sample": RandomUnderSampler,
        }  # 可擴充：未來新增演算法

    def _apply_sampler(
        self, sampler: BaseSampler, X: pd.DataFrame, y: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        sampler_name = sampler.__class__.__name__
        print(f"\n===== 開始執行 {sampler_name} 採樣 =====")
        print("原始樣本數分佈:\n" + y.value_counts().to_string())

        try:
            resampled_X, resampled_y = sampler.fit_resample(X, y)
            print(
                f"{sampler_name} 後樣本數分佈:\n"
                + resampled_y.value_counts().to_string()
            )
            print(f"✅ {sampler_name} 採樣完成。")
            return resampled_X, resampled_y
        except ValueError as e:
            print(f"❌ {sampler_name} 執行失敗: {e}")
            raise e

    def resample(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        method: str,
        **kwargs: Any,  # 依照原演算法參數
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        method_key = method.lower()
        if method_key not in self._samplers:
            raise ValueError(
                f"暫不支援: '{method}'。可用選項: {list(self._samplers.keys())}"
            )

        sampler_class = self._samplers[method_key]

        if "random_state" not in kwargs:
            kwargs["random_state"] = self.seed

        sampler_instance = sampler_class(**kwargs)

        return self._apply_sampler(sampler_instance, X, y)
