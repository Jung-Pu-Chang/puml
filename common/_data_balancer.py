from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
import warnings

from ._base import BaseWithSeed

warnings.filterwarnings("ignore")


class Balancer(BaseWithSeed):

    def smote_resample(self, train_x, train_y, k_neighbors):
        print("原始樣本數分佈:\n", train_y.value_counts().to_string())

        smote = SMOTE(
            sampling_strategy="auto", k_neighbors=k_neighbors, random_state=self.seed
        )

        try:
            resampled_x, resampled_y = smote.fit_resample(train_x, train_y)
        except ValueError as e:
            print(f"❌ SMOTE 執行失敗，請檢查資料：{e}")

        resampled_x = pd.DataFrame(resampled_x, columns=train_x.columns)
        resampled_y = pd.DataFrame(resampled_y, columns=train_y.columns)

        print(
            "SMOTE 後樣本數分佈:\n", resampled_y.iloc[:, 0].value_counts().to_string()
        )
        print("✅ SMOTE 過採樣完成。")

        return resampled_x, resampled_y

    def adasyn_resample(self, train_x, train_y, k_neighbors):
        print("原始樣本數分佈:\n", train_y.value_counts().to_string())
        adasyn = ADASYN(
            sampling_strategy="minority",
            random_state=self.seed,
            n_neighbors=k_neighbors,
        )

        try:
            resampled_x, resampled_y = adasyn.fit_resample(train_x, train_y)
        except ValueError as e:
            print(f"❌ ADASYN 執行失敗，請檢查資料：{e}")

        resampled_x = pd.DataFrame(resampled_x, columns=train_x.columns)
        resampled_y = pd.DataFrame(resampled_y, columns=train_y.columns)

        print(
            "ADASYN 後樣本數分佈:\n", resampled_y.iloc[:, 0].value_counts().to_string()
        )
        print("✅ ADASYN 過採樣完成。")

        return resampled_x, resampled_y

    def under_sample(self, train_x, train_y):
        print("原始樣本數分佈:\n", train_y.value_counts().to_string())

        rus = RandomUnderSampler(random_state=self.seed)

        try:
            resampled_x, resampled_y = rus.fit_resample(train_x, train_y)
        except ValueError as e:
            print(f"❌ 欠取樣執行失敗，請檢查資料：{e}")

        print(
            "欠取樣後樣本數分佈:\n", resampled_y.iloc[:, 0].value_counts().to_string()
        )
        print("✅ 欠取樣完成。")

        return resampled_x, resampled_y
