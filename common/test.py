if __name__ == "__main__":
    # === 模擬資料 ===
    np.random.seed(42)
    n = 200
    train = pd.DataFrame(
        {
            "A": np.random.randn(n),
            "B": np.random.randint(0, 10, n),
            "C": np.random.choice(["X", "Y", "Z"], n),
            "D": np.random.choice([np.nan, 1, 2, 3], n, p=[0.2, 0.3, 0.3, 0.2]),
            "E": np.random.randn(n) * 10,
            "target": np.random.choice([0, 1], n),
        }
    )

    # 測試集隨機抽一半
    test = train.sample(frac=0.5, random_state=42).reset_index(drop=True)
    test["target"] = np.nan

    # === 初始化並執行 ===
    pp = PreProcess(
        train=train,
        test=test,
        target_col="target",
        drop_col=["E"],  # 假設想忽略某欄
        seed=999,
    )

    df = pp.auto_type_convert()
    df_encoded = pp.cat_encode(df, strategy="target", cat_cols=None)
    fillna_df = pp.sparse_fill_na(
        df_encoded, na_rate=0.1, algorithm="knn", n_neighbors=5
    )
    scaled_df = pp.robust_scaling(fillna_df)
    df_clean = pp.anomaly_detect(scaled_df, max_features=1.0, contamination="auto")
    chk = pp.data_drift(df_clean)
    chk2 = pp.stat_summary_ad_pearson(df_clean)

    print("\n==== Step 1. 自動前處理 ====")
    processed_df = pp.fit_transform(
        na_alg="knn", encode_strategy="auto", outlier_method="iqr", top_k=5
    )
    print(processed_df.head())

    print("\n==== Step 2. Scaling 測試 ====")
    scaled_df = pp.scaling(processed_df, method="standard")
    print(scaled_df.describe())

    print("\n==== Step 3. Train/Test 相似度 ====")
    sim = pp.train_test_similarity()
    print(f"Train/Test Similarity: {sim:.4f}")

    print("\n==== Step 4. Clustering 測試 ====")
    cluster_result = pp.clustering(k=3)
    print(cluster_result["cluster"].value_counts())

    print("\n==== Step 5. 異常值檢測 ====")
    anomalies = pp.anomaly_detect()
    print(f"Anomaly count: {anomalies.sum()}")

    print("\n==== Step 6. 假設檢定 ====")
    hypo = pp.hypothesis_test("A", "target")
    print(hypo)
