import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from data.loader import get_data
from explainers.generic_explainer import GenericVectorizedExplainer

def run_speed_benchmark(data_path):
    datasets = ["BreastCancer", "Arrhythmia", "CreditCard", "Covertype"]
    models_to_test = ['iforest', 'svm', 'lof']
    results = {m: {} for m in models_to_test}

    print("\n" + "="*60)
    print("RUNNING CROSS-MODEL VECTORIZATION TEST")
    print("="*60)

    for ds in datasets:
        print(f"\nProcessing Dataset: {ds}")
        X, y = get_data(ds, data_path)

        if X is not None:
            X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
            contamination = y.sum() / len(y)
            if contamination == 0 or contamination > 0.5: 
                contamination = 0.01

            for m_type in models_to_test:
                print(f"  > Training {m_type.upper()}...", end=" ")
                try:
                    explainer = GenericVectorizedExplainer(model_type=m_type, contamination=contamination)
                    
                    if len(X) > 50000 and m_type in ['svm', 'lof']:
                        X_train = X.iloc[:20000]
                    else:
                        X_train = X

                    explainer.fit(X_train)
                    
                    check_subset = X.iloc[:5000]
                    preds = explainer.predict(check_subset)
                    anomaly_idx = np.where(preds == -1)[0]

                    if len(anomaly_idx) > 0:
                        n_samples = min(50, len(anomaly_idx))
                        subset = check_subset.iloc[anomaly_idx[:n_samples]]

                        raw_time = explainer.get_explanation_time(subset)
                        time_per_100 = (raw_time / n_samples) * 100
                        results[m_type][ds] = round(time_per_100, 3)
                        print(f"Done. (Speed: {time_per_100:.2f}s / 100 items)")
                    else:
                        results[m_type][ds] = "No Anomalies Found"
                        print("No anomalies detected.")

                except Exception as e:
                    print(f"Failed ({e})")
                    results[m_type][ds] = "Error/Timeout"
        else:
            for m_type in models_to_test:
                results[m_type][ds] = "Dataset Missing"

    print("\n" + "="*60)
    print("FINAL RESULTS: Explanation Time per 100 Anomalies (Sec)")
    print("="*60)
    df_res = pd.DataFrame(results)
    print(df_res)