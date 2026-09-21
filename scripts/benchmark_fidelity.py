import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from data.loader import get_data

def run_fidelity_benchmark(data_path):
    print("\nLoading CreditCard data for Fidelity Benchmark...")
    X, y = get_data("CreditCard", data_path)
    
    if X is None:
        print("CreditCard dataset not found. Please ensure the CSV exists in the data path.")
        return

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    clf = IsolationForest(n_estimators=100, contamination=0.0017, random_state=42, n_jobs=-1)
    clf.fit(X_scaled)

    preds = clf.predict(X_scaled)
    anomalies = X_scaled[preds == -1].iloc[:100] 

    print("\n" + "="*40)
    print("CREDIT CARD SENSITIVITY RESULTS")
    print("="*40)

    baselines = {
        'Mean': X_scaled.mean().values,
        'Median': X_scaled.median().values,
        'Zero': np.zeros(X_scaled.shape[1])
    }

    for name, base_vec in baselines.items():
        orig_scores = clf.decision_function(anomalies)
        fidelity_scores = []
        
        for i in range(len(anomalies)):
            row = anomalies.iloc[i].values
            mat = np.tile(row, (len(row), 1))
            np.fill_diagonal(mat, base_vec)
            
            new_scores = clf.decision_function(mat)
            gain = np.max(new_scores - orig_scores[i])
            fidelity_scores.append(gain)

        print(f" -> Baseline '{name}': {np.mean(fidelity_scores):.4f}")