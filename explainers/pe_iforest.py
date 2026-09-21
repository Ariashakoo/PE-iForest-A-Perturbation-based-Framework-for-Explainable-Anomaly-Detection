import time
import numpy as np
from sklearn.ensemble import IsolationForest

class VectorizedPEForest:
    def __init__(self, contamination='auto'):
        self.model = IsolationForest(n_estimators=100, contamination=contamination, n_jobs=-1, random_state=42)
        self.feature_names = None
        self.global_mean = None

    def fit(self, X):
        self.feature_names = X.columns
        self.global_mean = X.mean()
        self.model.fit(X)

    def predict(self, X):
        return self.model.predict(X)

    def get_explanation_time(self, X_anomalies):
        start_time = time.time()
        
        # Pre-calculate original scores
        original_scores = self.model.decision_function(X_anomalies)
        n_features = len(self.feature_names)

        for i in range(len(X_anomalies)):
            row = X_anomalies.iloc[i].values
            # Vectorized Perturbation Batching
            batch = np.tile(row, (n_features, 1))
            np.fill_diagonal(batch, self.global_mean.values)
            
            # Batch Predict
            _ = self.model.decision_function(batch)

        end_time = time.time()
        return end_time - start_time