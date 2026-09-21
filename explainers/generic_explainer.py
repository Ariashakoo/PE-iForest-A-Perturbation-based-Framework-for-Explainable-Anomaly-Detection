import time
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

class GenericVectorizedExplainer:
    def __init__(self, model_type='iforest', contamination='auto'):
        self.model_type = model_type
        self.feature_names = None
        self.global_mean = None

        if model_type == 'iforest':
            self.model = IsolationForest(n_estimators=100, contamination=contamination, n_jobs=-1, random_state=42)
        elif model_type == 'svm':
            self.model = OneClassSVM(kernel='linear', nu=0.01)
        elif model_type == 'lof':
            self.model = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=contamination, n_jobs=-1)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def fit(self, X):
        self.feature_names = X.columns
        self.global_mean = X.mean()
        self.model.fit(X)

    def predict(self, X):
        return self.model.predict(X)

    def decision_function(self, X):
        return self.model.decision_function(X)

    def get_explanation_time(self, X_anomalies):
        start_time = time.time()
        _ = self.decision_function(X_anomalies)
        n_features = len(self.feature_names)

        for i in range(len(X_anomalies)):
            row = X_anomalies.iloc[i].values
            batch = np.tile(row, (n_features, 1))
            np.fill_diagonal(batch, self.global_mean.values)
            _ = self.decision_function(batch)

        end_time = time.time()
        return end_time - start_time