import os
import pandas as pd
from sklearn.datasets import load_svmlight_file

def get_data(name, data_path):
    try:
        if name == "BreastCancer":
            data = []
            path = os.path.join(data_path, 'breast-cancer.txt')
            if not os.path.exists(path): return None, None
            
            with open(path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    lbl = int(float(parts[0]))
                    y = 1 if lbl == 4 else 0
                    feats = {int(p.split(':')[0]): float(p.split(':')[1]) for p in parts[1:]}
                    feats['y'] = y
                    data.append(feats)
            df = pd.DataFrame(data).fillna(0)
            return df.drop('y', axis=1), df['y']

        elif name == "Arrhythmia":
            path = os.path.join(data_path, 'arrhythmia.data')
            if not os.path.exists(path): return None, None
            df = pd.read_csv(path, header=None, na_values='?')
            df = df.fillna(df.mean())
            X = df.iloc[:, :-1]
            y = (df.iloc[:, -1] != 1).astype(int)
            return X, y

        elif name == "CreditCard":
            path = os.path.join(data_path, 'creditcard.csv')
            if not os.path.exists(path): return None, None
            df = pd.read_csv(path)
            return df.drop(['Time', 'Class'], axis=1), df['Class']

        elif name == "Covertype":
            path = os.path.join(data_path, 'covtype.libsvm.binary')
            if not os.path.exists(path): return None, None
            X_sparse, y_raw = load_svmlight_file(path)
            X = pd.DataFrame(X_sparse.toarray())
            y = pd.Series(y_raw).map({2: 0, 4: 1}).fillna(0)
            return X, y

    except Exception as e:
        print(f"Skipping {name}: {e}")
        return None, None