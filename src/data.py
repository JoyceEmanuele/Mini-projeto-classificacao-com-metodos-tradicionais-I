import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_cdc():
    print("carregando CDC...")
    ds = fetch_openml(data_id=46841, as_frame=True, parser="auto")
    X = ds.data.astype(float)
    y = ds.target.astype(int).values
    classes = ["Sem Diabetes", "Diabetes"]
    print(f"  {X.shape[0]} amostras, {X.shape[1]} features")
    return X, y, list(ds.feature_names), classes


def load_iris_data():
    print("carregando Iris...")
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target.values
    feature_names = [str(f) for f in iris.feature_names]
    class_names   = [str(c) for c in iris.target_names]
    print(f"  {X.shape[0]} amostras, {X.shape[1]} features")
    return X, y, feature_names, class_names


def describe_dataset(X, y, nome, n_features=6):
    print(f"\nResumo: {nome}")
    df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    print(df[df.columns[:n_features]].describe().round(3).to_string())
    unique, counts = np.unique(y, return_counts=True)
    print("classes:")
    for u, c in zip(unique, counts):
        print(f"  {u}: {c} amostras ({100*c/len(y):.1f}%)")


def prepare_data(X, y, test_size=0.2, random_state=42, max_samples=None):
    X_arr = X.values if hasattr(X, "values") else np.asarray(X, dtype=float)
    y_arr = np.asarray(y)

    if max_samples and len(X_arr) > max_samples:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(X_arr), max_samples, replace=False)
        X_arr, y_arr = X_arr[idx], y_arr[idx]
        print(f"usando {max_samples} amostras (subamostrado)")

    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr, test_size=test_size, random_state=random_state, stratify=y_arr
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"  treino={X_train.shape[0]}  teste={X_test.shape[0]}")
    return X_train, X_test, y_train, y_test, scaler
