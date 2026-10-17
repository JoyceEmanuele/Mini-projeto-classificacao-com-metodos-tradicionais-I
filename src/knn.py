import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from .utils import print_metrics, plot_confusion_matrix, plot_decision_boundary


def run_knn(X_train, X_test, y_train, y_test, classes, dataset, figures_dir):
    print(f"\nkNN — {dataset}")
    n_folds = 5 if len(X_train) >= 1000 else 3
    k_max = min(30, len(X_train) // n_folds)

    knn = KNeighborsClassifier(metric="euclidean")
    grid = GridSearchCV(knn, {"n_neighbors": list(range(1, k_max + 1))},
                        cv=n_folds, scoring="f1_weighted", n_jobs=-1)
    grid.fit(X_train, y_train)
    melhor_k = grid.best_params_["n_neighbors"]
    melhor = grid.best_estimator_
    print(f"  melhor k: {melhor_k}")

    m_treino = print_metrics(y_train, melhor.predict(X_train), f"kNN k={melhor_k} treino ({dataset})")
    m_teste  = print_metrics(y_test,  melhor.predict(X_test),  f"kNN k={melhor_k} teste ({dataset})")

    prefixo = os.path.join(figures_dir, f"knn_{dataset.lower()}")
    plot_confusion_matrix(y_test, melhor.predict(X_test), classes, f"kNN k={melhor_k} — {dataset}", f"{prefixo}_cm.png")
    plot_decision_boundary(melhor, X_test, y_test, f"Fronteira kNN k={melhor_k} — {dataset}", classes, f"{prefixo}_boundary.png")

    scores = grid.cv_results_["mean_test_score"]
    k_range = list(range(1, k_max + 1))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(k_range, scores, "o-", markersize=4)
    ax.axvline(melhor_k, color="r", linestyle="--", label=f"k={melhor_k}")
    ax.set(xlabel="k", ylabel="F1 (validação cruzada)", title=f"kNN — {dataset}: F1 × k")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{prefixo}_k_curve.png", dpi=150)
    plt.close()

    return m_treino, m_teste
