import os
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from .utils import print_metrics, plot_confusion_matrix, plot_decision_boundary


def run_svm_sklearn(X_train, X_test, y_train, y_test, classes, dataset, figures_dir):
    resultados = {}
    n_folds = 5 if len(X_train) >= 1000 else 3

    configs = {
        "linear": {"C": [0.01, 0.1, 1, 10, 100]},
        "rbf":    {"C": [0.1, 1, 10, 100], "gamma": ["scale", "auto", 0.01, 0.1]},
    }

    for kernel, params in configs.items():
        print(f"\nSVM {kernel} — {dataset}")
        svc = SVC(kernel=kernel, class_weight="balanced", random_state=42)
        grid = GridSearchCV(svc, params, cv=n_folds, scoring="f1_weighted", n_jobs=-1)
        grid.fit(X_train, y_train)
        melhor = grid.best_estimator_
        print(f"melhores params: {grid.best_params_}")

        y_pred = melhor.predict(X_test)
        m = print_metrics(y_test, y_pred, f"SVM {kernel} ({dataset})")
        resultados[kernel] = m

        prefixo = os.path.join(figures_dir, f"svm_{kernel}_{dataset.lower()}")
        plot_confusion_matrix(y_test, y_pred, classes, f"SVM {kernel} — {dataset}", f"{prefixo}_cm.png")
        plot_decision_boundary(melhor, X_test, y_test, f"Fronteira SVM {kernel} — {dataset}", classes, f"{prefixo}_boundary.png")

    return resultados
