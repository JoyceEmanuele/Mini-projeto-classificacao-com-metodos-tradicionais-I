import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from .utils import print_metrics, plot_confusion_matrix, plot_decision_boundary


def run_logistic_sklearn(X_train, X_test, y_train, y_test, classes, dataset, figures_dir):
    print(f"\nRegressão Logística — {dataset}")
    n_folds = 5 if len(X_train) >= 1000 else 3

    lr = LogisticRegression(solver="lbfgs", class_weight="balanced", max_iter=2000, random_state=42)
    grid = GridSearchCV(lr, {"C": [0.001, 0.01, 0.1, 1, 10, 100]},
                        cv=n_folds, scoring="f1_weighted", n_jobs=-1)
    grid.fit(X_train, y_train)
    melhor = grid.best_estimator_
    print(f"melhores params: {grid.best_params_}")

    y_pred = melhor.predict(X_test)
    m = print_metrics(y_test, y_pred, f"LogReg ({dataset})")

    prefixo = os.path.join(figures_dir, f"logistic_{dataset.lower()}")
    plot_confusion_matrix(y_test, y_pred, classes, f"Regressão Logística — {dataset}", f"{prefixo}_cm.png")
    plot_decision_boundary(melhor, X_test, y_test, f"Fronteira LogReg — {dataset}", classes, f"{prefixo}_boundary.png")
    return m
