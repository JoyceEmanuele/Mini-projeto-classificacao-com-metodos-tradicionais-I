import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data     import load_cdc, load_iris_data, describe_dataset, prepare_data
from src.svm      import run_svm_sklearn
from src.logistic import run_logistic_sklearn
from src.knn      import run_knn
from src.compare  import plot_comparison, print_comparison_table

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def rodar_dataset(nome, X, y, classes, max_samples=None):
    print(f"\n {nome} ")
    describe_dataset(X, y, nome)
    X_train, X_test, y_train, y_test, _ = prepare_data(X, y, max_samples=max_samples)

    resultados = {}

    svm = run_svm_sklearn(X_train, X_test, y_train, y_test, classes, nome, FIGURES_DIR)
    resultados["SVM Linear"] = svm["linear"]
    resultados["SVM RBF"]    = svm["rbf"]

    resultados["LogReg"] = run_logistic_sklearn(X_train, X_test, y_train, y_test, classes, nome, FIGURES_DIR)

    _, knn = run_knn(X_train, X_test, y_train, y_test, classes, nome, FIGURES_DIR)
    resultados["kNN"] = knn

    print_comparison_table(resultados, nome)
    plot_comparison(resultados, nome, os.path.join(FIGURES_DIR, f"comparison_{nome.lower()}.png"))

    return resultados


if __name__ == "__main__":
    X_cdc, y_cdc, _, classes_cdc = load_cdc()
    rodar_dataset("CDC", X_cdc, y_cdc, classes_cdc, max_samples=20_000)

    X_iris, y_iris, _, classes_iris = load_iris_data()
    rodar_dataset("Iris", X_iris, y_iris, classes_iris)

    print(f"\nfiguras salvas em: {FIGURES_DIR}")
