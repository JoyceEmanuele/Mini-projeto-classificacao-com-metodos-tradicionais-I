import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)

COLORS_LIGHT = ["#FFAAAA", "#AAFFAA", "#AAAAFF", "#FFDDAA"]
COLORS_BOLD  = ["#CC0000", "#007700", "#0000CC", "#CC7700"]


def compute_metrics(y_true, y_pred, average="weighted"):
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall":    recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1":        f1_score(y_true, y_pred, average=average, zero_division=0),
    }


def print_metrics(y_true, y_pred, titulo="", average="weighted"):
    m = compute_metrics(y_true, y_pred, average)
    print(f"\n{titulo}")
    print(f"acuracia={m['accuracy']:.4f}  precisao={m['precision']:.4f}"
          f"revocacao={m['recall']:.4f}  f1={m['f1']:.4f}")
    print("matriz de confusao:")
    print(confusion_matrix(y_true, y_pred))
    return m


def plot_confusion_matrix(y_true, y_pred, classes, titulo, caminho):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)
    ticks = np.arange(len(classes))
    ax.set(xticks=ticks, yticks=ticks,
           xticklabels=classes, yticklabels=classes,
           title=titulo, ylabel="Real", xlabel="Predito")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)
    plt.tight_layout()
    plt.savefig(caminho, dpi=150)
    plt.close()


def plot_decision_boundary(clf, X, y, titulo, classes, caminho, h=0.05):
    n_classes = len(np.unique(y))
    cmap_light = ListedColormap(COLORS_LIGHT[:n_classes])
    cmap_bold  = ListedColormap(COLORS_BOLD[:n_classes])

    if X.shape[1] > 2:
        pca = PCA(n_components=2, random_state=42)
        X2d = pca.fit_transform(X)
        x_min, x_max = X2d[:, 0].min() - 0.5, X2d[:, 0].max() + 0.5
        y_min, y_max = X2d[:, 1].min() - 0.5, X2d[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = np.asarray(clf.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))).reshape(xx.shape)
        X_plot = X2d
        xlabel, ylabel = "PC1", "PC2"
    else:
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = np.asarray(clf.predict(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape)
        X_plot = X
        xlabel, ylabel = "Feature 1", "Feature 2"

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.75)
    sc = ax.scatter(X_plot[:, 0], X_plot[:, 1], c=y,
                    cmap=cmap_bold, edgecolors="k", s=15, alpha=0.7, linewidths=0.4)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=titulo)
    handles, _ = sc.legend_elements()
    ax.legend(handles, classes, title="Classe", fontsize=7)
    plt.tight_layout()
    plt.savefig(caminho, dpi=150)
    plt.close()
