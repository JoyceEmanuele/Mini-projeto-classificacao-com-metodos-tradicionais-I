import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICAS = ["accuracy", "precision", "recall", "f1"]
LABELS   = ["Acurácia", "Precisão", "Revocação", "F1"]


def plot_comparison(resultados, dataset, caminho):
    metodos = list(resultados.keys())
    x = np.arange(len(metodos))
    largura = 0.18

    fig, ax = plt.subplots(figsize=(max(8, len(metodos) * 1.5), 5))
    for i, (chave, label) in enumerate(zip(METRICAS, LABELS)):
        vals = [resultados[m].get(chave, 0) for m in metodos]
        ax.bar(x + i * largura, vals, largura, label=label, alpha=0.85)

    ax.set(xlabel="Método", ylabel="Score", title=f"Comparação — {dataset}",
           ylim=(0, 1.1))
    ax.set_xticks(x + largura * 1.5)
    ax.set_xticklabels(metodos, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(caminho, dpi=150)
    plt.close()


def print_comparison_table(resultados, dataset):
    print(f"\nComparação final — {dataset}")
    print(f"{'Método':<22} {'Acurácia':>10} {'Precisão':>10} {'Revocação':>10} {'F1':>8}")
    print("-" * 62)
    for metodo, m in resultados.items():
        print(f"{metodo:<22} {m.get('accuracy',0):>10.4f} {m.get('precision',0):>10.4f}"
              f" {m.get('recall',0):>10.4f} {m.get('f1',0):>8.4f}")
    melhor = max(resultados, key=lambda k: resultados[k].get("f1", 0))
    print(f"\nMelhor F1: {melhor} ({resultados[melhor]['f1']:.4f})")
