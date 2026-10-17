# Mini-projeto: Classificação com Métodos Tradicionais

Implementação e avaliação comparativa de três métodos clássicos de classificação supervisionada sobre dois conjuntos de dados.

## Métodos implementados

- **SVM** — kernel linear e RBF (scikit-learn `SVC`)
- **Regressão Logística** — solver LBFGS (scikit-learn `LogisticRegression`)
- **kNN** — distância euclidiana, melhor k via validação cruzada (scikit-learn `KNeighborsClassifier`)

## Datasets

| Dataset | Fonte | Amostras | Features | Classes |
|---|---|---|---|---|
| CDC Diabetes Health Indicators | [OpenML ID 46841](https://openml.org/d/46841) | 253.680 | 21 | 2 (desbalanceado: 86%/14%) |
| Iris | scikit-learn built-in | 150 | 4 | 3 (balanceado) |

## Estrutura do projeto

```
.
├── main.py          # ponto de entrada — executa todos os experimentos
├── src/
│   ├── data.py      # carregamento e preparação dos dados
│   ├── svm.py       # SVM linear e RBF
│   ├── logistic.py  # Regressão Logística
│   ├── knn.py       # kNN com busca do melhor k
│   ├── compare.py   # tabela e gráfico comparativo
│   └── utils.py     # métricas, matrizes de confusão, fronteiras de decisão
└── figures/         # gerado automaticamente ao rodar o projeto
```

## Como executar

### Dependências

```bash
pip install scikit-learn numpy pandas matplotlib
```

### Rodar

```bash
python3 main.py
```

Os resultados são impressos no terminal e as figuras salvas em `figures/`.

## Figuras geradas

Para cada método e dataset são geradas:
- Fronteira de decisão (projeção PCA 2D para dados com mais de 2 features)
- Matriz de confusão
- Curva F1 × k (apenas kNN)
- Painel comparativo com todas as métricas
