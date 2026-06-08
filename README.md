# arene-des-algos-Mathieu_PONNOU

Repo de la semaine ML/DL — pipeline de classification supervisée et non-supervisée, avec comparaison de plusieurs algorithmes sur plusieurs datasets.

## Ce que vous trouverez ici

- **Notebook principal** (`ml_arena_breast_cancer.ipynb`) : pipeline ML complet de bout en bout
- Exploration des datasets (breast cancer, wine)
- Split train/test, entraînement, prédiction, évaluation
- L'Arène : classement de plusieurs algos sur le même split
- Clustering non-supervisé (KMeans) sans les étiquettes
- Visualisations : barplot des accuracies, matrices de confusion

## Datasets utilisés

| Dataset | Type | Classes | Exemples | Features |
|---|---|---|---|---|
| `load_breast_cancer` | Classification binaire | 2 (bénigne / maligne) | 569 | 30 |
| `load_wine` | Classification multi-classe | 3 | 178 | 13 |

## Algos comparés

- Régression logistique
- K-Nearest Neighbors (KNN)
- Arbre de décision

