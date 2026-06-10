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

---

## Preprocessing — Telco Customer Churn

Notebook : `telco_churn_preprocessing.ipynb`
Dataset : Telco Customer Churn — 7 043 clients, 21 colonnes, cible `Churn` (~73% No / ~27% Yes).

### Ce qui a été fait

| Phase | Action | Justification |
|---|---|---|
| 2 | Conversion `TotalCharges` object → float, imputation médiane (11 NaN) | Les 11 trous correspondent tous à `tenure=0` : trou structurel, pas une erreur |
| 3 | Suppression `customerID`, encodage binaire, ordinal (`Contract`), One-Hot (9 colonnes) | `customerID` = fuite de données ; `Contract` a un ordre naturel de durée |
| 4 | Détection outliers IQR sur `tenure`, `MonthlyCharges`, `TotalCharges` — aucun supprimé | Valeurs réelles (clients premium) ; supprimer biaiserait le profil churners |
| 5 | Suppression `TotalCharges` après VIF | Corrélation exacte avec `tenure × MonthlyCharges` = 1.000 → zéro info ajoutée |
| 6 | Classement des features (Pearson + Random Forest) | `Contract` et `tenure` dominent les deux méthodes |

### Colonnes après encodage

21 colonnes → 39 (après encodage) → 38 (après suppression `TotalCharges`)

### Features les plus discriminantes

1. `Contract` — les clients sans engagement (`Month-to-month`) résilient massivement plus
2. `tenure` — les nouveaux clients sont beaucoup plus volatils

