# arene-des-algos-Mathieu_PONNOU

Repo de la semaine ML/DL — preprocessing, régression, clustering, classification texte et binaire, comparaison d'algorithmes sur plusieurs datasets réels.

## Ce que vous trouverez ici

- **`j1_ml_arena_breast_cancer.ipynb`** : premier pipeline supervisé complet, arène de 3 algos sur breast cancer et wine, clustering KMeans, démonstration data leakage
- **`j2_telco_churn_preprocessing.ipynb`** : pipeline de preprocessing complet sur le dataset Telco Customer Churn (audit qualité, encodage, outliers, multicolinéarité, features discriminantes)
- **`j3_arene_des_algos.ipynb`** : régression, clustering, classification texte, classification binaire — 4 datasets, cas normal / limite / adversarial sur chaque phase, fight final avec leaderboard

---

## Jour 1 — Premier pipeline et Arène

Notebook : `j1_ml_arena_breast_cancer.ipynb`

| Dataset | Type | Exemples | Features |
|---|---|---|---|
| `load_breast_cancer` | Classification binaire (bénigne / maligne) | 569 | 30 |
| `load_wine` | Classification multi-classe (3 cépages) | 178 | 13 |

- Pipeline supervisé complet : split stratifié, entraînement, prédiction, accuracy
- Arène : Régression logistique, KNN, Arbre de décision comparés sur le même split
- Clustering KMeans sans étiquettes pour retrouver les classes naturelles

---

## Jour 2 — Preprocessing Telco Customer Churn

Notebook : `j2_telco_churn_preprocessing.ipynb`  
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

---

## Arène des Algos — Jour 3

Notebook : `j3_arene_des_algos.ipynb`

### Datasets utilisés

| Dataset | Type | Exemples | Features |
|---|---|---|---|
| California Housing (`fetch_california_housing`) | Régression | 20 640 | 8 |
| AirBnB Ottawa (Inside Airbnb) | Clustering | 2 440 | 5 |
| SMS Spam Collection (UCI) | Classification texte | 5 572 | TF-IDF |
| Sonar (UCI id=151) | Classification binaire | 208 | 60 |

### Ce qui a été fait

- **Phase A — Régression** : LinearRegression (R²=0.58) vs RandomForest (R²=0.81) sur California Housing ; cas limite 100 lignes, cas adversarial quartier fictif
- **Phase B — Clustering** : KMeans sur listings AirBnB Ottawa, k choisi par méthode du coude (k=6), 6 segments décrits ; cas limite sans StandardScaler, cas adversarial outlier à 100 000 €/nuit
- **Phase C — Spam** : TF-IDF + Naive Bayes (recall spam=0.77) vs Régression logistique (recall spam=0.83) ; cas limite message vide, cas adversarial spam déguisé en français
- **Phase D — Sonar** : LR / SVC(rbf) / RandomForest avec et sans StandardScaler ; SVC chute de 0.93→0.83 sans scaling ; cas adversarial écho à zéro
- **Phase E — Fight des IA** : leaderboard sur même split Sonar, F1 + accuracy + timing — champion : SVC_rbf (F1=0.936)

