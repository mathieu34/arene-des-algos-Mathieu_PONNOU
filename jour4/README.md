# Jour 4 - Évaluation et mise en production

## Plan de l'après-midi

### Phase 0 : Mise en route
- Création du dossier jour4/
- Installation des dépendances : scikit-learn, tensorflow, joblib, flask, streamlit

### Phase 1 : Séparer les données proprement
- Split train / validation / test
- Stratification pour conserver la répartition des classes

### Phase 2 : Bootstrap et bagging
- Évaluation de la stabilité par rééchantillonnage avec remise
- Comprendre le bagging (base du Random Forest)

### Phase 3 : Validation croisée k-fold
- cross_val_score sur k=5 folds
- Moyenne et écart-type pour mesurer la stabilité du modèle

### Phase 4 : Choisir la bonne métrique selon le coût métier
- Pourquoi l'accuracy ment sur données déséquilibrées
- Precision, recall, F1, coût métier
- ROC/AUC et courbe de lift

### Phase 5 : Sérialiser le modèle et le servir derrière une API
- Sauvegarde avec joblib
- Endpoint Flask /predict

### Phase 6 : WebApp de prédiction
- Interface Streamlit avec champs de saisie
- Déploiement Streamlit Community Cloud

### Phase 7 : Arbitrage final
- Random Forest vs PMC Keras en validation croisée
- Leaderboard final : accuracy, recall, coût métier, temps, latence

## Dataset
- Breast Cancer Wisconsin (scikit-learn) — ou dataset alternatif au choix
- Classification binaire, 569 lignes, 30 features numériques

## Stack
- scikit-learn, tensorflow/keras, joblib, flask, streamlit