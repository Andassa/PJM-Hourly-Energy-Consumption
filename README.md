# Prévision de la consommation énergétique (PJM) — LSTM, Bi-LSTM et Attention

Projet académique de groupe — Réseaux de neurones artificiels (RNA).

**État du dépôt :** documentation et données brutes présentes ; le code applicatif (`energy_forecast/`, scripts, notebooks) est à produire selon ce guide. Ce fichier sert de référence unique pour cadrer le travail, les livrables et l’ordre d’exécution.

---

## Table des matières

1. [Objectif et question de recherche](#1-objectif-et-question-de-recherche)  
2. [Données](#2-données)  
3. [Structure cible du projet](#3-structure-cible-du-projet)  
4. [Pipeline technique (ordre strict)](#4-pipeline-technique-ordre-strict)  
5. [Prétraitement et feature engineering](#5-prétraitement-et-feature-engineering)  
6. [Fenêtrage, cible et split temporel](#6-fenêtrage-cible-et-split-temporel)  
7. [Les trois modèles](#7-les-trois-modèles)  
8. [Entraînement et critères de qualité](#8-entraînement-et-critères-de-qualité)  
9. [Évaluation, visualisations et tests statistiques](#9-évaluation-visualisations-et-tests-statistiques)  
10. [Rapport HTML et synthèse](#10-rapport-html-et-synthèse)  
11. [Répartition équipe et jalons](#11-répartition-équipe-et-jalons)  
12. [Environnement et exécution](#12-environnement-et-exécution)  
13. [Pièges courants à éviter](#13-pièges-courants-à-éviter)  
14. [Références](#14-références)  

---

## 1. Objectif et question de recherche

Les opérateurs de réseau ont besoin de prévisions fiables de la charge pour planifier la production, limiter le stress sur le réseau et arbitrer les achats. Ce projet met en œuvre des modèles séquentiels profonds sur des mesures horaires réelles.

**Question posée :** parmi un LSTM classique, un Bi-LSTM et un LSTM assorti d’une couche d’attention (Bahdanau), quelle architecture donne la meilleure précision pour une **prévision multi-horizon** (ici 24 heures), à **conditions expérimentales identiques** (mêmes données, même découpe temporelle, même procédure d’entraînement sauf architecture) ?

Le livrable final n’est pas seulement un meilleur score : il inclut des courbes, des métriques comparables, une interprétation des poids d’attention et une discussion des compromis (coût, risque de surapprentissage, sens physique des pics).

---

## 2. Données

### 2.1 Source et fichier principal

- **Jeu :** *PJM Hourly Energy Consumption* (Kaggle), par Robik Scube / communauté Kaggle.  
- **URL :** https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption  
- **Fichier requis pour ce projet :** `PJME_hourly.csv` (zone PJM East, consommation horaire).

Dans ce dépôt, une copie des CSV Kaggle se trouve déjà sous :

`Hourly Energy Consumption/PJME_hourly.csv`

Lorsque la structure `energy_forecast/` sera en place, la convention est de placer (ou de lier) ce fichier dans `energy_forecast/data/raw/PJME_hourly.csv` pour que les scripts et la documentation restent alignés.

### 2.2 Schéma attendu

| Colonne   | Type (logique) | Description                          |
|-----------|----------------|--------------------------------------|
| `Datetime` | datetime      | Horodatage à pas horaire             |
| `PJME_MW`  | float           | Consommation en mégawatts (MW)       |

Vérification rapide : les deux premières lignes doivent ressembler à `Datetime,PJME_MW` puis une date-heure et une valeur numérique.

### 2.3 Ordre de grandeur

- Période couverte : **2002 — 2018** (vérifier sur votre copie après chargement).  
- Fréquence : **1 point par heure**.  
- Volume : **environ 145 000 lignes** (le fichier du dépôt contient ~145k enregistrements).  
- Plage de valeurs typique : l’ordre de grandeur se situe entre ~19 000 MW et ~57 000 MW selon saison et heure ; les statistiques exactes sont à recalculer dans le notebook d’EDA.

### 2.4 Qualité des données

Avant tout modèle : contrôler doublons d’index, trous dans la série, valeurs manquantes et incohérences (ex. heures manquantes après resampling). Documenter dans le notebook toute correction appliquée (interpolation, suppression d’intervalles, etc.) pour que l’équipe reproduise le même jeu.

---

## 3. Structure cible du projet

Le code n’est pas encore présent ; la cible est la suivante (à créer à la racine du dépôt ou dans un sous-dossier dédié, selon choix d’équipe, mais **une seule arborescence** pour éviter la confusion) :

```
energy_forecast/
├── data/
│   ├── raw/
│   │   └── PJME_hourly.csv
│   └── processed/
│       ├── train.pkl
│       ├── val.pkl
│       └── test.pkl
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── results/
│   ├── plots/
│   ├── models/
│   ├── metrics.csv
│   └── report.html
├── requirements.txt
├── config.yaml
└── README.md
```

**Note :** le présent `README.md` à la racine du dépôt peut rester la documentation globale ; à l’ouverture du dossier `energy_forecast/`, un lien ou une copie courte peut renvoyer ici. L’important est qu’**une** entrée documentaire fasse foi pour l’équipe.

---

## 4. Pipeline technique (ordre strict)

Chaque étape dépend de la précédente. Respecter cet ordre évite les fuites d’information et les résultats non reproductibles.

| Étape | Contenu | Sortie attendue |
|-------|---------|-----------------|
| 1 | Chargement, tri chronologique, contrôle qualité | `DataFrame` indexé par le temps |
| 2 | Feature engineering à partir du calendrier | Colonnes supplémentaires |
| 3 | Split **chronologique** 80 / 10 / 10 | Indices train / val / test |
| 4 | `MinMaxScaler` (ou équivalent) **ajusté uniquement sur le train** | Paramètres de normalisation sauvegardés |
| 5 | Application du scaler au val et test | Séries normalisées cohérentes |
| 6 | Construction des fenêtres 168 → 24 | Tenseurs ou arrays + métadonnées |
| 7 | Entraînement des trois modèles | Checkpoints `.pth`, logs |
| 8 | Évaluation sur le **test** + visualisations | Figures, `metrics.csv`, rapport HTML |

**Règle d’or :** aucune statistique issue du validation set ou du test set ne doit entrer dans le calcul des paramètres de normalisation ou dans le choix de features dérivées de la consommation future.

---

## 5. Prétraitement et feature engineering

### 5.1 Chargement

- Lire le CSV avec `parse_dates=['Datetime']` et utiliser `Datetime` comme index si cela simplifie le resampling et le tri.  
- Trier par ordre chronologique croissant.  
- S’assurer qu’il n’y a pas de décalage de fuseau ambigu : le jeu Kaggle est généralement traité comme des timestamps « naïfs » localement cohérents ; documenter le choix.

### 5.2 Variables calendaires (à partir de l’index temporel)

À produire au minimum (noms suggérés, ajustables si le code est homogène) :

| Feature        | Description |
|----------------|-------------|
| `hour`         | 0–23 |
| `dayofweek`    | 0 = lundi … 6 = dimanche (ou convention pandas explicite) |
| `month`        | 1–12 |
| `year`         | année entière |
| `quarter`      | 1–4 |
| `is_weekend`   | 1 si samedi ou dimanche, sinon 0 |
| `is_holiday`   | 1 si jour férié US (bibliothèque `holidays`), sinon 0 |

La cible principale reste `PJME_MW` ; les autres colonnes servent de **covariables** en entrée du réseau (après encodage adapté : par ex. normalisation ou embedding selon choix d’équipe, à figer dans `config.yaml`).

### 5.3 Normalisation Min-Max (cible MW)

Pour la série de consommation (et éventuellement d’autres features continues si l’équipe décide de les mettre à la même échelle) :

\[
x_{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
\]

Les valeurs \(x_{\min}\) et \(x_{\max}\) sont calculées **sur le sous-ensemble train uniquement**, puis appliquées au validation et au test. Sauvegarder le scaler (pickle ou attributs dans un fichier) pour dénormaliser les prédictions avant les métriques en MW.

---

## 6. Fenêtrage, cible et split temporel

### 6.1 Fenêtre et horizon

- **Longueur d’entrée :** `window_size = 168` (une semaine d’heures).  
- **Horizon de prédiction :** `horizon = 24` (les 24 prochaines heures de consommation).

Chaque échantillon d’entraînement est donc :

- **Entrée :** une matrice de forme `(168, n_features)` — typiquement les features listées plus haut + la série MW normalisée dans la fenêtre, selon la convention retenue (souvent la colonne MW et les features calendaires alignées pas à pas).  
- **Sortie :** un vecteur de longueur **24** (valeurs MW futures normalisées, puis dénormalisées à l’évaluation).

La convention exacte (est-ce que la fenêtre inclut uniquement le passé strict jusqu’à \(t\), et la cible \(t+1 \ldots t+24\) ?) doit être **écrite dans le code et dans le notebook** pour éviter un décalage d’une heure.

### 6.2 Split chronologique 80 % / 10 % / 10 %

- **Pas de mélange aléatoire** des lignes : le découpage se fait sur l’axe du temps.  
- Répartition indicative alignée sur l’énoncé pédagogique :

  - Train : environ **2002 → 2016** (80 %)  
  - Validation : **2016 → 2017** (10 %)  
  - Test : **2017 → 2018** (10 %)  

Les dates exactes dépendent du nombre d’enregistrements réels ; l’implémentation recommandée est de couper sur des **indices temporels** (80 % / 10 % / 10 % des timestamps) plutôt que sur des années arrondies à la main, tout en vérifiant que le résultat reste proche de la répartition ci-dessus.

---

## 7. Les trois modèles

Framework : **PyTorch**. Toutes les architectures partagent les mêmes dimensions d’entrée/sortie et le même prétraitement.

### 7.1 Modèle 1 — LSTM (référence)

- Empilement : **2 couches** LSTM.  
- Taille de l’état caché : **128**.  
- **Dropout 0.2** entre couches (selon support PyTorch, appliquer de manière documentée).  
- Tête de régression : couches fully-connected (ex. `128 → 64` avec activation ReLU, puis `64 → 24`).  
- Agrégation temporelle : typiquement **dernier état caché** de la séquence (à préciser dans le code et figer pour toute l’équipe).

### 7.2 Modèle 2 — Bi-LSTM

- Même profondeur (2 couches) en **bidirectionnel**.  
- La dimension concaténée avant la tête fully-connected est **256** (128 avant + 128 arrière) si concaténation standard.  
- Même tête de sortie vers **24** valeurs.  
- Attention au fait que la bidirectionnalité utilise des informations « futures » dans la fenêtre d’entrête : c’est acceptable **tant que la fenêtre ne contient que le passé relatif à l’instant de prédiction** ; ce point doit être clair dans le rapport (sinon risque de fuite conceptuelle si mal défini).

### 7.3 Modèle 3 — LSTM + attention de Bahdanau

- LSTM empilé produit une séquence de sorties cachées \(h_1, \ldots, h_{168}\).  
- Couche d’attention implémentée **explicitement** (pas seulement un module « black box » sans exposition des poids), avec sortie :

  - un **vecteur de contexte** utilisé pour la régression ;  
  - les **poids d’attention** \(\alpha\) pour visualisation (heatmap sur les 168 pas).

Équations de référence (notation compacte) :

- Score : \(e_{t,s} = v^\top \tanh(W_h h_s + W_q q_t)\)  
- Poids : \(\alpha_{t,s} = \frac{\exp(e_{t,s})}{\sum_k \exp(e_{t,k})}\)  
- Contexte : \(c_t = \sum_s \alpha_{t,s} h_s\)

Le vecteur requête \(q_t\) est à définir dans l’implémentation (souvent le dernier état caché ou une projection de celui-ci) ; **une fois choisi, ne pas changer entre runs comparatifs** sans le noter.

### 7.4 Rappel LSTM (rédaction cours)

Portes : oubli \(f_t\), entrée \(i_t\), candidat \(\tilde{C}_t\), sortie \(o_t\), état cellule \(C_t\), état caché \(h_t\), avec \(\sigma\) sigmoïde et \(\odot\) produit terme à terme. Les détails complets sont laissés au support de cours ; le code utilisera `nn.LSTM`.

---

## 8. Entraînement et critères de qualité

### 8.1 Hyperparamètres (à centraliser dans `config.yaml`)

Exemple de champs à prévoir :

- `epochs` : 100 (plafond ; early stopping peut stopper avant)  
- `batch_size` : 32  
- `learning_rate` : 0.001  
- `optimizer` : Adam  
- `scheduler` : `ReduceLROnPlateau` (sur la métrique de validation retenue, ex. loss)  
- `early_stopping_patience` : 10  
- `seed` : 42 (et fixer les graines `random`, `numpy`, `torch`, `torch.cuda` si GPU)

### 8.2 Boucle d’entraînement

Pour **chaque** modèle :

1. Entraîner en minimisant une loss adaptée à la régression (MSE ou L1, à décider et à **garder identique** pour les trois modèles).  
2. À chaque époque : loss train, loss validation.  
3. Sauvegarder le **meilleur** checkpoint sur la validation dans `results/models/`.  
4. Arrêt anticipé si la validation n’améliore plus le critère pendant `patience` époques.

### 8.3 Journalisation

Fichier type `results/training_logs.csv` ou un sous-dossier par modèle : époque, loss train, loss val, learning rate, temps écoulé.

---

## 9. Évaluation, visualisations et tests statistiques

Toutes les métriques sur le **test** sont calculées **après dénormalisation** des prédictions et des cibles, en MW.

### 9.1 Métriques

- **MAE** : \(\frac{1}{n} \sum_i |y_i - \hat{y}_i|\)  
- **RMSE** : \(\sqrt{\frac{1}{n} \sum_i (y_i - \hat{y}_i)^2}\)  
- **MAPE** : \(\frac{100\%}{n} \sum_i \left| \frac{y_i - \hat{y}_i}{y_i} \right|\) — attention aux \(y_i\) proches de zéro (non le cas ici en MW, mais garder la formule défensive si réutilisation).

### 9.2 Graphiques minimum

1. Série **réelle vs prédite** sur une portion représentative du test (ou sur tout le test si lisible avec sur-échantillonnage).  
2. Courbes **train / validation loss** par modèle (ou un graphique comparatif propre).  
3. **Heatmap** des poids d’attention (modèle 3), au moins pour un ou plusieurs batches exemplaires.  
4. **Tableau comparatif** des trois modèles (export CSV + figure tabulaire si besoin).  
5. Option pédagogique : prédictions à plusieurs horizons (1 h, 24 h, etc.) si le code le permet sans refonte lourde.

### 9.3 Test de Diebold–Mariano

Comparer les erreurs de prévision (par ex. sur le test) entre paires de modèles pour tester si une différence de performance est **statistiquement significative**, et non due au hasard d’échantillonnage. Documenter la fonction de perte utilisée dans le test (souvent carré des erreurs ou valeur absolue) et interpréter avec prudence (hypothèses de stationnarité faible des erreurs).

---

## 10. Rapport HTML et synthèse

Produire un fichier `results/report.html` qui regroupe :

- résumé du contexte et de la question ;  
- table des hyperparamètres ;  
- schémas ou texte décrivant chaque architecture ;  
- figures intégrées (chemins relatifs vers `results/plots/`) ;  
- tableau des métriques ;  
- brève discussion des poids d’attention et du test de Diebold–Mariano.

Outils possibles : génération programmatique HTML, ou export depuis notebooks avec figures figées. L’équipe choisit une méthode **reproductible** (script `evaluate.py` ou notebook final).

---

## 11. Répartition équipe et jalons

### 11.1 Modules et responsabilités

| Module | Rôle | Fichiers principaux |
|--------|------|---------------------|
| **A — Données** | Téléchargement / copie vers `data/raw`, EDA, preprocessing, fenêtres, splits, artefacts `.pkl` | `notebooks/01_EDA.ipynb`, `src/preprocessing.py`, `src/feature_engineering.py` |
| **B — Modèles** | Trois classes PyTorch + test de formes sur batch fictif | `src/models.py` |
| **C — Entraînement** | Config, boucles, early stopping, checkpoints, logs | `config.yaml`, `src/train.py` |
| **D — Évaluation** | Métriques, figures, CSV, rapport HTML, Diebold–Mariano | `src/evaluate.py`, `results/*` |

### 11.2 Jalons suggérés

| Semaine | Objectif mesurable |
|---------|---------------------|
| 1 | Données nettoyées, EDA rédigée, `train/val/test` sérialisés et décrits |
| 2 | `models.py` complet, tests de dimensions OK |
| 3 | Trois entraînements terminés, checkpoints et logs archivés |
| 4 | Figures finales, `metrics.csv`, rapport HTML, présentation |

### 11.3 Critères de « terminé » par module

- **A :** notebook EDA avec graphiques de série complète, profil par heure, profil par mois, section anomalies ; code preprocessing reproductible ; fichiers dans `data/processed/`.  
- **B :** `forward` clair ; pour l’attention, retour explicite des poids ; assertion sur tenseur de sortie `(batch, 24)`.  
- **C :** meilleur modèle sauvegardé par architecture ; early stopping et scheduler actifs ; graines fixées.  
- **D :** toutes les métriques et figures listées ; test DM documenté ; rapport HTML remis.

---

## 12. Environnement et exécution

### 12.1 Prérequis

- Python **3.10+**  
- Git  
- Au moins **4 Go de RAM** (plus confortable pour entraînement local ; GPU optionnel)

### 12.2 Installation (à valider quand `requirements.txt` existera)

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Linux / macOS : `source venv/bin/activate`

### 12.3 Dépendances prévues

```
torch>=2.0.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0
scikit-learn>=1.3.0
holidays>=0.30
pyyaml>=6.0
jupyter>=1.0.0
tqdm>=4.65.0
```

### 12.4 Commandes cibles (après implémentation)

```bash
python src/preprocessing.py
python src/train.py --model lstm
python src/train.py --model bilstm
python src/train.py --model attention
python src/evaluate.py
```

Les arguments exacts (`--model`, chemins, device) sont à harmoniser dans `train.py` et documentés en en-tête de ce README lorsque le code sera stabilisé.

---

## 13. Pièges courants à éviter

1. **Scaler ajusté sur tout le dataset** — invalide la comparaison et gonfle artificiellement les performances.  
2. **Shuffle global** des fenêtres sans garde-fou — mélange passé/futur ; si shuffle, le faire **à l’intérieur** du train uniquement après split.  
3. **Fuite via la validation** — tuner des hyperparamètres sur le test ; réserver le test à la toute fin.  
4. **MAPE** — vérifier les divisions par des valeurs très petites si autre jeu.  
5. **Bi-LSTM** — clarifier dans le rapport ce que représente la bidirectionnalité dans la fenêtre.  
6. **Reproductibilité** — sans graines fixes et sans version des librairies (`pip freeze` ou équivalent), les comparaisons inter-membres se dégradent.

---

## 14. Références

1. Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation.  
2. Bahdanau, D., Cho, K., & Bengio, Y. (2015). *Neural Machine Translation by Jointly Learning to Align and Translate.* ICLR.  
3. Schuster, M., & Paliwal, K. K. (1997). *Bidirectional Recurrent Neural Networks.* IEEE Transactions on Signal Processing.  
4. Mulla, R. et communauté Kaggle (2018). *PJM Hourly Energy Consumption.* Jeu de données Kaggle : https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption  

---

## Annexe — Ordre de lecture pour un nouveau membre

1. Lire les sections **1**, **2**, **4** et **6** (cadrage + données + pipeline + fenêtres).  
2. Parcourir **11** pour voir où son module s’insère.  
3. Lire **7** et **8** en détail si responsable modèles ou entraînement.  
4. Lire **9** et **10** si responsable évaluation et livrable final.  
5. Garder **13** sous les yeux pendant toute l’implémentation.

Ce document peut être complété au fil du projet (chemins d’exécution définitifs, figures clés, résultats numériques obtenus). Toute modification de convention (nom des colonnes, loss, agrégation LSTM) doit être **tracée ici ou dans `config.yaml`** pour rester le contrat d’équipe unique.
