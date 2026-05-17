# Prévision de la consommation énergétique (PJM), LSTM, Bi-LSTM et attention

Projet académique de groupe en réseaux de neurones artificiels (RNA).

**Navigation rapide :** [Exécution et avancement](#execution-avancement) · [Par où commencer](#guide-debut) · [Recette pas à pas](#recette-projet) · [Glossaire](#glossaire-projet) · [Jargon détaillé](#jargon-detail) · [Mathématiques](#maths-projet) · [Table des matières](#table-des-matieres)

**État du dépôt :** le pipeline `energy_forecast/` est **implémenté et exécuté** (EDA, prétraitement, fenêtres 168→24, trois modèles PyTorch entraînés et évalués sur le test). Il reste principalement le **rapport** (synthèse, figures de loss, heatmap d’attention si demandée), le test **Diebold–Mariano** et éventuellement `report.html` — voir [Exécution et avancement](#execution-avancement).

---

<a id="execution-avancement"></a>
## Exécution et avancement (référence équipe)

### Installation

À la **racine du dépôt** (là où se trouve `requirements.txt`) :

```bash
python -m venv .venv
# Windows (Git Bash) : source .venv/Scripts/activate
# Windows (PowerShell) : .\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Sous **Git Bash**, utiliser des **slashes** `/` dans les chemins (`energy_forecast/src/train.py`), pas `\`.

### Chaîne de commandes (données → modèles)

Depuis la racine du dépôt, après avoir produit `energy_forecast/data/processed/PJME_hourly_clean.csv` (notebook EDA) :

```bash
python energy_forecast/src/preprocessing.py --all
# ou séparément : preprocessing.py puis build_splits_sequences.py

python energy_forecast/src/train.py --config energy_forecast/config.yaml
python energy_forecast/src/evaluate.py --config energy_forecast/config.yaml
```

Pour chaque architecture, modifier `model.name` dans `energy_forecast/config.yaml` : `lstm`, `bilstm`, `lstm_attention`, puis relancer train + evaluate. Les checkpoints vont dans `energy_forecast/results/checkpoints/best_<nom>.pth`.

`device` dans `config.yaml` : `cpu` par défaut ; mettre `cuda` si PyTorch GPU est installé.

### Résultats test (PJME, split chronologique, `n_points` = 344 376)

| Modèle | MAE (MW) | RMSE (MW) | MAPE (%) |
|--------|----------|-----------|----------|
| LSTM | **1417** | **1982** | **4,50** |
| Bi-LSTM | 1424 | 1999 | 4,51 |
| LSTM + attention | 1457 | 2020 | 4,61 |

Détail et historique des runs : `energy_forecast/results/metrics.csv`. Figures : `energy_forecast/results/plots/`.

### Recette : statut des étapes

| # | Étape | Statut |
|---|--------|--------|
| 0–2 | Environnement, structure, données `data/raw/` | Fait |
| 3–7 | EDA, features, split, Min-Max, fenêtres `.pkl` | Fait |
| 8–10 | `models.py`, train, evaluate, `metrics.csv` | Fait (3 modèles) |
| 11 | Figures scatter / résidus / fenêtre 24 h | Fait ; courbes loss + heatmap attention : optionnel |
| 12–13 | Diebold–Mariano, `report.html` | À faire (livrable rapport) |

---

<a id="guide-debut"></a>
## Par où commencer (lecture obligatoire)

**Si tu arrives sur le projet sans contexte technique :** lis dans cet ordre : [Recette du projet](#recette-projet) (fil d’A à Z) → [Glossaire](#glossaire-projet) (table mémo) → [Jargon expliqué en détail](#jargon-detail) (LSTM, Bi-LSTM, attention, PyTorch, métriques, fichiers) → [Mathématiques](#maths-projet) (formules) → puis [1. Objectif](#1-objectif-et-question-de-recherche) et la suite.

**Si tu es assigné à un module précis :** ouvre d’abord [11. Répartition équipe](#11-répartition-équipe-et-jalons), repère ton module (A/B/C/D), puis suis uniquement les **étapes de la recette** qui concernent ce module ; reviens au glossaire et aux maths quand un terme bloque.

**Première action concrète (nouveau membre) :** cloner le dépôt, installer les dépendances ([§ installation](#execution-avancement)), vérifier `energy_forecast/data/raw/PJME_hourly.csv`, puis suivre la [chaîne de commandes](#execution-avancement). Pour l’EDA : `energy_forecast/notebooks/01_eda_pjm_hourly_load.ipynb`.

**Critère « on a démarré correctement » :** une personne de l’équipe peut expliquer en une phrase ce qu’est une fenêtre **168 → 24**, pourquoi le découpage est **chronologique**, et pourquoi le **Min-Max** se calcule sur le **train seulement**.

---

<a id="recette-projet"></a>
## Recette du projet : de zéro au livrable final

Analogie : le **plat final** = rapport HTML + table `metrics.csv` + graphiques + 3 modèles entraînés comparés (LSTM, Bi-LSTM, LSTM+Attention) + interprétation des poids d’attention. Les **ingrédients** = données PJME, Python, librairies listées plus bas. Les **ustensiles** = notebooks + dossier `src/` + `config.yaml`.

| # | Étape (à faire dans l’ordre) | Tu obtiens quoi ? | Où c’est détaillé dans ce README |
|---|------------------------------|-------------------|----------------------------------|
| 0 | Installer Python, Git ; créer `venv` ; préparer `requirements.txt` | machine prête | [12. Environnement](#12-environnement-et-exécution) |
| 1 | Créer l’arborescence `energy_forecast/` (dossiers `data`, `src`, `notebooks`, `results`) | structure du code | [3. Structure cible](#3-structure-cible-du-projet) |
| 2 | Copier `PJME_hourly.csv` dans `data/raw/` (depuis `Hourly Energy Consumption/` si besoin) | chemin unique pour les scripts | [2. Données](#2-données) |
| 3 | Notebook EDA : série complète, saisonnalité, anomalies | décisions de nettoyage documentées | [2.4](#24-qualité-des-données), module A |
| 4 | Charger, trier par temps ; créer features calendaires + `is_holiday` | tableau enrichi | [5. Prétraitement](#5-prétraitement-et-feature-engineering) |
| 5 | Découper **dans le temps** 80 % / 10 % / 10 % (sans mélanger aléatoirement tout le jeu) | bornes train / val / test | [6.2](#62-split-chronologique-80--10--10-) |
| 6 | Calculer Min-Max **sur le train** ; appliquer au val et test | pas de fuite d’information | [5.3](#53-normalisation-min-max-cible-mw), [Maths M1](#maths-projet) |
| 7 | Construire fenêtres : entrée **168** pas, sortie **24** pas | tenseurs `(N, 168, F)` et `(N, 24)` | [6.1](#61-fenêtre-et-horizon) |
| 8 | Implémenter les 3 modèles PyTorch + test sur batch aléatoire | `models.py` validé | [7. Les trois modèles](#7-les-trois-modèles) |
| 9 | Entraîner chaque modèle (Adam, scheduler, early stopping, checkpoints) | fichiers `.pth` + logs | [8. Entraînement](#8-entraînement-et-critères-de-qualité) |
| 10 | Évaluer sur le **test** en MW (dénormaliser) ; MAE, RMSE, MAPE | `metrics.csv` | [9. Évaluation](#9-évaluation-visualisations-et-tests-statistiques) |
| 11 | Figures : réel vs prédit, losses, heatmap attention, tableau comparatif | dossier `results/plots/` | idem |
| 12 | Test de Diebold–Mariano entre modèles | phrase « significatif ou non » appuyée sur un test | [9.3](#93-test-de-dieboldmariano) |
| 13 | Générer `report.html` (résumé + hyperparamètres + figures) | livrable lisible par l’encadrant | [10. Rapport HTML](#10-rapport-html-et-synthèse) |

**Règle de recette :** ne pas passer à l’étape *n* + 1 si l’étape *n* n’a pas de sortie vérifiable (fichier, graphique ou métrique) partagée avec l’équipe.

---

<a id="glossaire-projet"></a>
## Glossaire des termes

Les définitions ci-dessous sont celles **utilisées dans ce projet** ; elles peuvent différer légèrement du vocabulaire d’un autre cours.

| Terme | Signification courte |
|--------|----------------------|
| **PJM / PJME** | Marché / zone de transport d’électricité aux États-Unis ; **PJME** = sous-zone « East », fichier `PJME_hourly.csv`. |
| **MW** | Mégawatt, unité de puissance ; la colonne `PJME_MW` est la charge observée à l’instant *t*. |
| **Série temporelle** | Suite de valeurs indexées par le temps (ici une mesure par heure). |
| **Feature engineering** | Créer des colonnes explicatives à partir des dates (heure, mois, week-end, férié, etc.). |
| **Covariable** | Entrée du modèle en plus de la consommation (souvent les features calendaires). |
| **Normalisation Min-Max** | Transformer les nombres dans un intervalle (souvent [0, 1]) pour stabiliser l’entraînement ; voir formules en [Mathématiques](#maths-projet). |
| **Fuite d’information (*data leakage*)** | Utiliser une information du **futur** ou du **test** pour construire le train (ex. Min-Max calculé sur tout le jeu) ; les scores deviennent irréalistes. |
| **Fenêtre glissante** | Tranche de **168** heures consécutives utilisée comme entrée ; on la fait « glisser » d’une ligne à l’autre le long du temps. |
| **Horizon de prédiction** | Nombre d’instants futurs prédits ici : **24** heures après la fin de la fenêtre d’entrée (convention exacte à figer dans le code). |
| **Train / validation / test** | **Train** = apprentissage des poids ; **validation** = choix early stopping et suivi de la généralisation pendant l’entraînement ; **test** = mesure finale **une fois** le modèle figé. |
| **Split chronologique** | Couper le temps en blocs successifs (passé → futur), sans tirage aléatoire des lignes, pour respecter la causalité. |
| **RNA / deep learning** | Réseau de neurones ; « profond » = plusieurs couches traitant les données. |
| **PyTorch** | Bibliothèque Python pour définir et entraîner les réseaux (tenseurs, `nn.Module`, etc.). |
| **Tenseur** | Tableau multidimensionnel généralisant matrice et vecteur ; ex. `(batch, 168, features)`. |
| **LSTM** | Type de couche récurrente qui garde une **mémoire** sur plusieurs pas de temps ; adaptée aux séries longues. |
| **État caché** | Vecteur résumant l’information du passé récent à l’instant *t* dans la LSTM. |
| **Bi-LSTM** | Deux LSTM, une lecture **chronologique** et une **anti-chronologique** ; leurs sorties sont concaténées. |
| **Attention (Bahdanau)** | Mécanisme qui calcule des **poids** sur chaque pas du passé pour construire un **vecteur de contexte** ; permet de visualiser « quelles heures comptent ». |
| **Softmax** | Fonction qui transforme des scores en **probabilités** (somme = 1) ; utilisée pour les poids d’attention. |
| **Dropout** | Désactivation aléatoire de neurones pendant l’entraînement pour limiter le surapprentissage. |
| **Batch** | Sous-groupe d’échantillons traité ensemble (ex. 32 fenêtres) à chaque mise à jour partielle. |
| **Époque (*epoch*)** | Un passage complet sur l’ensemble d’entraînement. |
| **Loss** | Fonction d’erreur à minimiser (souvent MSE entre prédictions et vérité sur le train). |
| **Adam** | Algorithme d’optimisation des poids (variante de descente de gradient). |
| **Learning rate** | Pas de mise à jour des poids ; ici typiquement 0.001 au départ. |
| **ReduceLROnPlateau** | Stratégie qui **diminue** le learning rate quand la validation stagne. |
| **Early stopping** | Arrêter l’entraînement si la validation ne s’améliore plus pendant *patience* époques (évite de trop surapprendre). |
| **Checkpoint** | Fichier `.pth` sauvegardant les poids du meilleur modèle sur la validation. |
| **MAE / RMSE / MAPE** | Trois façons de mesurer l’erreur entre vérité et prédiction (voir [Mathématiques](#maths-projet)). |
| **Diebold–Mariano** | Test statistique pour comparer deux séries d’erreurs de prévision ; indique si une différence de performance est **compatible avec le hasard** ou non. |
| **EDA** | *Exploratory Data Analysis* : exploration visuelle et statistique avant modélisation. |
| **Heatmap** | Image où une couleur représente l’intensité (ici les poids d’attention sur 168 heures). |

Le tableau ci-dessus sert de **mémo**. Les sous-sections suivantes expliquent le **jargon** comme dans un cours : à lire au moins une fois au début du projet, puis en retour lors de la rédaction du rapport.

<a id="jargon-detail"></a>
### Acronymes et sigles (liste rapide)

| Sigle | Écriture complète | Rôle dans ce projet |
|--------|-------------------|---------------------|
| **RNA** | Réseau de neurones artificiels | Famille de modèles ; ici on utilise des RNA **récurrents** profonds. |
| **LSTM** | *Long Short-Term Memory* (mémoire court et long terme) | Couche qui traite une **séquence** heure par heure en gardant une mémoire interne. |
| **Bi-LSTM** | LSTM **bidirectionnelle** | Deux sens de lecture de la séquence ; le nom **Bi** vient de *bi* = deux. |
| **MW** | Mégawatt | Unité de la colonne de charge `PJME_MW`. |
| **MSE** | *Mean Squared Error* (erreur quadratique moyenne) | Loss d’entraînement classique pour la régression (voir [M2](#maths-projet)). |
| **MAE** | *Mean Absolute Error* | Métrique d’évaluation : erreur moyenne en valeur absolue, en MW après dénormalisation. |
| **RMSE** | *Root Mean Squared Error* | Métrique : racine de l’erreur quadratique moyenne ; pénalise fort les grosses erreurs. |
| **MAPE** | *Mean Absolute Percentage Error* | Erreur relative moyenne en % ; utile pour comparer à l’échelle relative. |
| **EDA** | *Exploratory Data Analysis* | Phase d’exploration des données avant modélisation. |
| **DM** | Test de Diebold-Mariano | Test statistique pour comparer deux modèles de prévision. |
| **GPU / CPU** | Processeur graphique / central | Le GPU accélère PyTorch si disponible ; le CPU suffit pour des tests plus lents. |

### Réseau récurrent, LSTM, Bi-LSTM : qu’est-ce que c’est ?

**RNN (*recurrent neural network*)**  
Un réseau « normal » (perceptron) traite une entrée fixe. Un **récurrent** traite une **séquence** : à chaque heure \(t\), il reçoit une entrée \(x_t\) et un **résumé du passé** (l’état caché). Il met à jour ce résumé et peut, à la fin, produire une prédiction. Problème classique des RNN simples : la mémoire **s’efface** sur de longues séries (gradient qui disparaît en apprentissage). La **LSTM** a été conçue pour mieux garder l’information sur de longues plages.

**LSTM (*Long Short-Term Memory*)**  
C’est un bloc de calcul avec des **portes** (oubli, entrée, sortie) et un **état cellule** séparé de l’**état caché**. Intuition : la cellule peut **stocker** une tendance (ex. saison, niveau de charge) pendant plusieurs pas, ou **l’oublier** si la porte d’oubli le décide, ce qui rend le modèle plus stable sur 168 heures d’historique. Dans ce projet, une ou **deux** couches LSTM empilées lisent la fenêtre de 168 pas et alimentent une **tête de régression** (couches linéaires) qui sort **24** valeurs futures.

**Bi-LSTM (*bidirectional LSTM*)**  
On fait tourner **une** LSTM du passé vers le futur et **une autre** du futur vers le passé **sur la même fenêtre d’entrée**. Les deux sorties sont **concaténées**. Intuition : le modèle peut croiser « ce qui ressemble à une montée vue depuis la gauche » et « vue depuis la droite ». Attention : tout reste **dans la fenêtre passée** (les 168 heures avant l’instant de prédiction) ; on ne regarde pas le futur **réel** à prédire. À expliquer clairement dans le rapport pour éviter la confusion avec une fuite de données.

**Attention de Bahdanau**  
Après la LSTM, on dispose d’un vecteur \(h_s\) par heure \(s\) de la fenêtre. L’**attention** calcule un **poids** \(\alpha_{t,s}\) par heure : les heures importantes pour la prédiction reçoivent un poids plus grand. Le **vecteur de contexte** est une moyenne pondérée des \(h_s\). Intuition : le modèle **choisit** quels moments du passé regarder (pics, veille de jour férié, etc.), au lieu de résumer toute la semaine seulement par le dernier état caché. Les \(\alpha_{t,s}\) se **visualisent** en heatmap.

### Vocabulaire entraînement et évaluation

**PyTorch**  
Bibliothèque qui représente les données en **tenseurs**, les opérations en graphe de calcul, et la **dérivée automatique** (*autograd*) pour ajuster les poids par **descente de gradient**. On définit un modèle en sous-classant `nn.Module`, on envoie des batchs sur `model(x)`, on calcule la loss, puis `loss.backward()` et `optimizer.step()`.

**Tenseur**  
Généralisation d’un vecteur ou d’une matrice à plusieurs dimensions. Ici des formes du type `(taille_batch, 168, nombre_de_features)`.

**Loss (*loss function*, fonction de coût)**  
Nombre que le réseau doit **minimiser**. Ici c’est typiquement la **MSE** entre les 24 valeurs prédites et les 24 vraies valeurs (sur le train). La loss sur le **validation** sert à l’**early stopping** et au **ReduceLROnPlateau**, pas à mettre à jour les poids directement.

**Batch**  
Le jeu d’entraînement est découpé en **lots** (par ex. 32 fenêtres). Une **mise à jour** des poids utilise un batch ; une **époque** a parcouru tous les batchs du train une fois.

**Adam**  
Méthode d’optimisation qui adapte le pas d’apprentissage par paramètre ; pratique par défaut en deep learning. Le **learning rate** est le pas global (souvent 0.001 au début).

**Early stopping**  
Si la loss de **validation** ne baisse plus pendant plusieurs époques (**patience**), on **arrête** pour ne pas surapprendre le train. On garde le **meilleur jeu de poids** vu sur la validation (**checkpoint** `.pth`).

**Surapprentissage (*overfitting*)**  
Le modèle mémorise le train et mal généralise. Le **dropout** et l’early stopping en limitent les effets.

**MinMaxScaler (scikit-learn)**  
Implémentation standard de la normalisation Min-Max du projet. On appelle `fit` sur le **train** seulement, puis `transform` sur val et test.

**Checkpoint `.pth`**  
Fichier binaire PyTorch qui sauvegarde les **poids** du réseau (et parfois l’état de l’optimiseur). Permet de recharger le meilleur modèle sans réentraîner.

### Données, fichiers et reste du pipeline

**Kaggle**  
Plateforme où se trouve le jeu *PJM Hourly Energy Consumption* ; le fichier utilisé ici est `PJME_hourly.csv`.

**CSV**  
Fichier texte tabulaire (séparateur virgule) ; `Datetime` et `PJME_MW` sont les deux colonnes principales.

**Pickle (`.pkl`)**  
Format Python pour sérialiser des objets (par ex. tenseurs ou DataFrames déjà fenêtrés). Les fichiers `train.pkl` / `val.pkl` / `test.pkl` accélèrent les relances d’entraînement.

**Fenêtre 168 et horizon 24**  
**168** = 7 jours × 24 h, une **semaine** d’historique comme entrée. **24** = nombre d’heures **futures** à prédire en sortie du réseau (multi-horizon court).

**Split chronologique 80 / 10 / 10**  
Les **premiers** instants dans le temps servent au train, la **tranche suivante** à la validation, la **dernière** au test. On ne tire pas les lignes au hasard, pour ne pas mélanger futur et passé.

**Rapport HTML**  
Page web unique qui regroupe texte, tableaux et images (`report.html`) pour présenter résultats et figures à l’encadrant sans ouvrir dix fichiers séparés.

---

<a id="maths-projet"></a>
## Mathématiques utilisées dans ce projet

Les formules sont écrites en **LaTeX** (affichage correct sur GitHub, VS Code avec aperçu math, et export PDF depuis plusieurs outils).

### Symboles et conventions (à utiliser partout dans le rapport)

| Symbole | Nom | Signification dans ce projet |
|:-------:|-----|------------------------------|
| $x_t$ | entrée au pas $t$ | vecteur des variables à l’heure $t$ (MW normalisé + features) |
| $y$, $\hat{y}$ | cible, prédiction | $\hat{y}$ se lit « y chapeau » ; valeurs sur l’horizon 24 h |
| $\sigma(\cdot)$ | sigmoïde | $\sigma(z) = \frac{1}{1 + e^{-z}}$ ; sortie dans $]0,\,1[$ |
| $\tanh(\cdot)$ | tangente hyperbolique | sortie dans $]-1,\,1[$ |
| $\odot$ | produit de Hadamard | multiplication **élément par élément** de deux vecteurs de même taille |
| $[\,h_{t-1}\,;\,x_t\,]$ | concaténation verticale | vecteur obtenu en empilant $h_{t-1}$ et $x_t$ (notée aussi $[h_{t-1}, x_t]$ selon les auteurs) |
| $W$, $b$ | poids et biais | matrices / vecteurs appris ; le produit $W\,[\,h_{t-1}\,;\,x_t\,] + b$ est un **produit matrice–vecteur** |
| $h_t$, $C_t$ | état caché, état cellule | sortie « visible » et mémoire interne de la LSTM |
| $\mathbf{v}^{\top}$ | transposée | ligne vectorielle ; $\mathbf{v}^{\top}\mathbf{u}$ est un scalaire (produit scalaire) |
| $\sum$, $\prod$ | somme, produit | agrégations sur les pas de temps ou les instants du test |
| $\|\cdot\|$ ou $\lvert \cdot \rvert$ | valeur absolue | pour MAE et MAPE sur des scalaires |

---

### M1 : Normalisation Min-Max (paramètres estimés sur le **train** uniquement)

Soit $x \in \mathbb{R}$ une valeur brute (ex. MW). Soient $x_{\min}$ et $x_{\max}$ le minimum et le maximum observés **sur le jeu d’entraînement** :

$$
x_{\mathrm{norm}} \;=\; \frac{x - x_{\min}}{x_{\max} - x_{\min}}
$$

**Dénormalisation** (revenir en MW) :

$$
x \;=\; x_{\min} + x_{\mathrm{norm}} \cdot \bigl(x_{\max} - x_{\min}\bigr)
$$

---

### M2 : Fonction de coût d’entraînement (régression sur 24 pas)

**MSE** (erreur quadratique moyenne) sur l’horizon $H=24$ pour un échantillon :

$$
\mathcal{L}_{\mathrm{MSE}} \;=\; \frac{1}{H}\sum_{h=1}^{H} \bigl( y_h - \hat{y}_h \bigr)^{2} \;=\; \frac{1}{24}\sum_{h=1}^{24} \bigl( y_h - \hat{y}_h \bigr)^{2}
$$

Sur un mini-batch de taille $B$, on moyenne en général sur le batch : $\displaystyle \mathcal{L} = \frac{1}{B}\sum_{b=1}^{B} \mathcal{L}_{\mathrm{MSE}}^{(b)}$.

**Alternative L1** (MAE comme loss) : $\displaystyle \mathcal{L}_{\mathrm{L1}} = \frac{1}{24}\sum_{h=1}^{24} \bigl\lvert y_h - \hat{y}_h \bigr\rvert$. À garder **identique** pour les trois modèles si l’on compare les losses.

---

### M3 : Cellule LSTM (quatre portes + états)

À l’instant $t$, entrée $x_t$, états précédents $h_{t-1}$ et $C_{t-1}$. On note $z_t = [\,h_{t-1}\,;\,x_t\,]$ le vecteur concaténé. Les poids $W_f, W_i, W_C, W_o$ et biais $b_f, b_i, b_C, b_o$ sont les paramètres appris.

**Porte d’oubli** (oublie une partie de la mémoire passée) :

$$
f_t \;=\; \sigma\!\bigl( W_f\, z_t + b_f \bigr)
$$

**Porte d’entrée** (filtre ce qui entre dans la cellule) :

$$
i_t \;=\; \sigma\!\bigl( W_i\, z_t + b_i \bigr)
$$

**Candidat** (nouvelle information proposée) :

$$
\tilde{C}_t \;=\; \tanh\!\bigl( W_C\, z_t + b_C \bigr)
$$

**État de la cellule** (mémoire mise à jour) :

$$
C_t \;=\; f_t \odot C_{t-1} \;+\; i_t \odot \tilde{C}_t
$$

**Porte de sortie** :

$$
o_t \;=\; \sigma\!\bigl( W_o\, z_t + b_o \bigr)
$$

**Sortie cachée** (ce qui est transmis aux couches suivantes) :

$$
h_t \;=\; o_t \odot \tanh(C_t)
$$

Dans PyTorch, `nn.LSTM` implémente ces relations ; les notations $W_f,\ldots$ correspondent à des blocs regroupés dans les matrices internes du module.

---

### M4 : Attention additive de Bahdanau (scores, softmax, contexte)

Soit $h_s \in \mathbb{R}^{d_h}$ la sortie cachée de la LSTM au pas $s$, avec $s \in \{1,\ldots,T\}$ et $T=168$. Soit $\mathbf{q}_t \in \mathbb{R}^{d_q}$ un vecteur **requête** (souvent $\mathbf{q}_t = h_T$ ou une couche linéaire appliquée à $h_T$). Soient $\mathbf{W}_h$, $\mathbf{W}_q$, $\mathbf{v}$ des paramètres appris (dimensions compatibles).

**Scores d’alignement** (un scalaire par couple $(t,s)$) :

$$
e_{t,s} \;=\; \mathbf{v}^{\top}\,\tanh\!\bigl( \mathbf{W}_h\, h_s + \mathbf{W}_q\, \mathbf{q}_t \bigr)
$$

**Poids d’attention** (softmax sur $s$, donc $\sum_{s=1}^{T}\alpha_{t,s}=1$ et $\alpha_{t,s} \ge 0$) :

$$
\alpha_{t,s} \;=\; \frac{\exp\!\bigl(e_{t,s}\bigr)}{\displaystyle\sum_{k=1}^{T} \exp\!\bigl(e_{t,k}\bigr)}
$$

**Vecteur de contexte** (barycentre des $h_s$ pondéré par les $\alpha_{t,s}$) :

$$
\mathbf{c}_t \;=\; \sum_{s=1}^{T} \alpha_{t,s}\, h_s
$$

Les $\alpha_{t,s}$ se visualisent en **heatmap** sur les $T=168$ heures.

---

### M5 : Métriques sur le jeu de test (après dénormalisation en MW)

Soit $n$ le nombre de points évalués, $y_i$ la valeur observée, $\hat{y}_i$ la prédiction (toutes deux en **MW**).

**MAE** : erreur absolue moyenne :

$$
\mathrm{MAE} \;=\; \frac{1}{n}\sum_{i=1}^{n} \bigl\lvert y_i - \hat{y}_i \bigr\rvert
$$

**RMSE** : racine de l’erreur quadratique moyenne :

$$
\mathrm{RMSE} \;=\; \sqrt{\,\frac{1}{n}\sum_{i=1}^{n} \bigl( y_i - \hat{y}_i \bigr)^{2}\,}
$$

**MAPE** : erreur relative moyenne en pourcentage :

$$
\mathrm{MAPE} \;=\; \frac{100}{n}\sum_{i=1}^{n} \left\lvert \frac{y_i - \hat{y}_i}{y_i} \right\rvert \;\;(\%)
$$

---

### M6 : Test de Diebold–Mariano (idée)

Pour chaque instant $i$, on définit une perte de prévision $L_i^{A}$ et $L_i^{B}$ (ex. $L_i = (y_i - \hat{y}_i)^2$ ou $L_i = \lvert y_i - \hat{y}_i\rvert$). La **différence de performance** instantanée est :

$$
d_i \;=\; L_i^{A} - L_i^{B}
$$

Le test étudie si la moyenne des $d_i$ est **significativement** différente de $0$ (sous des hypothèses de stationnarité faible des $d_i$). L’implémentation (statistique, p-valeur) passe par une bibliothèque ou un code documenté ; il faut **fixer** la définition de $L_i$ dans le rapport.

---

<a id="table-des-matieres"></a>
## Table des matières

**Guide de lecture** : [Par où commencer](#guide-debut) · [Recette](#recette-projet) · [Glossaire](#glossaire-projet) · [Jargon détaillé](#jargon-detail) · [Mathématiques](#maths-projet)

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

Une copie utilisée par les scripts se trouve dans `energy_forecast/data/raw/PJME_hourly.csv`. Une copie Kaggle peut aussi rester sous `Hourly Energy Consumption/PJME_hourly.csv`.

### 2.2 Schéma attendu

| Colonne   | Type (logique) | Description                          |
|-----------|----------------|--------------------------------------|
| `Datetime` | datetime      | Horodatage à pas horaire             |
| `PJME_MW`  | float           | Consommation en mégawatts (MW)       |

Vérification rapide : les deux premières lignes doivent ressembler à `Datetime,PJME_MW` puis une date-heure et une valeur numérique.

### 2.3 Ordre de grandeur

- Période couverte : **2002 à 2018** (vérifier sur votre copie après chargement).  
- Fréquence : **1 point par heure**.  
- Volume : **environ 145 000 lignes** (le fichier du dépôt contient ~145k enregistrements).  
- Plage de valeurs typique : l’ordre de grandeur se situe entre ~19 000 MW et ~57 000 MW selon saison et heure ; les statistiques exactes sont à recalculer dans le notebook d’EDA.

### 2.4 Qualité des données

Avant tout modèle : contrôler doublons d’index, trous dans la série, valeurs manquantes et incohérences (ex. heures manquantes après resampling). Documenter dans le notebook toute correction appliquée (interpolation, suppression d’intervalles, etc.) pour que l’équipe reproduise le même jeu.

---

## 3. Structure du projet (implémentée)

Arborescence actuelle sous `energy_forecast/` (les gros fichiers `.pkl` sont en général **ignorés par Git** ; les régénérer localement avec `preprocessing.py --all`) :

```
energy_forecast/
├── config.yaml
├── data/
│   ├── raw/
│   │   └── PJME_hourly.csv
│   └── processed/
│       ├── PJME_hourly_clean.csv
│       ├── PJME_hourly_with_features.csv
│       ├── train.pkl, val.pkl, test.pkl
│       ├── sequence_scaler.joblib
│       └── sequence_meta.json
├── notebooks/
│   └── 01_eda_pjm_hourly_load.ipynb
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── build_splits_sequences.py
│   ├── models.py
│   ├── train.py
│   └── evaluate.py
└── results/
    ├── checkpoints/          # best_lstm.pth, best_bilstm.pth, best_lstm_attention.pth, history_*.json
    ├── plots/                # eval_best_* (scatter, résidus, fenêtre 24 h)
    └── metrics.csv
```

`requirements.txt` est à la **racine** du dépôt Git (parent de `energy_forecast/`).

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

Pour la série de consommation (et éventuellement d’autres features continues si l’équipe décide de les mettre à la même échelle), même formule que **[M1](#maths-projet)** :

$$
x_{\mathrm{norm}} \;=\; \frac{x - x_{\min}}{x_{\max} - x_{\min}}
$$

Les valeurs $x_{\min}$ et $x_{\max}$ sont calculées **sur le sous-ensemble train uniquement**, puis appliquées au validation et au test. Sauvegarder le scaler (pickle ou attributs dans un fichier) pour dénormaliser les prédictions avant les métriques en MW.

---

## 6. Fenêtrage, cible et split temporel

### 6.1 Fenêtre et horizon

- **Longueur d’entrée :** `window_size = 168` (une semaine d’heures).  
- **Horizon de prédiction :** `horizon = 24` (les 24 prochaines heures de consommation).

Chaque échantillon d’entraînement est donc :

- **Entrée :** une matrice de forme `(168, n_features)`, en général avec les features listées plus haut et la série MW normalisée dans la fenêtre, selon la convention retenue (souvent la colonne MW et les features calendaires alignées pas à pas).  
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

### 7.1 Modèle 1 : LSTM (référence)

- Empilement : **2 couches** LSTM.  
- Taille de l’état caché : **128**.  
- **Dropout 0.2** entre couches (selon support PyTorch, appliquer de manière documentée).  
- Tête de régression : couches fully-connected (ex. `128 → 64` avec activation ReLU, puis `64 → 24`).  
- Agrégation temporelle : typiquement **dernier état caché** de la séquence (à préciser dans le code et figer pour toute l’équipe).

### 7.2 Modèle 2 : Bi-LSTM

- Même profondeur (2 couches) en **bidirectionnel**.  
- La dimension concaténée avant la tête fully-connected est **256** (128 avant + 128 arrière) si concaténation standard.  
- Même tête de sortie vers **24** valeurs.  
- Attention au fait que la bidirectionnalité utilise des informations « futures » dans la fenêtre d’entrête : c’est acceptable **tant que la fenêtre ne contient que le passé relatif à l’instant de prédiction** ; ce point doit être clair dans le rapport (sinon risque de fuite conceptuelle si mal défini).

### 7.3 Modèle 3 : LSTM + attention de Bahdanau

- LSTM empilé produit une séquence de sorties cachées \(h_1, \ldots, h_{168}\).  
- Couche d’attention implémentée **explicitement** (pas seulement un module « black box » sans exposition des poids), avec sortie :

  - un **vecteur de contexte** utilisé pour la régression ;  
  - les **poids d’attention** \(\alpha\) pour visualisation (heatmap sur les 168 pas).

Équations de référence (forme compacte ; développement complet et symboles : **[M4](#maths-projet)**) :

$$
e_{t,s} \;=\; \mathbf{v}^{\top}\,\tanh\!\bigl(\mathbf{W}_h\, h_s + \mathbf{W}_q\,\mathbf{q}_t\bigr)
$$

$$
\alpha_{t,s} \;=\; \frac{\exp\!\bigl(e_{t,s}\bigr)}{\displaystyle\sum_{k=1}^{T} \exp\!\bigl(e_{t,k}\bigr)}
$$

$$
\mathbf{c}_t \;=\; \sum_{s=1}^{T} \alpha_{t,s}\, h_s
$$

Le vecteur requête $\mathbf{q}_t$ est à définir dans l’implémentation (souvent le dernier état caché $h_T$ ou une projection de celui-ci) ; **une fois choisi, ne pas changer entre runs comparatifs** sans le noter.

### 7.4 Rappel LSTM (lien avec les mathématiques)

Les équations des portes (oubli, entrée, candidat, cellule, sortie) et l’attention Bahdanau sont regroupées dans **[Mathématiques utilisées dans ce projet](#maths-projet)** (repères **M3** et **M4**). Le code utilisera `nn.LSTM` ; la compréhension des portes sert surtout au rapport et à l’oral.

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

Même définitions que **[M5](#maths-projet)** (après dénormalisation en MW) :

$$
\mathrm{MAE} \;=\; \frac{1}{n}\sum_{i=1}^{n} \bigl\lvert y_i - \hat{y}_i \bigr\rvert
$$

$$
\mathrm{RMSE} \;=\; \sqrt{\,\frac{1}{n}\sum_{i=1}^{n} \bigl( y_i - \hat{y}_i \bigr)^{2}\,}
$$

$$
\mathrm{MAPE} \;=\; \frac{100}{n}\sum_{i=1}^{n} \left\lvert \frac{y_i - \hat{y}_i}{y_i} \right\rvert \quad (\%)
$$

Attention aux $y_i$ très proches de $0$ dans d’autres jeux de données ; ici les MW restent loin de zéro.

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
| **A : Données** | Téléchargement / copie vers `data/raw`, EDA, preprocessing, fenêtres, splits, artefacts `.pkl` | `notebooks/01_EDA.ipynb`, `src/preprocessing.py`, `src/feature_engineering.py` |
| **B : Modèles** | Trois classes PyTorch + test de formes sur batch fictif | `src/models.py` |
| **C : Entraînement** | Config, boucles, early stopping, checkpoints, logs | `config.yaml`, `src/train.py` |
| **D : Évaluation** | Métriques, figures, CSV, rapport HTML, Diebold–Mariano | `src/evaluate.py`, `results/*` |

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

### 12.2 Installation

Voir [Exécution et avancement](#execution-avancement) (venv + `pip install -r requirements.txt` à la racine du dépôt).

### 12.3 Dépendances

Liste complète dans `requirements.txt` à la racine (PyTorch, pandas, scikit-learn, joblib, holidays, pyyaml, matplotlib, jupyter, tqdm, etc.).

### 12.4 Commandes (implémentées)

Depuis la **racine du dépôt** :

```bash
python energy_forecast/src/preprocessing.py --all
python energy_forecast/src/train.py --config energy_forecast/config.yaml
python energy_forecast/src/evaluate.py --config energy_forecast/config.yaml
```

Le modèle actif est `model.name` dans `energy_forecast/config.yaml` (`lstm` | `bilstm` | `lstm_attention`). Options utiles : `--max-epochs`, `--subset` (debug rapide sur train/val). Evaluate : `--checkpoint` pour forcer un `.pth` précis.

---

## 13. Pièges courants à éviter

1. **Scaler ajusté sur tout le dataset** : invalide la comparaison et gonfle artificiellement les performances.  
2. **Shuffle global** des fenêtres sans garde-fou : mélange passé/futur ; si shuffle, le faire **à l’intérieur** du train uniquement après split.  
3. **Fuite via la validation** : tuner des hyperparamètres sur le test ; réserver le test à la toute fin.  
4. **MAPE** : vérifier les divisions par des valeurs très petites si autre jeu.  
5. **Bi-LSTM** : clarifier dans le rapport ce que représente la bidirectionnalité dans la fenêtre.  
6. **Reproductibilité** : sans graines fixes et sans version des librairies (`pip freeze` ou équivalent), les comparaisons inter-membres se dégradent.

---

## 14. Références

1. Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation.  
2. Bahdanau, D., Cho, K., & Bengio, Y. (2015). *Neural Machine Translation by Jointly Learning to Align and Translate.* ICLR.  
3. Schuster, M., & Paliwal, K. K. (1997). *Bidirectional Recurrent Neural Networks.* IEEE Transactions on Signal Processing.  
4. Mulla, R. et communauté Kaggle (2018). *PJM Hourly Energy Consumption.* Jeu de données Kaggle : https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption  

---

## Annexe : Ordre de lecture pour un nouveau membre

1. **[Recette](#recette-projet)** : vision d’ensemble en une table (étapes 0 à 13).  
2. **[Glossaire](#glossaire-projet)** : définitions en une ligne (mémo).  
3. **[Jargon détaillé](#jargon-detail)** : LSTM, Bi-LSTM, attention, PyTorch, loss, batch, fichiers, etc.  
4. **[Mathématiques](#maths-projet)** : formules alignées avec le rapport et les slides.  
5. Sections **1**, **2**, **4** et **6** : cadrage, données, pipeline, fenêtres et split.  
6. Section **11** : rôle de chaque membre (modules A–D).  
7. Sections **7** et **8** si modèles ou entraînement ; **9** et **10** si évaluation et rapport HTML.  
8. Section **13** à garder ouverte pendant le codage (fuites d’information, shuffle, etc.).

### Ce que ce README couvre (complétude)

| Zone du projet | Couvert ? | Où ? |
|----------------|-----------|------|
| Jargon (LSTM, métriques, pipeline) | Oui | [Glossaire](#glossaire-projet) + [Jargon détaillé](#jargon-detail) |
| Données, fichier, qualité | Oui | §2, recette étapes 2–3 |
| Features, normalisation, fenêtres, split | Oui | §5–6, maths M1 |
| Trois architectures PyTorch | Oui | §7, maths M3–M4 |
| Entraînement (Adam, scheduler, early stopping) | Oui | §8, maths M2 |
| Métriques, figures, attention, DM, HTML | Oui | §9–10, maths M5–M6 |
| Rôles équipe et jalons | Oui | §11 |
| **Code source exécutable** | Oui | `energy_forecast/src/`, [Exécution](#execution-avancement) |
| **Trois modèles entraînés + métriques test** | Oui | `results/metrics.csv`, checkpoints `best_*.pth` |
| **Rapport HTML / Diebold–Mariano** | Partiel / à faire | §9–10, recette étapes 12–13 |

Toute modification de convention (nom des colonnes, loss, agrégation LSTM) doit être **tracée ici ou dans `energy_forecast/config.yaml`** pour rester le contrat d’équipe unique.
