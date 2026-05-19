# test.py — DQN sur Acrobot-v1

Implémentation d'un agent Deep Q-Network (DQN) entraîné sur l'environnement `Acrobot-v1` de Gymnasium, puis rejoué en direct dans une fenêtre pygame.

---

## Vue d'ensemble

L'agent apprend à faire basculer un double pendule sous-actionné pour que son extrémité dépasse une ligne horizontale, en choisissant parmi 3 couples appliqués au joint central. Il utilise deux réseaux de neurones (policy et target) et un replay buffer pour stabiliser l'apprentissage.

Observation : 6 floats — `[cos(θ1), sin(θ1), cos(θ2), sin(θ2), θ1_dot, θ2_dot]`.
Action : 3 discrètes — couple {-1, 0, +1} sur le joint actif.
Récompense : -1 par step tant que l'objectif n'est pas atteint (max 500 steps). Un bon agent finit vite, score proche de -80 / -100.

---

## Dépendances

| Bibliothèque | Rôle |
|---|---|
| `gymnasium` | Environnement Acrobot-v1 |
| `torch` | Réseaux de neurones et optimisation |
| `matplotlib` | Visualisation des récompenses d'épisode |
| `pygame` | Rendu graphique en direct (via `render_mode="human"`) |
| `collections.deque` | Replay buffer circulaire |

---

## Hyperparamètres

| Nom | Valeur | Description |
|---|---|---|
| `BATCH_SIZE` | 128 | Nombre de transitions échantillonnées par mise à jour |
| `GAMMA` | 0.99 | Facteur d'actualisation des récompenses futures |
| `EPS_START` | 0.9 | Valeur initiale d'epsilon (exploration) |
| `EPS_END` | 0.05 | Valeur finale d'epsilon (exploitation) |
| `EPS_DECAY` | 5000 | Vitesse de décroissance exponentielle d'epsilon |
| `TAU` | 0.005 | Taux de mise à jour douce du réseau cible |
| `LR` | 5e-4 | Taux d'apprentissage de l'optimiseur AdamW |
| `memory capacity` | 50 000 | Taille du replay buffer |
| `num_episodes` | 600 (GPU/MPS) / 300 (CPU) | Nombre total d'épisodes d'entraînement |

---

## Structures de données

### `Transition` (namedtuple)

```
Transition(state, action, next_state, reward)
```

### `ReplayMemory`

Buffer circulaire qui stocke les transitions passées pour briser les corrélations temporelles lors de l'entraînement.

| Méthode | Description |
|---|---|
| `__init__(capacity)` | Initialise le buffer avec une capacité maximale |
| `push(*args)` | Ajoute une transition (écrase la plus ancienne si plein) |
| `sample(batch_size)` | Retourne un échantillon aléatoire de transitions |
| `__len__()` | Retourne le nombre de transitions stockées |

---

## Réseau de neurones — `DQN`

Réseau fully-connected à 3 couches avec activations ReLU.

```
Entrée (6) → Linear(128) → ReLU → Linear(128) → ReLU → Linear(3)
```

Deux instances sont créées :
- **`policy_net`** — réseau principal, mis à jour à chaque étape d'optimisation
- **`target_net`** — réseau cible, mis à jour doucement via TAU pour stabiliser l'entraînement

---

## Fonctions

### `select_action(state)`

Sélectionne une action selon une politique epsilon-greedy décroissante.

```
eps = EPS_END + (EPS_START - EPS_END) * exp(-steps_done / EPS_DECAY)
```

### `plot_rewards(show_result=False)`

Affiche la récompense totale de chaque épisode en temps réel, avec moyenne glissante sur 100 épisodes dès qu'elle est calculable et ligne de seuil à -100.

### `optimize_model()`

Étape de gradient DQN classique : échantillonne un batch, calcule `Q(s,a)` via `policy_net`, la cible `r + γ·max_a' Q_target(s',a')`, perte Huber (`SmoothL1Loss`), gradient clipping à ±100, mise à jour AdamW.

---

## Boucle d'entraînement

```
Pour chaque épisode :
    Réinitialise l'environnement
    Pour chaque pas t :
        Sélectionne une action (epsilon-greedy)
        Exécute l'action → obtient observation, reward, done
        Stocke la transition dans le replay buffer
        Appelle optimize_model()
        Met à jour target_net (soft update : θ' ← τθ + (1-τ)θ')
        Si done : enregistre la récompense, trace le graphe, passe à l'épisode suivant
```

---

## Phase d'évaluation

Une fois l'entraînement terminé, le script exécute une vraie suite d'évaluation — ne pas confondre avec la courbe d'entraînement, qui est biaisée par l'exploration ε-greedy en cours et par les updates de gradient à chaque step.

### Protocole

| Étape | Politique | ε | Apprentissage | N épisodes |
|---|---|---|---|---|
| Évaluation agent | Greedy sur `policy_net` | 0 | désactivé | 50 |
| Baseline | `action_space.sample()` | — | — | 50 |

Même environnement, mêmes seeds disponibles (non fixées), aucune mise à jour du réseau : on mesure uniquement la qualité de la politique apprise.

### Métriques calculées

Pour chaque politique :
- **mean / std / min / max / median** du reward total
- **steps_moy** — longueur moyenne des épisodes (plus c'est bas, mieux c'est pour Acrobot)
- **success rate** — % d'épisodes avec reward ≥ -100 (seuil "résolu")

Enfin on affiche le gain relatif de l'agent entraîné par rapport au random — ça répond à la question *"qu'est-ce que l'apprentissage a vraiment apporté ?"*.

### Graphes analytiques (figure en 4 panneaux)

| Panneau | Ce qu'il montre | Ce qu'on cherche |
|---|---|---|
| **(a) Histogramme rewards** | Distribution des 50 rewards, trained (vert) vs random (rouge), superposés | Décalage net des deux distributions — l'agent doit être à droite |
| **(b) Boxplot** | Médiane, quartiles, outliers, moyenne (triangle) | Resserrement de la dispersion chez trained — politique stable |
| **(c) Distribution des actions** | % de temps passé sur chaque action (-1/0/+1), trained vs random | Le random est uniforme (≈33% chacune). Le trained exploite la physique : il alterne couples extrêmes pour pomper de l'énergie, l'action "0" devrait être sous-représentée |
| **(d) Longueur des épisodes** | Histogramme du nombre de steps avant fin | Random plafonne à 500 (timeout). Trained doit clairement piquer entre 60 et 150 steps |

### Pourquoi c'est plus intéressant que juste "l'agent marche"

- La courbe d'entraînement ne dit **pas** si l'agent est bon — elle dit qu'il s'améliore, pendant qu'il apprend. L'éval greedy révèle la **vraie performance**.
- La baseline aléatoire donne une échelle absolue (ici -500 plancher) : savoir que trained atteint -85 est utile, mais `-85 vs -500` est percutant.
- La distribution des actions révèle la **stratégie** apprise — pas juste le score.
- Le boxplot révèle la **stabilité** : un agent qui fait 50% d'épisodes à -60 et 50% à -300 a la même moyenne qu'un agent qui fait tout à -180, mais c'est un agent pire.

## Phase de démo pygame

Après les graphes analytiques, le script ouvre une fenêtre pygame (`render_mode="human"`) pour :
- **3 épisodes avec l'agent entraîné** — tu vois la stratégie de pompage d'énergie : l'agent applique des couples alternés jusqu'à faire passer l'extrémité au-dessus de la ligne
- **1 épisode avec la policy aléatoire** — pour le contraste visuel : le pendule s'agite sans direction et ne monte jamais

Chaque score est affiché en console.

---

## Commande

```bash
.venv/bin/python test_acrobot/test.py
```

Pendant le run :
- une fenêtre matplotlib se met à jour en temps réel (courbe d'apprentissage)
- `Complete` s'affiche en console quand l'entraînement est terminé
- une fenêtre pygame s'ouvre ensuite pour 5 démos, puis le script affiche le graphe final
