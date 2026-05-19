# test.py — DQN sur MountainCar-v0

Implémentation d'un agent Deep Q-Network (DQN) entraîné sur l'environnement `MountainCar-v0` de Gymnasium, puis rejoué en direct dans une fenêtre pygame.

---

## Vue d'ensemble

L'agent apprend à faire monter une voiture sous-puissante au sommet d'une colline. Le moteur seul est trop faible pour gravir la pente directement : la stratégie gagnante consiste à osciller en arrière puis en avant pour accumuler de l'énergie cinétique. L'agent utilise deux réseaux de neurones (policy et target) et un replay buffer pour stabiliser l'apprentissage.

Observation : 2 floats — `[position, vélocité]` avec `position ∈ [-1.2, 0.6]` et `vélocité ∈ [-0.07, 0.07]`.
Action : 3 discrètes — `0` pousser à gauche, `1` ne rien faire, `2` pousser à droite.
Récompense : -1 par step tant que la voiture n'atteint pas le drapeau (position ≥ 0.5). Épisode tronqué à 200 steps. Un bon agent finit autour de -100 / -110.

---

## Dépendances

| Bibliothèque | Rôle |
|---|---|
| `gymnasium` | Environnement MountainCar-v0 |
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
| `EPS_DECAY` | 10000 | Vitesse de décroissance exponentielle d'epsilon |
| `TAU` | 0.005 | Taux de mise à jour douce du réseau cible |
| `LR` | 5e-4 | Taux d'apprentissage de l'optimiseur AdamW |
| `memory capacity` | 50 000 | Taille du replay buffer |
| `num_episodes` | 1000 (GPU/MPS) / 700 (CPU) | Nombre total d'épisodes d'entraînement |

> **Pourquoi `EPS_DECAY` plus grand qu'Acrobot ?** MountainCar-v0 a une récompense extrêmement éparse — l'agent ne reçoit jamais de signal positif tant qu'il n'a pas atteint le drapeau. Sans une exploration suffisamment longue, il peut ne jamais découvrir la stratégie de pompage et stagner à -200 indéfiniment.

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
Entrée (2) → Linear(128) → ReLU → Linear(128) → ReLU → Linear(3)
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

Affiche la récompense totale de chaque épisode en temps réel, avec moyenne glissante sur 100 épisodes dès qu'elle est calculable et ligne de seuil à -110.

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
- **steps_moy** — longueur moyenne des épisodes (plus c'est bas, mieux c'est pour MountainCar)
- **success rate** — % d'épisodes avec reward ≥ -110 (seuil "résolu")

Enfin on affiche le gain relatif de l'agent entraîné par rapport au random — ça répond à la question *"qu'est-ce que l'apprentissage a vraiment apporté ?"*.

### Graphes analytiques (figure en 4 panneaux)

| Panneau | Ce qu'il montre | Ce qu'on cherche |
|---|---|---|
| **(a) Histogramme rewards** | Distribution des 50 rewards, trained (vert) vs random (rouge), superposés | Décalage net des deux distributions — l'agent doit être à droite (proche de -100) tandis que le random plafonne à -200 |
| **(b) Boxplot** | Médiane, quartiles, outliers, moyenne (triangle) | Resserrement de la dispersion chez trained — politique stable |
| **(c) Distribution des actions** | % de temps passé sur chaque action (0/1/2), trained vs random | Le random est uniforme (≈33% chacune). Le trained doit alterner *gauche* et *droite* (action "1" sous-représentée) pour pomper de l'énergie |
| **(d) Longueur des épisodes** | Histogramme du nombre de steps avant fin | Random reste collé à 200 (timeout, jamais le drapeau). Trained doit piquer entre 90 et 150 steps |

### Pourquoi c'est plus intéressant que juste "l'agent marche"

- La courbe d'entraînement ne dit **pas** si l'agent est bon — elle dit qu'il s'améliore, pendant qu'il apprend. L'éval greedy révèle la **vraie performance**.
- La baseline aléatoire donne une échelle absolue (ici -200 plancher, atteint quasi-systématiquement) : savoir que trained atteint -110 est utile, mais `-110 vs -200` est percutant.
- La distribution des actions révèle la **stratégie** apprise : sans pompage gauche/droite, pas de drapeau atteignable.
- Le boxplot révèle la **stabilité** : un agent qui réussit 50% des fois à -90 et échoue 50% à -200 a la même moyenne qu'un agent constant à -145, mais c'est un agent pire.

### Note sur la difficulté de MountainCar-v0

MountainCar est notoirement difficile pour un DQN vanilla parce que la récompense est uniformément -1 jusqu'à l'arrivée — aucun signal de progrès. Si l'exploration aléatoire n'atteint jamais le drapeau pendant les premiers épisodes, le réseau n'apprend rien d'utile. C'est pour ça que `EPS_DECAY` est augmenté à 10000 (vs 5000 sur Acrobot). Si l'agent reste bloqué à -200, des techniques classiques aident : reward shaping basé sur la position/vitesse, ou augmenter `num_episodes`.

## Phase de démo pygame

Après les graphes analytiques, le script ouvre une fenêtre pygame (`render_mode="human"`) pour :
- **3 épisodes avec l'agent entraîné** — tu vois la stratégie de pompage : la voiture recule d'abord pour prendre de l'élan, puis fonce à droite, et répète si nécessaire jusqu'à franchir le drapeau
- **1 épisode avec la policy aléatoire** — pour le contraste visuel : la voiture s'agite près du fond de la vallée et n'arrive jamais en haut

Chaque score est affiché en console.

---

## Commande

```bash
.venv/bin/python test_mountainCar/test.py
```

Pendant le run :
- une fenêtre matplotlib se met à jour en temps réel (courbe d'apprentissage)
- `Complete` s'affiche en console quand l'entraînement est terminé
- une fenêtre pygame s'ouvre ensuite pour 4 démos, puis le script affiche le graphe final
