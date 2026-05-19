# test.py — DQN sur CartPole-v1

Implémentation d'un agent Deep Q-Network (DQN) entraîné sur l'environnement `CartPole-v1` de Gymnasium.

---

## Vue d'ensemble

L'agent apprend à équilibrer un bâton sur un chariot en choisissant des actions discrètes (gauche / droite). Il utilise deux réseaux de neurones (policy et target) et un replay buffer pour stabiliser l'apprentissage.

---

## Dépendances

| Bibliothèque | Rôle |
|---|---|
| `gymnasium` | Environnement CartPole-v1 |
| `torch` | Réseaux de neurones et optimisation |
| `matplotlib` | Visualisation des durées d'épisode |
| `collections.deque` | Replay buffer circulaire |

---

## Hyperparamètres

| Nom | Valeur | Description |
|---|---|---|
| `BATCH_SIZE` | 128 | Nombre de transitions échantillonnées par mise à jour |
| `GAMMA` | 0.99 | Facteur d'actualisation des récompenses futures |
| `EPS_START` | 0.9 | Valeur initiale d'epsilon (exploration) |
| `EPS_END` | 0.01 | Valeur finale d'epsilon (exploitation) |
| `EPS_DECAY` | 2500 | Vitesse de décroissance exponentielle d'epsilon |
| `TAU` | 0.005 | Taux de mise à jour douce du réseau cible |
| `LR` | 3e-4 | Taux d'apprentissage de l'optimiseur AdamW |
| `num_episodes` | 600 (GPU/MPS) / 50 (CPU) | Nombre total d'épisodes d'entraînement |

---

## Structures de données

### `Transition` (namedtuple)

Représente une transition dans l'environnement :

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
Entrée (n_observations) → Linear(128) → ReLU → Linear(128) → ReLU → Linear(n_actions)
```

Pour CartPole-v1 : `n_observations = 4`, `n_actions = 2`.

| Méthode | Description |
|---|---|
| `__init__(n_observations, n_actions)` | Construit les 3 couches linéaires |
| `forward(x)` | Passe avant — retourne les Q-values pour chaque action |

Deux instances sont créées :
- **`policy_net`** — réseau principal, mis à jour à chaque étape d'optimisation
- **`target_net`** — réseau cible, mis à jour doucement via TAU pour stabiliser l'entraînement

---

## Fonctions

### `select_action(state)`

Sélectionne une action selon une politique epsilon-greedy décroissante.

- Si `random() > eps_threshold` → action greedy via `policy_net` (exploitation)
- Sinon → action aléatoire (exploration)

L'epsilon décroît exponentiellement selon :

```
eps = EPS_END + (EPS_START - EPS_END) * exp(-steps_done / EPS_DECAY)
```

### `plot_durations(show_result=False)`

Affiche la durée de chaque épisode en temps réel. Si au moins 100 épisodes ont été joués, trace également la moyenne glissante sur 100 épisodes. Compatible avec les environnements Jupyter (`is_ipython`).

### `optimize_model()`

Effectue une étape de gradient sur le `policy_net` en utilisant l'algorithme DQN :

1. Échantillonne un batch depuis le replay buffer
2. Calcule `Q(s, a)` via `policy_net`
3. Calcule `V(s')` via `target_net` (0 pour les états terminaux)
4. Calcule les Q-values cibles : `Q_target = reward + GAMMA * V(s')`
5. Calcule la perte Huber (`SmoothL1Loss`) entre Q prédit et Q cible
6. Rétropropagation avec clipping du gradient à ±100
7. Mise à jour des poids via AdamW

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
        Si done : enregistre la durée, trace le graphe, passe à l'épisode suivant
```

### Mise à jour douce du réseau cible

À chaque pas, les poids du `target_net` sont interpolés vers ceux du `policy_net` :

```
θ'_key = TAU * θ_key + (1 - TAU) * θ'_key
```

---

## Sélection du device

Le device est choisi automatiquement dans cet ordre de priorité :

```
CUDA (GPU NVIDIA) > MPS (GPU Apple Silicon) > CPU
```

---

## Reproductibilité

Les seeds aléatoires ne sont pas fixés par défaut, ce qui permet à l'agent d'explorer différentes trajectoires d'entraînement. Pour fixer les seeds et rendre les runs reproductibles, initialiser `random`, `torch`, `env` et `torch.cuda` avec une seed commune avant l'entraînement.

---

## Sortie

- Affichage en temps réel de la durée de chaque épisode via matplotlib
- Affichage du graphe final avec la courbe de résultat
- Message `Complete` en console à la fin de l'entraînement
