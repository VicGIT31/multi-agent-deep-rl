# test.py — SAC sur Ant-v5

Implémentation d'un agent Soft Actor-Critic (SAC) entraîné sur l'environnement `Ant-v5` de Gymnasium (MuJoCo).

---

## Vue d'ensemble

L'agent apprend à faire avancer une fourmi quadrupède en contrôlant 8 couples articulaires continus. Il utilise un réseau d'acteur stochastique (politique gaussienne squashée par tanh), deux critiques (twin Q-networks), deux réseaux cibles, et un replay buffer pour stabiliser l'apprentissage.

SAC est l'équivalent canonique de DQN pour les espaces d'actions continus : on remplace le `argmax` discret par une politique stochastique entraînée à maximiser à la fois la récompense et l'entropie de la politique.

---

## Dépendances

| Bibliothèque | Rôle |
|---|---|
| `gymnasium[mujoco]` | Environnement Ant-v5 |
| `torch` | Réseaux de neurones et optimisation |
| `matplotlib` | Visualisation des récompenses d'épisode |
| `collections.deque` | Replay buffer circulaire |

---

## Hyperparamètres

| Nom | Valeur | Description |
|---|---|---|
| `BATCH_SIZE` | 256 | Nombre de transitions échantillonnées par mise à jour |
| `GAMMA` | 0.99 | Facteur d'actualisation des récompenses futures |
| `TAU` | 0.005 | Taux de mise à jour douce des réseaux cibles |
| `LR` | 3e-4 | Taux d'apprentissage des optimiseurs AdamW |
| `ALPHA` | 0.2 | Coefficient d'entropie (trade-off exploration/exploitation) |
| `START_STEPS` | 1 000 | Steps d'actions aléatoires avant d'utiliser l'acteur |
| `MEMORY_CAPACITY` | 50 000 | Taille du replay buffer (numpy préalloué) |
| `NUM_EPISODES` | 400 | Nombre total d'épisodes d'entraînement |
| `DEMO_EPISODES` | 10 | Nombre d'épisodes pygame après l'entraînement |

---

## Structures de données

### `Transition` (namedtuple)

Représente une transition dans l'environnement :

```
Transition(state, action, next_state, reward)
```

### `ReplayMemory`

Buffer circulaire **numpy préalloué** (5 arrays `float32` : `states`, `actions`, `next_states`, `rewards`, `dones`). Bien plus rapide et léger qu'une `deque` de tensors PyTorch, ce qui évite l'OOM killer sur Ant-v5.

| Méthode | Description |
|---|---|
| `__init__(capacity, n_observations, n_actions)` | Préalloue les arrays numpy |
| `push(s, a, ns, r, d)` | Écrit la transition à l'index courant (circulaire) |
| `sample(batch_size)` | Tire un batch aléatoire et le convertit en tensors sur le bon device |
| `__len__()` | Retourne le nombre de transitions stockées |

---

## Réseaux de neurones

### `Actor` — politique stochastique

Réseau fully-connected qui produit une distribution gaussienne sur les actions, squashée par tanh pour respecter les bornes de l'action space.

```
Entrée (n_observations) → Linear(256) → ReLU → Linear(256) → ReLU → (mean, log_std)
```

| Méthode | Description |
|---|---|
| `forward(x)` | Retourne `(mean, log_std)` clampé sur `[-20, 2]` |
| `sample(x)` | Échantillonne une action via reparameterization trick, applique tanh, retourne `(action, log_prob)` corrigé du changement de variable |

### `Critic` — Q-network

Estime `Q(s, a)`. Deux instances indépendantes (twin Q-learning) pour réduire le biais de surestimation.

```
Entrée (n_observations + n_actions) → Linear(256) → ReLU → Linear(256) → ReLU → Linear(1)
```

Quatre instances sont créées :
- **`critic_1`, `critic_2`** — réseaux principaux, mis à jour à chaque étape
- **`target_1`, `target_2`** — réseaux cibles, mis à jour doucement via TAU

Pour Ant-v5 : `n_observations = 105`, `n_actions = 8` (Box continu `[-1, 1]^8`).

---

## Fonctions

### `select_action(state)`

- Pendant les `START_STEPS` premiers steps → action uniformément aléatoire dans `action_space` (warmup pour remplir le buffer).
- Ensuite → action échantillonnée par l'acteur stochastique (`actor.sample`). L'exploration vient de la stochasticité gaussienne plutôt que d'un epsilon-greedy.

### `plot_rewards(show_result=False)`

Graphique amélioré, mis à jour tous les 5 épisodes :
- courbe brute en bleu clair (reward par épisode)
- moyenne glissante sur 50 épisodes en rouge
- bande ± écart-type autour de la moyenne (zone d'incertitude)
- ligne verte horizontale au meilleur reward atteint
- titre avec compteur d'épisode, grille, légende
- sauvegarde finale dans `test_ant/training_curve.png`

### `optimize_model()`

Effectue une étape de gradient SAC :

1. Échantillonne un batch depuis le replay buffer.
2. **Cible Q** : `target = r + γ · (min(target_1, target_2)(s', a') - α · log π(a'|s'))` avec `a' ~ π(·|s')`.
3. Met à jour `critic_1` et `critic_2` par perte Huber (`SmoothL1Loss`) entre `Q(s,a)` et `target`, avec gradient clipping à ±100.
4. **Loss acteur** : `(α · log π(a|s) - min(Q1, Q2)(s, a)).mean()` avec `a ~ π(·|s)` via reparameterization trick.
5. Mise à jour des poids via AdamW.

---

## Boucle d'entraînement

```
Pour chaque épisode :
    Réinitialise l'environnement
    Pour chaque pas t :
        Sélectionne une action (random pendant warmup, sinon actor.sample)
        Exécute l'action → obtient observation, reward, done
        Stocke la transition dans le replay buffer
        Appelle optimize_model()
        Met à jour target_1 et target_2 (soft update : θ' ← τθ + (1-τ)θ')
        Si done : enregistre la récompense, trace le graphe, passe à l'épisode suivant
```

### Mise à jour douce des réseaux cibles

À chaque update, les poids des `target_*` sont interpolés in-place vers ceux des `critic_*` (vectorisé via `param.data.mul_/add_`, beaucoup plus rapide que de manipuler `state_dict()`) :

```
θ'_key ← (1 - TAU) * θ'_key + TAU * θ_key
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

## Phase de démo pygame

Après l'entraînement, le script ouvre une fenêtre pygame (`render_mode="human"`) pour rejouer **10 épisodes avec l'agent entraîné** en politique déterministe (on prend `tanh(mean)` au lieu d'échantillonner — pas d'exploration au moment de la démo).

Chaque score est affiché en console, suivi d'un récapitulatif (mean / std / min / max).

---

## Sortie

- Affichage en temps réel de la récompense totale de chaque épisode via matplotlib
- Affichage du graphe final avec la courbe de résultat
- Message `Complete` en console à la fin de l'entraînement
- Fenêtre pygame avec 3 démos de l'agent entraîné
