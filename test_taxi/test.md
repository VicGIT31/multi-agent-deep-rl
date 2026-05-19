# test.py — DQN sur Taxi-v3

Implémentation d'un agent Deep Q-Network (DQN) entraîné sur l'environnement `Taxi-v3` de Gymnasium, puis comparé à une baseline aléatoire et rejoué en mode rendu.

---

## Pourquoi ce modèle

Après `CartPole-v1`, `MountainCar-v0`, `Acrobot-v1` et `Ant-v5`, `Taxi-v3` est le bonus le plus simple à ajouter proprement :
- environnement déjà inclus dans Gymnasium, sans MuJoCo ni Box2D ;
- espace d'actions discret, donc compatible avec le DQN déjà utilisé ;
- récompense dense, donc plus facile à apprendre que `FrozenLake-v1` ;
- exécution rapide sur CPU.

L'environnement représente une grille où le taxi doit récupérer un passager puis le déposer à la bonne destination.

Observation : 1 état discret parmi 500 combinaisons possibles — position du taxi, position du passager, destination.
Action : 6 actions discrètes — `0` sud, `1` nord, `2` est, `3` ouest, `4` pickup, `5` dropoff.
Récompense : `-1` par step, `-10` pour pickup/dropoff illégal, `+20` pour une livraison réussie.

---

## Dépendances

| Bibliothèque | Rôle |
|---|---|
| `gymnasium` | Environnement Taxi-v3 |
| `torch` | Réseaux de neurones et optimisation |
| `numpy` | Encodage one-hot et métriques |
| `matplotlib` | Visualisation des récompenses et de l'évaluation |
| `collections.deque` | Replay buffer circulaire |

---

## Hyperparamètres

| Nom | Valeur | Description |
|---|---:|---|
| `BATCH_SIZE` | 128 | Nombre de transitions échantillonnées par mise à jour |
| `GAMMA` | 0.99 | Facteur d'actualisation des récompenses futures |
| `EPS_START` | 1.0 | Exploration initiale |
| `EPS_END` | 0.05 | Exploration minimale |
| `EPS_DECAY` | 25000 | Décroissance exponentielle d'epsilon |
| `TAU` | 0.005 | Mise à jour douce du réseau cible |
| `LR` | 1e-3 | Taux d'apprentissage AdamW |
| `MEMORY_CAPACITY` | 50 000 | Taille du replay buffer |
| `num_episodes` | 900 CPU / 1200 GPU-MPS | Nombre d'épisodes d'entraînement par défaut |

---

## Adaptation importante

`Taxi-v3` ne retourne pas un vecteur continu comme `MountainCar` ou `CartPole`, mais un identifiant d'état entier entre `0` et `499`.

Pour garder exactement la même logique DQN :

```text
state entier -> one-hot(500) -> DQN -> Q-values pour 6 actions
```

Le réseau apprend donc une approximation de table Q, mais avec la même mécanique que les autres tests : replay buffer, target network, epsilon-greedy, perte Huber et soft update.

---

## Réseau de neurones — `DQN`

Réseau fully-connected à 3 couches avec activations ReLU.

```text
Entrée (500) -> Linear(128) -> ReLU -> Linear(128) -> ReLU -> Linear(6)
```

Deux instances sont créées :
- **`policy_net`** — réseau principal, mis à jour par gradient ;
- **`target_net`** — réseau cible, mis à jour doucement via `TAU`.

---

## Boucle d'entraînement

```text
Pour chaque épisode :
    reset Taxi-v3
    convertir l'état discret en one-hot
    choisir une action epsilon-greedy
    exécuter l'action
    stocker transition dans le replay buffer
    optimiser le DQN
    soft-update target_net
    logger reward, longueur, réussite, actions illégales
```

La courbe d'entraînement est sauvegardée dans :

```bash
test_taxi/training_curve.png
```

---

## Phase d'évaluation

Après l'entraînement, le script évalue :
- l'agent entraîné en greedy pur (`epsilon = 0`) sur 100 épisodes ;
- une baseline aléatoire sur 100 épisodes.

Métriques affichées :
- moyenne, écart-type, minimum, maximum, médiane du reward ;
- longueur moyenne des épisodes ;
- taux de livraison réussie ;
- taux de score supérieur ou égal à `8` ;
- nombre moyen d'actions illégales.

Résultat obtenu sur le run de référence :

| Politique | Reward moyen | Livraison | Steps moyens | Actions illégales moyennes |
|---|---:|---:|---:|---:|
| `TRAINED` | `8.05` | `100.0%` | `12.9` | `0.0` |
| `RANDOM` | `-755.68` | `10.0%` | `194.1` | `62.6` |

Le graphe analytique est sauvegardé dans :

```bash
test_taxi/evaluation.png
```

---

## Graphes analytiques

La figure d'évaluation contient 4 panneaux :

| Panneau | Ce qu'il montre | Ce qu'on cherche |
|---|---|---|
| Histogramme rewards | Distribution trained vs random | L'agent doit être très à droite du random |
| Boxplot | Dispersion des scores | Politique stable, peu de variance |
| Distribution des actions | Fréquence des 6 actions | Le trained utilise pickup/dropoff au bon moment, pas uniformément |
| Actions illégales | Nombre d'erreurs pickup/dropoff | Le trained doit tomber proche de 0 |

---

## Phase de démo

Après les graphes, le script lance :
- 3 épisodes avec l'agent entraîné ;
- 1 épisode avec la policy aléatoire pour comparaison.

Pour éviter la démo lors d'un run headless :

```bash
SKIP_DEMO=1 ./venv/bin/python test_taxi/test.py
```

---

## Commande

```bash
./venv/bin/python test_taxi/test.py
```

Options utiles :

```bash
TAXI_EPISODES=300 SKIP_DEMO=1 ./venv/bin/python test_taxi/test.py
```

```bash
MPLBACKEND=Agg TAXI_EPISODES=900 SKIP_DEMO=1 ./venv/bin/python test_taxi/test.py
```
