# PPO — LunarLander-v3

Implémentation de l'algorithme **Proximal Policy Optimization (PPO)** appliqué à l'environnement `LunarLander-v3` de Gymnasium.

---

## Fichiers

| Fichier | Rôle |
|---|---|
| `test.py` | Entraînement de l'agent |
| `result.py` | Visualisation du modèle entraîné |
| `ppo_lunar_model_250.pth` | Poids du modèle sauvegardé à la victoire |
| `ppo_lunar_current.pth` | Checkpoint sauvegardé toutes les 10 épisodes |
| `env_stats.pkl` | Statistiques de normalisation des observations |

---

## Architecture — `ActorCritic`

Le réseau de neurones partage une couche commune entre l'acteur et le critique.

```
Observation (8) → [Linear 128 → ReLU → Linear 128 → ReLU]
                              ↓                    ↓
                         Actor head            Critic head
                    (distribution sur        (valeur d'état V(s))
                      4 actions)
```

- **Feature layer** : deux couches fully-connected de 128 neurones avec activation ReLU. Extrait une représentation partagée de l'état.
- **Actor** : couche linéaire → softmax. Produit une distribution de probabilités sur les 4 actions discrètes.
- **Critic** : couche linéaire → scalaire. Estime la valeur de l'état courant V(s).

---

## Entraînement — `test.py`

### 1. Initialisation

```python
env = gym.make("LunarLander-v3")
env = gym.wrappers.NormalizeObservation(env)
```

L'environnement est enveloppé avec `NormalizeObservation` qui normalise les observations en ligne (moyenne 0, variance 1) en utilisant des statistiques accumulées au fil des épisodes. Cela stabilise l'apprentissage.

### 2. Collecte des transitions

À chaque pas de temps, l'agent sélectionne une action via `select_action` :

1. L'état est converti en tenseur et passé dans le réseau.
2. Une distribution catégorielle est créée à partir des probabilités de l'acteur.
3. Une action est **échantillonnée** (exploration), et son log-probabilité et la valeur d'état sont mémorisées.

Les transitions `(state, action, logprob, reward, value, mask)` s'accumulent dans un buffer `memory`. Le masque (`mask`) vaut 0 si l'épisode est terminé, 1 sinon — il coupe le bootstrap des valeurs aux frontières d'épisodes.

### 3. Mise à jour PPO (tous les 2048 pas)

Quand le buffer atteint `update_timestep = 2048` transitions, une mise à jour est déclenchée.

**Calcul des retours via GAE (Generalized Advantage Estimation) :**

```
δt = rt + γ * V(st+1) * maskt - V(st)
GAEt = δt + γ * λ * maskt * GAEt+1
Retour = GAEt + V(st)
```

- `γ = 0.99` : facteur de discount (importance des récompenses futures)
- `λ = 0.95` : paramètre GAE (compromis biais/variance)

Les avantages sont ensuite normalisés (moyenne 0, écart-type 1) pour stabiliser les gradients.

**Optimisation sur `k_epochs = 10` passes :**

Pour chaque passe, on calcule le ratio entre la nouvelle politique et l'ancienne :

```
ratio = exp(log π_new(a|s) - log π_old(a|s))
```

La perte de l'acteur est la **perte PPO clippée** :

```
L_actor = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)    avec ε = 0.2
```

Le clipping empêche des mises à jour trop grandes qui déstabiliseraient la politique.

La perte totale combine les trois termes :

```
L = L_actor + 0.5 * MSE(V(s), retour) - 0.01 * entropy
```

- `0.5 * L_critic` : apprend la fonction de valeur
- `-0.01 * entropy` : bonus d'entropie pour encourager l'exploration

Un gradient clipping à `0.5` est appliqué avant chaque étape d'optimisation.

### 4. Critère d'arrêt

L'entraînement s'arrête quand la **moyenne glissante sur 100 épisodes** atteint **200 points**. Le modèle et les statistiques de normalisation sont alors sauvegardés.

---

## Visualisation — `result.py`

### 1. Chargement des statistiques

Le wrapper `NormalizeObservation` de l'entraînement a accumulé une moyenne et une variance des observations. Ces stats sont rechargées depuis `env_stats.pkl` pour reproduire exactement la même normalisation à l'inférence.

```python
state = (state - mean) / std
```

Sans cette étape, le modèle recevrait des observations hors distribution et produirait des actions aberrantes.

### 2. Chargement du modèle

```python
model.load_state_dict(torch.load("ppo_lunar_model_250.pth", ...))
model.eval()
```

`model.eval()` désactive le dropout et le batch normalization (non utilisés ici, mais bonne pratique).

### 3. Inférence déterministe

```python
action = torch.argmax(probs).item()
```

À l'inférence, on utilise **argmax** au lieu d'échantillonner. Le modèle étant entraîné et stable, on prend l'action la plus probable à chaque pas — comportement déterministe et optimal.

5 épisodes sont joués avec rendu visuel (`render_mode="human"`).

---

## Hyperparamètres

| Paramètre | Valeur | Rôle |
|---|---|---|
| `gamma` | 0.99 | Discount des récompenses futures |
| `lmbda` | 0.95 | Paramètre GAE (biais/variance) |
| `eps_clip` | 0.2 | Amplitude max du clipping PPO |
| `k_epochs` | 10 | Passes d'optimisation par buffer |
| `entropy_coef` | 0.01 | Poids du bonus d'entropie |
| `update_timestep` | 2048 | Taille du buffer avant mise à jour |
| `lr` | 3e-4 | Taux d'apprentissage Adam |
| `max_episodes` | 5000 | Limite d'épisodes d'entraînement |
