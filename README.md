# VisuModel

Un réseau de neurones écrit **de zéro** en Python pur, capable de repérer des informations (formes, motifs) dans des données bruitées. Aucune dépendance externe : ni numpy, ni PyTorch.

## Contenu

| Fichier | Rôle |
|---|---|
| `neural_net.py` | Le réseau : perceptron multicouche, propagation avant, rétropropagation, SGD |
| `demo_detection.py` | Démo : repérer une croix, un carré ou un X dans une grille 5×5 bruitée |
| `test_neural_net.py` | Tests unitaires (softmax, sorties, décroissance de la perte, apprentissage du XOR) |

## Fonctionnement

- **Architecture** : couches denses, activation `tanh` dans les couches cachées, `softmax` en sortie.
- **Perte** : entropie croisée.
- **Apprentissage** : descente de gradient stochastique, rétropropagation écrite à la main.
- **Initialisation** : poids tirés selon une loi normale de variance `2 / n_entrées`.

## Utilisation

Prérequis : Python 3.8 ou plus récent.

```bash
git clone https://github.com/matzer667/VisuModel.git
cd VisuModel

python demo_detection.py      # entraîne le réseau et affiche la précision
python -m unittest test_neural_net -v   # lance les tests
```

Exemple d'usage dans votre propre code :

```python
from neural_net import MLP

# 25 entrées (grille 5x5 aplatie), une couche cachée de 16 neurones, 3 classes
net = MLP([25, 16, 3], seed=0)
net.fit(train_data, epochs=15, lr=0.05)   # train_data = [(vecteur, classe), ...]

classe = net.predict(vecteur)
probas = net.predict_proba(vecteur)
```

## Résultats de la démo

Avec 300 exemples par classe et 8 % de pixels inversés :

- précision avant entraînement : environ 56 % (hasard)
- précision après 15 époques sur le jeu de test : 100 %

## Limites

- Réseau dense uniquement : pas de convolution, donc il convient aux petites entrées (quelques centaines de valeurs), pas aux grandes images.
- Pur Python : lent sur de gros volumes. Pour de grands jeux de données, un passage à numpy est le prochain pas logique.

## Licence

À définir.
