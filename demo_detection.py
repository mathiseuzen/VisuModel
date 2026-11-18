"""Démo : le réseau repère une forme (information) dans une grille 5x5 bruitée.

Classes : 0 = croix, 1 = carré, 2 = diagonales (X).
"""

import random

from neural_net import MLP

SHAPES = [
    [  # croix
        "..#..",
        "..#..",
        "#####",
        "..#..",
        "..#..",
    ],
    [  # carré
        "#####",
        "#...#",
        "#...#",
        "#...#",
        "#####",
    ],
    [  # X
        "#...#",
        ".#.#.",
        "..#..",
        ".#.#.",
        "#...#",
    ],
]


def to_vector(rows):
    return [1.0 if c == "#" else 0.0 for row in rows for c in row]


def noisy_sample(label, rng, flip=0.08):
    x = to_vector(SHAPES[label])
    return [1.0 - v if rng.random() < flip else v for v in x]


def make_dataset(n_per_class, seed):
    rng = random.Random(seed)
    data = [(noisy_sample(c, rng), c) for c in range(len(SHAPES)) for _ in range(n_per_class)]
    rng.shuffle(data)
    return data


def main():
    train = make_dataset(300, seed=1)
    test = make_dataset(100, seed=2)

    net = MLP([25, 16, 3], seed=0)
    print(f"Précision avant entraînement : {net.accuracy(test):.2%}")
    net.fit(train, epochs=15, lr=0.05)
    print(f"Précision après entraînement : {net.accuracy(test):.2%}")

    names = ["croix", "carré", "X"]
    x, y = test[0]
    probs = net.predict_proba(x)
    print(f"Exemple : vrai = {names[y]}, prédit = {names[net.predict(x)]}, "
          f"probas = {[round(p, 3) for p in probs]}")


if __name__ == "__main__":
    main()
