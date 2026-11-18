"""Réseau de neurones multicouche écrit de zéro (Python pur, sans dépendance).

Couches cachées : tanh. Couche de sortie : softmax. Perte : entropie croisée.
Apprentissage : descente de gradient stochastique avec rétropropagation.
"""

import math
import random


def softmax(z):
    m = max(z)  # soustraire le max évite les débordements
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


class MLP:
    def __init__(self, sizes, seed=0):
        """sizes = [entrées, couche cachée 1, ..., sorties]."""
        rng = random.Random(seed)
        self.W = [
            [[rng.gauss(0.0, math.sqrt(2.0 / n_in)) for _ in range(n_in)] for _ in range(n_out)]
            for n_in, n_out in zip(sizes, sizes[1:])
        ]
        self.b = [[0.0] * n_out for n_out in sizes[1:]]

    def forward(self, x):
        """Renvoie les activations de chaque couche (entrée incluse)."""
        acts = [x]
        last = len(self.W) - 1
        for li, (W, b) in enumerate(zip(self.W, self.b)):
            a_prev = acts[-1]
            z = [sum(w * a for w, a in zip(row, a_prev)) + bj for row, bj in zip(W, b)]
            acts.append(softmax(z) if li == last else [math.tanh(v) for v in z])
        return acts

    def predict_proba(self, x):
        return self.forward(x)[-1]

    def predict(self, x):
        probs = self.predict_proba(x)
        return probs.index(max(probs))

    def train_step(self, x, label, lr):
        """Un pas de SGD sur un exemple. Renvoie la perte avant mise à jour."""
        acts = self.forward(x)
        out = acts[-1]
        loss = -math.log(max(out[label], 1e-12))
        # Gradient de softmax + entropie croisée : p - onehot(label)
        delta = [p - (1.0 if i == label else 0.0) for i, p in enumerate(out)]
        for li in reversed(range(len(self.W))):
            a_prev = acts[li]
            if li > 0:
                # Propager l'erreur avant de modifier W ; dérivée de tanh = 1 - a^2
                W = self.W[li]
                delta_prev = [
                    sum(W[j][i] * delta[j] for j in range(len(delta))) * (1.0 - a_prev[i] ** 2)
                    for i in range(len(a_prev))
                ]
            for j, row in enumerate(self.W[li]):
                g = lr * delta[j]
                for i in range(len(row)):
                    row[i] -= g * a_prev[i]
                self.b[li][j] -= g
            if li > 0:
                delta = delta_prev
        return loss

    def fit(self, data, epochs=20, lr=0.05, seed=0):
        """data = liste de (x, label). Mélange à chaque époque."""
        rng = random.Random(seed)
        data = list(data)
        for _ in range(epochs):
            rng.shuffle(data)
            for x, y in data:
                self.train_step(x, y, lr)
        return self

    def accuracy(self, data):
        return sum(self.predict(x) == y for x, y in data) / len(data)
