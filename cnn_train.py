"""Entraîne un CNN écrit de zéro (PyTorch) à classer les imagettes 64x64 : 6 types de boutons + peau saine.

Usage : python cnn_train.py [--data datasets/patches] [--epochs 30] [--out cnn_model.pt]

Tourne à l'identique sur processeur, sur GPU, et sur Kaggle (utilise le GPU s'il existe).
Le meilleur modèle est choisi sur la validation (F1 moyen des classes), puis mesuré UNE fois sur le test.
"""

import argparse
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class PatchCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        # 64 -> 32 -> 16 -> 8, puis moyenne globale : fonctionne aussi sur d'autres tailles d'imagettes
        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def confusion(preds, labels, n):
    """Matrice n x n : lignes = vraie classe, colonnes = classe prédite."""
    m = np.zeros((n, n), dtype=int)
    np.add.at(m, (np.asarray(labels), np.asarray(preds)), 1)
    return m


def scores(m):
    """Précision, rappel et F1 de chaque classe d'après la matrice de confusion."""
    tp = np.diag(m).astype(float)
    precision = tp / np.maximum(m.sum(0), 1)
    recall = tp / np.maximum(m.sum(1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    return precision, recall, f1


def evaluate(model, loader, device, n_classes):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            preds += model(x.to(device)).argmax(1).cpu().tolist()
            labels += y.tolist()
    return confusion(preds, labels, n_classes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/patches")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default="cnn_model.pt")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    norm = transforms.Normalize((0.5,) * 3, (0.5,) * 3)
    augment = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomChoice([transforms.RandomRotation((a, a)) for a in (0, 90, 180, 270)]),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),  # pas de teinte : la rougeur compte
        transforms.ToTensor(), norm,
    ])
    plain = transforms.Compose([transforms.ToTensor(), norm])
    train = datasets.ImageFolder(f"{args.data}/train", augment)
    valid = datasets.ImageFolder(f"{args.data}/valid", plain)
    test = datasets.ImageFolder(f"{args.data}/test", plain)
    classes, n = train.classes, len(train.classes)
    loader = lambda ds, shuffle: DataLoader(ds, batch_size=args.batch, shuffle=shuffle, num_workers=args.workers)
    train_loader, valid_loader, test_loader = loader(train, True), loader(valid, False), loader(test, False)

    # Classes rares pondérées plus fort : racine de l'inverse de la fréquence (sans exagérer)
    counts = np.bincount(train.targets, minlength=n)
    weights = torch.tensor((counts.sum() / (n * counts)) ** 0.5, dtype=torch.float32, device=device)

    model = PatchCNN(n).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    print(f"classes : {classes}")
    print(f"images : train {len(train)}, valid {len(valid)}, test {len(test)} | appareil : {device} | "
          f"paramètres : {sum(p.numel() for p in model.parameters()):,}")
    print(f"référence (toujours répondre « {classes[counts.argmax()]} ») : {counts.max() / counts.sum():.1%} de précision\n")

    best_f1 = -1.0
    for epoch in range(args.epochs):
        start = time.time()
        model.train()
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        sched.step()

        m = evaluate(model, valid_loader, device, n)
        f1 = scores(m)[2].mean()
        mark = ""
        if f1 > best_f1:
            best_f1, mark = f1, "  <- meilleur"
            torch.save({"state_dict": model.state_dict(), "classes": classes}, args.out)
        print(f"époque {epoch + 1:2d}/{args.epochs} | perte {total / len(train):.3f} | "
              f"valid : précision {np.diag(m).sum() / m.sum():.1%}, F1 moyen {f1:.3f} | {time.time() - start:.0f} s{mark}", flush=True)

    # Mesure finale sur le test, avec le meilleur modèle de la validation
    model.load_state_dict(torch.load(args.out, weights_only=True)["state_dict"])
    m = evaluate(model, test_loader, device, n)
    precision, recall, f1 = scores(m)
    print(f"\n=== TEST : précision globale {np.diag(m).sum() / m.sum():.1%}, F1 moyen {f1.mean():.3f}")
    print(f"{'classe':12s}{'précision':>10s}{'rappel':>9s}{'F1':>8s}{'images':>8s}")
    for i, c in enumerate(classes):
        print(f"{c:12s}{precision[i]:10.2f}{recall[i]:9.2f}{f1[i]:8.2f}{m[i].sum():8d}")
    print("\nmatrice de confusion (lignes = vrai, colonnes = prédit) :")
    print(" " * 12 + "".join(f"{c[:6]:>8s}" for c in classes))
    for i, c in enumerate(classes):
        print(f"{c:12s}" + "".join(f"{v:8d}" for v in m[i]))
    print(f"\nmodèle enregistré dans {args.out}")


if __name__ == "__main__":
    main()
