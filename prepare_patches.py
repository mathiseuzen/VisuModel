"""Prépare les imagettes (patches) pour entraîner le classifieur à partir d'un jeu de données YOLO.

Usage : python prepare_patches.py [--src archive] [--out datasets/patches]

Pour chaque image du jeu de données :
  - chaque boîte devient une imagette 64x64 (carré centré sur la boîte, avec du contexte autour),
    rangée dans <out>/<split>/<classe>/ ;
  - autant de carrés de PEAU SAINE sont tirés au hasard dans <out>/<split>/peau_saine/ : uniquement
    sur de la peau (modèle de segmentation, comme pour nos photos) et sans toucher aucune boîte.
Le découpage train / valid / test du jeu de données est conservé tel quel (pas de fuite entre les deux).
"""

import argparse
import ast
import glob
import os
import random
import re

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

PATCH = 64            # taille finale des imagettes
CONTEXT = 2.0         # côté du carré = CONTEXT x le plus grand côté de la boîte
MIN_SIDE = 24         # côté minimal du carré découpé (évite d'agrandir 8 px en 64 px)
MIN_BOX_PX = 6        # boîtes plus petites que ça (dans l'image d'origine) : ignorées
SKIN_CLASSES = (2, 3)  # peau du corps ou du visage, d'après le modèle de segmentation
SKIN_SHARE = 0.9      # un carré « peau saine » doit être couvert à 90 % par de la peau
MAX_OVERLAP = 0.1     # ... et ne recouvrir aucune boîte de plus de 10 % de sa surface
SEG_SIZE = 1024       # la segmentation est calculée sur une version réduite de l'image
SEG_MODEL = "selfie_multiclass.tflite"
HEALTHY = "peau_saine"


def crop_square(img, cx, cy, side):
    """Carré de côté `side` centré sur (cx, cy), décalé pour rester dans l'image, redimensionné en PATCH x PATCH."""
    h, w = img.shape[:2]
    side = int(min(max(side, MIN_SIDE), h, w))
    x0 = int(np.clip(cx - side / 2, 0, w - side))
    y0 = int(np.clip(cy - side / 2, 0, h - side))
    interp = cv2.INTER_AREA if side > PATCH else cv2.INTER_CUBIC
    return cv2.resize(img[y0:y0 + side, x0:x0 + side], (PATCH, PATCH), interpolation=interp)


def overlap_ratio(patch, box):
    """Intersection / surface de la plus petite des deux zones ; les deux sont (x0, y0, x1, y1).

    Vaut 1 si l'une contient l'autre, quelle que soit la plus grande.
    """
    ix = max(0, min(patch[2], box[2]) - max(patch[0], box[0]))
    iy = max(0, min(patch[3], box[3]) - max(patch[1], box[1]))
    areas = ((patch[2] - patch[0]) * (patch[3] - patch[1]), (box[2] - box[0]) * (box[3] - box[1]))
    return ix * iy / max(min(areas), 1e-9)


def sample_healthy(skin, boxes, sides, n, rng):
    """Jusqu'à n carrés (x0, y0, côté) de peau saine : couverts de peau et loin des boîtes."""
    h, w = skin.shape
    integral = cv2.integral(skin.astype(np.uint8))  # somme d'une zone en 4 lectures
    found = []
    for _ in range(n * 30):
        if len(found) == n:
            break
        side = int(min(max(rng.choice(sides), MIN_SIDE), h, w))
        x0, y0 = rng.randint(0, w - side), rng.randint(0, h - side)
        x1, y1 = x0 + side, y0 + side
        skin_pixels = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
        if skin_pixels / (side * side) < SKIN_SHARE:
            continue
        if any(overlap_ratio((x0, y0, x1, y1), b) > MAX_OVERLAP for b in boxes):
            continue
        found.append((x0, y0, side))
    return found


def skin_map(img, segmenter):
    """Tableau booléen (h, w) : True là où le modèle voit de la peau."""
    h, w = img.shape[:2]
    scale = min(1.0, SEG_SIZE / max(h, w))
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    seg = np.squeeze(segmenter.segment(image).category_mask.numpy_view())
    skin = np.isin(seg, SKIN_CLASSES).astype(np.uint8)
    return cv2.resize(skin, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)


def read_class_names(src):
    text = open(os.path.join(src, "data.yaml"), encoding="utf-8").read()
    return [n.replace(" ", "_") for n in ast.literal_eval(re.search(r"names:\s*(\[.*?\])", text).group(1))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="archive")
    ap.add_argument("--out", default="datasets/patches")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    names = read_class_names(args.src)
    segmenter = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
        base_options=mp_python.BaseOptions(model_asset_path=SEG_MODEL), output_category_mask=True))

    counts = {}
    try:
        for split in ("train", "valid", "test"):
            files = sorted(glob.glob(os.path.join(args.src, split, "images", "*.jpg")))
            for cls in names + [HEALTHY]:
                os.makedirs(os.path.join(args.out, split, cls), exist_ok=True)
            for i, path in enumerate(files):
                if i % 100 == 0:
                    print(f"{split} : {i}/{len(files)}", flush=True)
                img = cv2.imread(path)
                h, w = img.shape[:2]
                stem = os.path.splitext(os.path.basename(path))[0]
                label_path = os.path.join(args.src, split, "labels", stem + ".txt")
                rows = [r.split() for r in open(label_path).read().splitlines() if r.strip()]

                boxes, sides, k = [], [], 0
                for cls_id, cx, cy, bw, bh in ((int(r[0]), *map(float, r[1:5])) for r in rows):
                    cx, cy, bw, bh = cx * w, cy * h, bw * w, bh * h
                    if min(bw, bh) < MIN_BOX_PX:
                        continue
                    boxes.append((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
                    side = max(bw, bh) * CONTEXT
                    sides.append(side)
                    cls = names[cls_id]
                    cv2.imwrite(os.path.join(args.out, split, cls, f"{stem[:20]}_{k}.png"), crop_square(img, cx, cy, side))
                    counts[(split, cls)] = counts.get((split, cls), 0) + 1
                    k += 1

                if sides:
                    skin = skin_map(img, segmenter)
                    for j, (x0, y0, side) in enumerate(sample_healthy(skin, boxes, sides, len(sides), rng)):
                        patch = crop_square(img, x0 + side / 2, y0 + side / 2, side)
                        cv2.imwrite(os.path.join(args.out, split, HEALTHY, f"{stem[:20]}_{j}.png"), patch)
                        counts[(split, HEALTHY)] = counts.get((split, HEALTHY), 0) + 1
    finally:
        segmenter.close()

    print("\nimagettes créées :")
    print(f"{'classe':12s}" + "".join(f"{s:>8s}" for s in ("train", "valid", "test")))
    for cls in names + [HEALTHY]:
        print(f"{cls:12s}" + "".join(f"{counts.get((s, cls), 0):8d}" for s in ("train", "valid", "test")))


if __name__ == "__main__":
    main()
