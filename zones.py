"""Découpe la peau de chaque photo en zones rectangulaires 100 % remplies de peau.

Usage : python zones.py [dossier_de_capture]     (par défaut : la dernière capture)

Zones : front, nez, joue_gauche, joue_droite, menton ("gauche"/"droite" = côté de la personne,
donc sur la photo la joue gauche est à droite de l'image).

Pour chaque zone, on cherche le plus grand rectangle sans aucun pixel hors peau (ni yeux,
sourcils, lèvres, cheveux, fond) : le fichier est une image RGB ordinaire, sans canal alpha,
prête pour l'entraînement. Résultat : captures/<session>/zones/<photo>/<zone>.png
et un aperçu captures/<session>/zones/<photo>_apercu.jpg (rectangles dessinés sur la photo).
"""

import glob
import os
import sys

import cv2
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from isolate_skin import FACE_MODEL, FOREHEAD_EXTENSION, SEG_MODEL, skin_mask

MIN_SIDE = 32  # une zone dont un côté fait moins de 32 px est ignorée (inutilisable pour l'entraînement)


def largest_rectangle(binary):
    """Plus grand rectangle rempli de True : (x0, y0, x1, y1), bornes x1/y1 exclues, ou None."""
    h, w = binary.shape
    heights = np.zeros(w, int)
    best_area, best = 0, None
    for y in range(h):
        heights = np.where(binary[y], heights + 1, 0)
        stack = []  # (colonne de départ, hauteur), hauteurs croissantes
        for x in range(w + 1):
            cur = heights[x] if x < w else 0
            start = x
            while stack and stack[-1][1] >= cur:
                s, hh = stack.pop()
                if hh * (x - s) > best_area:
                    best_area, best = hh * (x - s), (s, y - hh + 1, x, y + 1)
                start = s
            stack.append((start, cur))
    return best


def zone_boxes(pts):
    """Zones de recherche (x0, y0, x1, y1) en pixels, d'après des points repères du maillage MediaPipe."""
    X = lambda i: int(pts[i, 0])
    Y = lambda i: int(pts[i, 1])
    face_a, face_b = sorted((X(234), X(454)))   # bords du visage
    nose_a, nose_b = sorted((X(129), X(358)))   # ailes du nez
    brow_a, brow_b = sorted((X(70), X(300)))    # extrémités des sourcils
    mouth_a, mouth_b = sorted((X(61), X(291)))  # commissures des lèvres
    eye_bottom = max(Y(145), Y(374))
    brow_top = min(Y(105), Y(334))
    forehead_top = Y(10) - int(FOREHEAD_EXTENSION * (pts[:, 1].max() - pts[:, 1].min()))
    return {
        "front": (brow_a, forehead_top, brow_b, brow_top),
        "nez": (nose_a, Y(168), nose_b, Y(2)),
        "joue_droite": (face_a, eye_bottom, nose_a, Y(17)),  # à gauche sur la photo = joue droite de la personne
        "joue_gauche": (nose_b, eye_bottom, face_b, Y(17)),
        "menton": (mouth_a, Y(17), mouth_b, Y(152)),
    }


def find_zones(mask, pts):
    """{zone: (x0, y0, x1, y1)} pour les zones où un rectangle de peau pleine existe ; + liste des ignorées."""
    h, w = mask.shape
    found, skipped = {}, []
    for name, (x0, y0, x1, y1) in zone_boxes(pts).items():
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
        rect = largest_rectangle(mask[y0:y1, x0:x1] > 0) if x1 > x0 and y1 > y0 else None
        if rect is None or rect[2] - rect[0] < MIN_SIDE or rect[3] - rect[1] < MIN_SIDE:
            skipped.append(name)
            continue
        found[name] = (rect[0] + x0, rect[1] + y0, rect[2] + x0, rect[3] + y0)
    return found, skipped


def main():
    if len(sys.argv) > 1:
        folder = sys.argv[1]
    else:
        sessions = sorted(glob.glob(os.path.join("captures", "*")))
        if not sessions:
            raise SystemExit("Aucune capture trouvée : lancez d'abord capture.py.")
        folder = sessions[-1]
    photos = sorted(glob.glob(os.path.join(folder, "*.jpg")))
    if not photos:
        raise SystemExit(f"Aucune photo .jpg dans {folder}")

    landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL), num_faces=1))
    segmenter = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
        base_options=mp_python.BaseOptions(model_asset_path=SEG_MODEL), output_category_mask=True))

    try:
        for path in photos:
            stem = os.path.splitext(os.path.basename(path))[0]
            bgr = cv2.imread(path)
            mask, pts = skin_mask(bgr, landmarker, segmenter)
            if mask is None:
                print(f"{stem} : aucun visage")
                continue
            found, skipped = find_zones(mask, pts)

            out_dir = os.path.join(folder, "zones", stem)
            os.makedirs(out_dir, exist_ok=True)
            preview = bgr.copy()
            for name, (x0, y0, x1, y1) in found.items():
                assert (mask[y0:y1, x0:x1] > 0).all(), f"{name} contient des pixels hors peau"
                cv2.imwrite(os.path.join(out_dir, name + ".png"), bgr[y0:y1, x0:x1])  # RGB, pas d'alpha
                cv2.rectangle(preview, (x0, y0), (x1, y1), (0, 255, 0), 2)
                cv2.putText(preview, name, (x0, max(y0 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imwrite(os.path.join(folder, "zones", stem + "_apercu.jpg"), preview)

            sizes = ", ".join(f"{n} {x1 - x0}x{y1 - y0}" for n, (x0, y0, x1, y1) in found.items())
            print(f"{stem} : {sizes}" + (f" | ignorées (trop petites) : {', '.join(skipped)}" if skipped else ""))
    finally:
        landmarker.close()
        segmenter.close()


if __name__ == "__main__":
    main()
