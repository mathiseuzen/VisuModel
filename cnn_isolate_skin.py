"""Isole la peau du visage sur chaque photo d'une capture.

Usage : python cnn_isolate_skin.py [dossier_de_capture]     (par défaut : la dernière capture)

Pour chaque photo, écrit captures/<session>/peau/<nom>.png : une image recadrée où tout ce qui
n'est pas de la peau (fond, cheveux, sourcils, yeux, lèvres) est transparent ET vidé (pixels à 0).

Le masque de peau est l'intersection de :
  - l'ovale du visage (points MediaPipe), moins les yeux, sourcils et lèvres ;
  - la classe « peau du visage » d'un modèle de segmentation (retire cheveux, barbe, fond).
Il est ensuite légèrement érodé pour ne garder aucun pixel de bordure.
"""

import glob
import os
import sys

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

FACE_MODEL = "face_landmarker.task"
SEG_MODEL = "selfie_multiclass.tflite"
# Le maillage s'arrête au milieu du front : on prolonge l'ovale vers le haut de 30 % de la hauteur
# du visage, puis la segmentation coupe au bord des cheveux.
FOREHEAD_EXTENSION = 0.3
FACE_SKIN = 3  # classes : 0 fond, 1 cheveux, 2 peau du corps, 3 peau du visage, 4 vêtements, 5 autres
CONN = vision.FaceLandmarksConnections


def _indices(connections):
    return sorted({i for c in connections for i in (c.start, c.end)})


OVAL = _indices(CONN.FACE_LANDMARKS_FACE_OVAL)
EXCLUDED = [
    _indices(c)
    for c in (
        CONN.FACE_LANDMARKS_LEFT_EYE,
        CONN.FACE_LANDMARKS_RIGHT_EYE,
        CONN.FACE_LANDMARKS_LEFT_EYEBROW,
        CONN.FACE_LANDMARKS_RIGHT_EYEBROW,
        CONN.FACE_LANDMARKS_LIPS,
    )
]


def _region(pts, idx, shape):
    m = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(pts[idx]), 255)
    return m


def skin_mask(bgr, landmarker, segmenter):
    """Renvoie (masque uint8 255 = peau, points du visage en pixels), ou (None, None) si aucun visage."""
    h, w = bgr.shape[:2]
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    found = landmarker.detect(image)
    if not found.face_landmarks:
        return None, None
    pts = np.array([(p.x * w, p.y * h) for p in found.face_landmarks[0]], np.int32)

    face_h = pts[:, 1].max() - pts[:, 1].min()
    oval = pts[OVAL]
    upper = oval[oval[:, 1] < pts[:, 1].mean()] - np.array([0, int(FOREHEAD_EXTENSION * face_h)])
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(np.vstack([oval, upper]).astype(np.int32)), 255)
    k =max(3, int((pts[:, 0].max() - pts[:, 0].min()) * 0.02)) | 1  # marge de ~2 % de la largeur du visage
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    for idx in EXCLUDED:
        mask[cv2.dilate(_region(pts, idx, (h, w)), kernel) > 0] = 0

    seg = np.squeeze(segmenter.segment(image).category_mask.numpy_view())  # (h, w, 1) -> (h, w)
    if seg.shape != (h, w):
        seg = cv2.resize(seg, (w, h), interpolation=cv2.INTER_NEAREST)
    mask[seg != FACE_SKIN] = 0
    return cv2.erode(mask, kernel), pts


def to_rgba(bgr, mask):
    """Image BGRA recadrée sur la peau ; hors masque : transparent et pixels à 0."""
    ys, xs = np.nonzero(mask)
    rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    rgba[mask == 0] = 0
    rgba[..., 3] = mask
    return rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def make_models():
    """(détecteur de points du visage, segmenteur de peau) prêts à l'emploi ; à fermer avec .close()."""
    landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL), num_faces=1))
    segmenter = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
        base_options=mp_python.BaseOptions(model_asset_path=SEG_MODEL), output_category_mask=True))
    return landmarker, segmenter


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
    out_dir = os.path.join(folder, "peau")
    os.makedirs(out_dir, exist_ok=True)

    landmarker, segmenter = make_models()

    try:
        for path in photos:
            bgr = cv2.imread(path)
            mask, _ = skin_mask(bgr, landmarker, segmenter)
            if mask is None or not mask.any():
                print(f"Aucun visage/peau trouvé : {path}")
                continue
            out = os.path.join(out_dir, os.path.splitext(os.path.basename(path))[0] + ".png")
            cv2.imwrite(out, to_rgba(bgr, mask))
            print(f"{os.path.basename(path)} : peau = {mask.mean() / 255:.1%} de l'image -> {out}")
    finally:
        landmarker.close()
        segmenter.close()


if __name__ == "__main__":
    main()
