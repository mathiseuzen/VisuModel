"""Détection des boutons : fenêtre glissante sur la peau isolée + classifieur.

En ligne de commande (teste le modèle sur les photos d'une capture) :
    python detect.py [dossier_de_capture] [--scale 0.12] [--conf 0.7] [--model model_patches.pt]
Résultat : captures/<session>/detections/<photo>.jpg (recadré sur le visage, boîtes dessinées).

Principe : on isole la peau (comme à l'étape 2), on fait glisser une fenêtre carrée qui doit être
ENTIÈREMENT sur de la peau (rien d'autre dedans), on la découpe comme à l'entraînement (64x64) et le réseau
dit : peau saine ou un des 6 types de boutons. On garde les boutons dont la confiance dépasse --conf et on
fusionne les fenêtres qui se chevauchent. --scale : côté de la fenêtre en fraction de la largeur du visage.
"""

import argparse
import glob
import os

import cv2
import numpy as np
import torch

from isolate_skin import make_models, skin_mask
from prepare_patches import HEALTHY, MIN_SIDE, crop_square
from train_patch_cnn import PatchCNN

# BGR
COLORS = {
    "blackheads": (255, 120, 0), "dark_spot": (0, 200, 255), "nodules": (0, 0, 255),
    "papules": (255, 0, 200), "pustules": (0, 140, 255), "whiteheads": (255, 255, 0),
}


def skin_windows(mask, win, stride):
    """Positions (x0, y0) des fenêtres win x win entièrement couvertes de peau (masque > 0)."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return []
    h, w = mask.shape
    integral = cv2.integral((mask > 0).astype(np.uint8))  # somme d'une zone en 4 lectures
    found = []
    for y0 in range(ys.min(), ys.max() - win + 2, stride):
        for x0 in range(xs.min(), xs.max() - win + 2, stride):
            x1, y1 = x0 + win, y0 + win
            if x1 <= w and y1 <= h and integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0] == win * win:
                found.append((x0, y0))
    return found


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def nms(boxes, scores, iou_max=0.3):
    """Indices à garder : on prend les plus confiantes et on écarte celles qui les chevauchent trop."""
    keep = []
    for i in np.argsort(scores)[::-1]:
        if all(iou(boxes[i], boxes[j]) <= iou_max for j in keep):
            keep.append(int(i))
    return keep


def classify(model, crops):
    """crops : liste d'imagettes BGR 64x64 -> probabilités (N, classes)."""
    x = np.stack(crops)[..., ::-1].astype(np.float32) / 255.0  # BGR -> RGB comme à l'entraînement
    x = torch.from_numpy(((x - 0.5) / 0.5).copy()).permute(0, 3, 1, 2)
    out = []
    with torch.no_grad():
        for i in range(0, len(x), 256):
            out.append(torch.softmax(model(x[i:i + 256]), 1).numpy())
    return np.concatenate(out)


def load_model(path="model_patches.pt"):
    """(réseau prêt à l'emploi, noms des classes). Chargement sécurisé : poids uniquement."""
    ck = torch.load(path, weights_only=True, map_location="cpu")
    model = PatchCNN(len(ck["classes"]))
    model.load_state_dict(ck["state_dict"])
    return model.eval(), ck["classes"]


def find_lesions(bgr, landmarker, segmenter, model, classes, scale=0.12, conf=0.7):
    """Analyse une photo BGR. Renvoie None si aucun visage, sinon (infos, détections).

    détections : liste de (x0, y0, x1, y1, classe, confiance), en pixels de la photo.
    infos : points du visage, taille de fenêtre, nombre de fenêtres, part de fenêtres classées bouton.
    """
    mask, pts = skin_mask(bgr, landmarker, segmenter)
    if mask is None:
        return None
    face_w = int(pts[:, 0].max() - pts[:, 0].min())
    face_h = int(pts[:, 1].max() - pts[:, 1].min())
    win = max(MIN_SIDE, round(scale * face_w))
    positions = skin_windows(mask, win, max(1, win // 2))
    info = {"pts": pts, "face_w": face_w, "face_h": face_h, "win": win, "n_windows": len(positions), "lesion_rate": 0.0}
    if not positions:
        return info, []

    probs = classify(model, [crop_square(bgr, x + win / 2, y + win / 2, win) for x, y in positions])
    pred, best = probs.argmax(1), probs.max(1)
    lesion = pred != classes.index(HEALTHY)
    info["lesion_rate"] = float(lesion.mean())
    idx = np.nonzero(lesion & (best >= conf))[0]
    boxes = [(positions[i][0], positions[i][1], positions[i][0] + win, positions[i][1] + win) for i in idx]
    keep = nms(boxes, best[idx]) if len(idx) else []
    return info, [(*boxes[k], classes[pred[idx[k]]], float(best[idx[k]])) for k in keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?")
    ap.add_argument("--scale", type=float, default=0.12)
    ap.add_argument("--conf", type=float, default=0.7)
    ap.add_argument("--model", default="model_patches.pt")
    args = ap.parse_args()

    folder = args.folder
    if folder is None:
        sessions = sorted(glob.glob(os.path.join("captures", "*")))
        if not sessions:
            raise SystemExit("Aucune capture trouvée : lancez d'abord capture.py.")
        folder = sessions[-1]
    photos = sorted(glob.glob(os.path.join(folder, "*.jpg")))
    if not photos:
        raise SystemExit(f"Aucune photo .jpg dans {folder}")
    out_dir = os.path.join(folder, "detections")
    os.makedirs(out_dir, exist_ok=True)

    model, classes = load_model(args.model)
    landmarker, segmenter = make_models()
    try:
        for path in photos:
            stem = os.path.splitext(os.path.basename(path))[0]
            bgr = cv2.imread(path)
            result = find_lesions(bgr, landmarker, segmenter, model, classes, args.scale, args.conf)
            if result is None:
                print(f"{stem} : aucun visage")
                continue
            info, detections = result
            if info["n_windows"] == 0:
                print(f"{stem} : aucune fenêtre de {info['win']} px entièrement sur la peau")
                continue

            # Vue recadrée sur le visage et agrandie, pour que les petites boîtes soient lisibles
            pts, face_w = info["pts"], info["face_w"]
            m = int(face_w * 0.1)
            cx0, cy0 = max(0, int(pts[:, 0].min()) - m), max(0, int(pts[:, 1].min()) - m)
            cx1, cy1 = min(bgr.shape[1], int(pts[:, 0].max()) + m), min(bgr.shape[0], int(pts[:, 1].max()) + m)
            s = 900 / (cx1 - cx0)
            view = cv2.resize(bgr[cy0:cy1, cx0:cx1], None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
            counts = {}
            for x0, y0, x1, y1, name, c in detections:
                counts[name] = counts.get(name, 0) + 1
                p0, p1 = (int((x0 - cx0) * s), int((y0 - cy0) * s)), (int((x1 - cx0) * s), int((y1 - cy0) * s))
                cv2.rectangle(view, p0, p1, COLORS[name], 2)
                cv2.putText(view, f"{name} {c:.2f}", (p0[0], max(p0[1] - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS[name], 1)
            cv2.imwrite(os.path.join(out_dir, stem + ".jpg"), view)

            print(f"{stem} : fenêtre {info['win']} px ({info['win'] / face_w:.0%} du visage), {info['n_windows']} fenêtres sur la peau, "
                  f"{info['lesion_rate']:.0%} classées bouton, {len(detections)} boutons retenus (confiance >= {args.conf}) "
                  f"{counts if counts else ''}")
    finally:
        landmarker.close()
        segmenter.close()
    print(f"\nImages dans {out_dir}")


if __name__ == "__main__":
    main()
