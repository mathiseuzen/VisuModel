"""Rendu « radar » anonyme : maillage du visage + imperfections détectées, sans aucun pixel de la photo.

Usage : python yolo_wireframe.py photo.jpg [--conf 0.25] [--out output/wireframe.png] [--model yolo_model.pt]
"""

import argparse
import os

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

from yolo_photo import face_box

BG, MESH, CONTOUR, BOX, TEXT = (42, 23, 15), (90, 70, 40), (255, 220, 0), (0, 165, 255), (235, 235, 235)  # BGR
PANEL, H = 340, 900
ZONES = ("front", "joue gauche", "joue droite", "nez / bouche", "menton")
CONN = vision.FaceLandmarksConnections


def zone(cx, cy, p):
    """Zone du visage d'un point, repérée avec le maillage (9 = entre les sourcils, 1 = nez, 17 = bas de la lèvre)."""
    if cy < p[9][1]:
        return "front"
    if cy > p[17][1]:
        return "menton"
    if abs(cx - p[1][0]) < 0.12 * (p[:, 0].max() - p[:, 0].min()):
        return "nez / bouche"
    return "joue gauche" if cx < p[1][0] else "joue droite"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photo")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default="output/wireframe.png")
    ap.add_argument("--model", default="yolo_model.pt")
    args = ap.parse_args()

    bgr = cv2.imread(args.photo)
    if bgr is None:
        raise SystemExit(f"Photo illisible : {args.photo}")
    landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path="face_landmarker.task"), num_faces=1))
    res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    box = face_box(landmarker, bgr)
    landmarker.close()
    if box is None:
        raise SystemExit("Aucun visage détecté.")
    x0, y0, x1, y1 = box
    r = YOLO(args.model).predict(bgr[y0:y1, x0:x1], imgsz=1024, conf=args.conf, verbose=False)[0]

    h, w = bgr.shape[:2]
    p = np.array([(m.x * w, m.y * h) for m in res.face_landmarks[0]])
    s = (H - 60) / (y1 - y0)  # échelle : le cadre du visage remplit la hauteur
    ox, oy = 30 - x0 * s, 30 - y0 * s
    pt = lambda x, y: (int(x * s + ox), int(y * s + oy))
    cw = int((x1 - x0) * s) + 60
    canvas = np.full((H, cw + PANEL, 3), BG, np.uint8)

    mesh = canvas.copy()  # maillage fin en transparence, contours par-dessus
    for c in CONN.FACE_LANDMARKS_TESSELATION:
        cv2.line(mesh, pt(*p[c.start]), pt(*p[c.end]), MESH, 1, cv2.LINE_AA)
    canvas = cv2.addWeighted(mesh, 0.8, canvas, 0.2, 0)
    for c in CONN.FACE_LANDMARKS_CONTOURS:
        cv2.line(canvas, pt(*p[c.start]), pt(*p[c.end]), CONTOUR, 1, cv2.LINE_AA)

    counts = dict.fromkeys(ZONES, 0)
    for (bx0, by0, bx1, by1), c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
        a, b = (x0 + bx0, y0 + by0), (x0 + bx1, y0 + by1)
        counts[zone((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, p)] += 1
        cv2.rectangle(canvas, pt(*a), pt(*b), BOX, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"imperfection {c:.2f}", (pt(*a)[0], pt(*a)[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, BOX, 1, cv2.LINE_AA)

    px, font = cw + 20, cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "VisuModel", (px, 50), font, 0.9, CONTOUR, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"seuil {args.conf:.0%}", (px, 80), font, 0.5, TEXT, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{len(r.boxes)} imperfection(s)", (px, 130), font, 0.7, BOX, 2, cv2.LINE_AA)
    for i, z in enumerate(ZONES):
        y = 190 + i * 50
        cv2.putText(canvas, f"{z}: {counts[z]}", (px, y), font, 0.55, TEXT, 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (px, y + 10), (px + counts[z] * 20, y + 18), BOX, -1)
    cv2.putText(canvas, "Outil experimental, aucun diagnostic.", (px, H - 20), font, 0.4, TEXT, 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cv2.imwrite(args.out, canvas)
    print(f"{len(r.boxes)} imperfection(s) -> {args.out}", counts)


if __name__ == "__main__":
    main()
