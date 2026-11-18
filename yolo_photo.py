"""Détection des boutons en temps réel avec le modèle YOLO entraîné (train_yolo.py).

Usage : python yolo_photo.py [photo.jpg] [--live] [--cam 1] [--conf 0.25] [--model yolo_model.pt]
  comme cnn_photo.py : sans photo, ESPACE prend la photo ; --live = temps réel (+ / - seuil, q quitte).

Le visage est d'abord repéré (MediaPipe) puis découpé avec une marge : YOLO ne voit que le visage, agrandi à
peu près comme dans les photos d'entraînement (visage plein cadre) -> les boutons gardent la bonne taille.
"""

import argparse
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

from cnn_photo import snapshot

MARGIN = 0.15  # marge autour du visage, en fraction de sa taille
COLORS = {"blackheads": (255, 120, 0), "dark_spot": (0, 200, 255), "nodules": (0, 0, 255),
          "papules": (255, 0, 200), "pustules": (0, 140, 255), "whiteheads": (255, 255, 0)}


def face_box(landmarker, bgr):
    """(x0, y0, x1, y1) du visage avec marge, ou None."""
    res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    if not res.face_landmarks:
        return None
    h, w = bgr.shape[:2]
    p = np.array([(m.x * w, m.y * h) for m in res.face_landmarks[0]])
    (x0, y0), (x1, y1) = p.min(0), p.max(0)
    mx, my = (x1 - x0) * MARGIN, (y1 - y0) * MARGIN
    return max(0, int(x0 - mx)), max(0, int(y0 - my)), min(w, int(x1 + mx)), min(h, int(y1 + my))


def analyze(bgr, landmarker, model, conf):
    """Dessine les boutons sur bgr ; renvoie (nombre de boutons, visage trouvé)."""
    box = face_box(landmarker, bgr)
    if box is None:
        return 0, False
    x0, y0, x1, y1 = box
    r = model.predict(bgr[y0:y1, x0:x1], imgsz=640, conf=conf, verbose=False)[0]
    thick = max(2, bgr.shape[1] // 600)
    cv2.rectangle(bgr, (x0, y0), (x1, y1), (80, 80, 80), 1)
    for (bx0, by0, bx1, by1), c, k in zip(r.boxes.xyxy.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)):
        name = r.names[k]
        cv2.rectangle(bgr, (x0 + bx0, y0 + by0), (x0 + bx1, y0 + by1), COLORS.get(name, (0, 255, 0)), thick)
        cv2.putText(bgr, f"{name} {c:.0%}", (x0 + bx0, max(y0 + by0 - 5, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS.get(name, (0, 255, 0)), 1)
    return len(r.boxes), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photo", nargs="?")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--model", default="yolo_model.pt")
    args = ap.parse_args()

    model = YOLO(args.model)
    landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path="face_landmarker.task"), num_faces=1))

    if not args.live:  # une photo : fichier donné, sinon webcam + ESPACE (comme cnn_photo.py)
        img = cv2.imread(args.photo) if args.photo else snapshot(args.cam)
        if img is None:
            raise SystemExit(f"Photo illisible : {args.photo}")
        n, found = analyze(img, landmarker, model, args.conf)
        landmarker.close()
        if not found:
            raise SystemExit("Aucun visage détecté : mettez-vous de face, bien éclairé.")
        print(f"{n} bouton(s) entouré(s) -> resultat.jpg")
        cv2.imwrite("resultat.jpg", img)
        cv2.imshow("Resultat (une touche pour fermer)", img)
        cv2.waitKey(0)
        return

    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    conf, last = args.conf, time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise SystemExit("Caméra introuvable : essayez un autre --cam (0, 1, 2…).")
            n, found = analyze(frame, landmarker, model, conf)
            fps, last = 1 / max(time.time() - last, 1e-3), time.time()
            msg = f"{n} bouton(s)" if found else "aucun visage"
            cv2.putText(frame, f"{msg} | seuil {conf:.0%} (+/-) | {fps:.0f} img/s | q = quitter", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("VisuModel YOLO", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in (ord("+"), ord("=")):
                conf = min(0.95, conf + 0.05)
            if key in (ord("-"), ord("6")):
                conf = max(0.05, conf - 0.05)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    main()
