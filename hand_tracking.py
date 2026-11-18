"""Détecte la main en temps réel et encadre chaque doigt en vert avec son nom. Quitter avec la touche q.

La détection des mains est faite par un réseau de neurones (MediaPipe Hand Landmarker, modèle hand_landmarker.task).
"""

import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = "hand_landmarker.task"
GREEN = (0, 255, 0)  # BGR
PAD = 10

# Indices des 21 points de la main : chaque doigt va de la base (MCP) à l'extrémité
FINGERS = {
    "Pouce": [1, 2, 3, 4],
    "Index": [5, 6, 7, 8],
    "Majeur": [9, 10, 11, 12],
    "Annulaire": [13, 14, 15, 16],
    "Auriculaire": [17, 18, 19, 20],
}


def draw_fingers(frame, landmarks):
    h, w = frame.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in landmarks]
    for name, idx in FINGERS.items():
        xs = [pts[i][0] for i in idx]
        ys = [pts[i][1] for i in idx]
        x1, y1 = min(xs) - PAD, min(ys) - PAD
        x2, y2 = max(xs) + PAD, max(ys) + PAD
        cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)
        cv2.putText(frame, name, (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2)


def main():
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Impossible d'ouvrir la caméra.")

    start = time.monotonic()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # effet miroir
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect_for_video(image, int((time.monotonic() - start) * 1000))

            for hand in result.hand_landmarks:
                draw_fingers(frame, hand)

            cv2.imshow("Mains", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
