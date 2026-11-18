"""Capture guidée du visage : face, menton levé, tête baissée, côté gauche, côté droit.

Placez votre visage dans l'ovale. L'écran indique quelle pose prendre ; la photo part
toute seule quand le visage est centré et la pose tenue 1 seconde.
Touches : ESPACE = prendre la photo tout de suite, q = quitter.
Les photos (sans aucun dessin dessus) sont enregistrées dans captures/<date_heure>/.
"""

import os
import time
from datetime import datetime

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = "face_landmarker.task"
OUT_DIR = "captures"
HOLD_SECONDS = 1.0
YAW_SIDE = 0.12      # écart de rotation (gauche/droite) par rapport à la pose de face
PITCH_DELTA = 0.06   # écart d'inclinaison (haut/bas) par rapport à la pose de face
NOSE, FOREHEAD, CHIN, CHEEK_A, CHEEK_B = 1, 10, 152, 234, 454  # points du maillage MediaPipe

# (nom du fichier, consigne) : le texte n'a pas d'accents car cv2.putText ne les affiche pas
STEPS = [
    ("face", "Regardez la camera"),
    ("menton_leve", "Levez le menton (dessous du visage)"),
    ("tete_baissee", "Baissez la tete"),
    ("gauche", "Tournez la tete vers la gauche"),
    ("droite", "Tournez la tete vers la droite"),
]


def measure(lm):
    """Renvoie (yaw, pitch, boite) à partir des points du visage (coordonnées 0..1).

    yaw   : position du nez entre les deux joues (0.5 = de face).
    pitch : position du nez entre le front et le menton (diminue quand on lève le menton).
    """
    xa, xb = sorted((lm[CHEEK_A].x, lm[CHEEK_B].x))
    yaw = (lm[NOSE].x - xa) / max(xb - xa, 1e-6)
    pitch = (lm[NOSE].y - lm[FOREHEAD].y) / max(lm[CHIN].y - lm[FOREHEAD].y, 1e-6)
    xs, ys = [p.x for p in lm], [p.y for p in lm]
    return yaw, pitch, (min(xs), min(ys), max(xs), max(ys))


def pose_ok(name, yaw, pitch, base):
    """base = (yaw, pitch) mesurés sur la photo de face."""
    if name == "face":
        return abs(yaw - 0.5) < 0.1
    base_yaw, base_pitch = base
    return {
        "menton_leve": pitch < base_pitch - PITCH_DELTA,
        "tete_baissee": pitch > base_pitch + PITCH_DELTA,
        "gauche": yaw < base_yaw - YAW_SIDE,
        "droite": yaw > base_yaw + YAW_SIDE,
    }[name]


def framing_problem(box):
    """Renvoie un message si le visage est mal cadré, sinon None."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    height = box[3] - box[1]
    if abs(cx - 0.5) > 0.12 or abs(cy - 0.5) > 0.15:
        return "Centrez votre visage"
    if height < 0.35:
        return "Rapprochez-vous"
    if height > 0.75:
        return "Eloignez-vous"
    return None


def draw_ui(img, step, message, ready, progress):
    h, w = img.shape[:2]
    color = (0, 255, 0) if ready else (255, 255, 255)
    cv2.ellipse(img, (w // 2, h // 2), (int(w * 0.17), int(h * 0.38)), 0, 0, 360, color, 3)
    cv2.putText(img, f"Etape {step + 1}/{len(STEPS)}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(img, message, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    if ready:
        cv2.rectangle(img, (20, h - 40), (20 + int((w - 40) * progress), h - 20), (0, 255, 0), -1)


def main():
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise SystemExit("Impossible d'ouvrir la caméra.")

    session = os.path.join(OUT_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(session)

    step, base, hold_start, flash_until, last_timestamp = 0, None, None, 0.0, -1
    start = time.monotonic()
    try:
        while step < len(STEPS):
            ok, raw = cap.read()
            if not ok:
                break
            view = cv2.flip(raw, 1)  # miroir pour l'affichage ; la photo enregistrée reste l'originale
            rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
            now = time.monotonic()
            timestamp = max(int((now - start) * 1000), last_timestamp + 1)  # doit croître strictement
            last_timestamp = timestamp
            result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp)

            name, instruction = STEPS[step]
            message, ready, measures = instruction, False, None
            if not result.face_landmarks:
                message = "Aucun visage detecte"
            else:
                yaw, pitch, box = measure(result.face_landmarks[0])
                measures = (yaw, pitch)
                problem = framing_problem(box)
                if problem:
                    message = problem
                else:
                    ready = pose_ok(name, yaw, pitch, base)

            hold_start = (hold_start or now) if ready else None
            progress = (now - hold_start) / HOLD_SECONDS if hold_start else 0.0
            if now < flash_until:
                message = "Photo prise !"

            draw_ui(view, step, message, ready, min(progress, 1.0))
            cv2.imshow("Capture visage", view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if (progress >= 1.0 or key == 32) and measures:  # 32 = ESPACE
                path = os.path.join(session, f"{step + 1}_{name}.jpg")
                cv2.imwrite(path, raw)
                print("Enregistré :", path)
                if name == "face":
                    base = measures
                step, hold_start, flash_until = step + 1, None, now + 0.8
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()

    print(f"{len(os.listdir(session))}/{len(STEPS)} photos dans {session}")


if __name__ == "__main__":
    main()
