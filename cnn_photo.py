"""Prend UNE photo de face avec la webcam et entoure les boutons trouvés par le modèle.

Usage : python cnn_photo.py [photo.jpg] [--conf 0.7] [--live] [--cam 1]
  sans photo : la webcam s'ouvre, ESPACE prend la photo, q quitte.
  avec photo : analyse ce fichier (sans webcam).
  --cam N    : numéro de la caméra (0 = webcam du PC, 1 = souvent Camo/iPhone).
  --live     : temps réel, les boutons sont entourés sur l'image de la webcam (q quitte, + et - règlent le seuil de confiance).
Le résultat d'une photo est affiché et enregistré dans resultat.jpg.
"""

import argparse
import threading
import time

import cv2

from cnn_detect import COLORS, find_lesions, load_model
from cnn_isolate_skin import make_models


def snapshot(cam):
    cap = cv2.VideoCapture(cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise SystemExit("Caméra introuvable : essayez un autre --cam (0, 1, 2…).")
            view = frame.copy()
            cv2.putText(view, "ESPACE = photo, q = quitter", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Photo", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                return frame
            if key == ord("q"):
                raise SystemExit
    finally:
        cap.release()
        cv2.destroyAllWindows()


def draw(bgr, detections):
    thick = max(2, bgr.shape[1] // 500)
    for x0, y0, x1, y1, name, c in detections:
        cv2.rectangle(bgr, (x0, y0), (x1, y1), COLORS[name], thick)
        cv2.putText(bgr, f"{name} {c:.0%}", (x0, max(y0 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * thick, COLORS[name], max(1, thick - 1))


def live(conf, cam, stride):
    """Webcam en continu : un thread analyse la dernière image (~2 par seconde), l'affichage reste fluide."""
    model, classes = load_model("cnn_model.pt")
    landmarker, segmenter = make_models()
    cap = cv2.VideoCapture(cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    state = {"frame": None, "dets": [], "info": None, "stop": False, "conf": conf}  # dets : boîtes de la dernière analyse

    def worker():
        while not state["stop"]:
            frame = state["frame"]
            if frame is None:
                time.sleep(0.01)  # évite de tourner à vide
                continue
            state["frame"] = None  # une image à la fois ; on prend la plus récente ensuite
            result = find_lesions(frame, landmarker, segmenter, model, classes, conf=state["conf"], stride=stride)
            state["info"] = result[0] if result else None
            state["dets"] = result[1] if result else []

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise SystemExit("Caméra introuvable : essayez un autre --cam (0, 1, 2…).")
            if state["frame"] is None:
                state["frame"] = frame.copy()
            view = frame.copy()
            draw(view, state["dets"])
            info = state["info"]
            if info is None:
                msg = "aucun visage detecte"
            else:  # diagnostic : taille du visage, peau exploitable, part de zones jugees "bouton"
                msg = (f"{len(state['dets'])} bouton(s) (seuil {state['conf']:.0%}, +/-) | visage {info['face_h'] / frame.shape[0]:.0%} de l'image (>50%) | "
                       f"{info['n_windows']} zones de peau | {info['lesion_rate']:.0%} suspectes")
            cv2.putText(view, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("VisuModel temps reel", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in (ord("+"), ord("=")):
                state["conf"] = min(0.95, state["conf"] + 0.05)
            if key in (ord("-"), ord("6")):
                state["conf"] = max(0.1, state["conf"] - 0.05)
    finally:
        state["stop"] = True
        t.join()
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        segmenter.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photo", nargs="?")
    ap.add_argument("--conf", type=float, default=0.7)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--stride", type=float, default=0.75, help="pas de la fenêtre en temps réel (0.5 = précis mais lent, 1 = rapide)")
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args()
    if args.live:
        return live(args.conf, args.cam, args.stride)

    bgr = cv2.imread(args.photo) if args.photo else snapshot(args.cam)
    if bgr is None:
        raise SystemExit(f"Photo illisible : {args.photo}")

    model, classes = load_model("cnn_model.pt")
    landmarker, segmenter = make_models()
    try:
        result = find_lesions(bgr, landmarker, segmenter, model, classes, conf=args.conf)
    finally:
        landmarker.close()
        segmenter.close()
    if result is None:
        raise SystemExit("Aucun visage détecté : mettez-vous de face, bien éclairé.")
    info, detections = result

    draw(bgr, detections)
    cv2.imwrite("resultat.jpg", bgr)
    print(f"{info['n_windows']} zones de peau analysées, {len(detections)} bouton(s) entouré(s) -> resultat.jpg")

    cv2.imshow("Resultat (une touche pour fermer)", bgr)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
