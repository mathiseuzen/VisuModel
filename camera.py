"""Affiche le flux de la caméra dans une fenêtre. Quitter avec la touche q."""

import cv2


def main():
    cap = cv2.VideoCapture(0)  # 0 = caméra par défaut
    if not cap.isOpened():
        raise SystemExit("Impossible d'ouvrir la caméra.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imshow("Caméra", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
