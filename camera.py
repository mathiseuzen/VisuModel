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
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            cv2.imshow("Caméra", blurred)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
