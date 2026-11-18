"""Appli Gradio (Hugging Face Space) : prenez un selfie, les boutons sont entourés.

Lancer en local : python app.py
"""

import threading

import cv2
import gradio as gr
import numpy as np

from detect import COLORS, find_lesions, load_model
from isolate_skin import make_models

MAX_SIDE = 2000          # les photos de téléphone sont réduites à 2000 px max (rapidité sur un serveur gratuit)
MIN_FACE_SHARE = 0.5     # le visage doit occuper au moins la moitié de la hauteur de la photo
NAMES = {                # texte sans accents : cv2.putText ne les affiche pas
    "blackheads": "point noir", "dark_spot": "tache", "nodules": "nodule",
    "papules": "papule", "pustules": "pustule", "whiteheads": "point blanc",
}
DISCLAIMER = ("Outil expérimental : il ne pose aucun diagnostic et ne remplace pas l'avis d'un médecin ou d'un "
              "dermatologue.")

MODEL, CLASSES = load_model("model_patches.pt")
LANDMARKER, SEGMENTER = make_models()
LOCK = threading.Lock()  # les modèles MediaPipe ne sont partagés qu'une requête à la fois


def analyze(photo, conf):
    if photo is None:
        return None, "Prenez ou envoyez une photo de votre visage."
    rgb = photo[..., :3]  # ignore un éventuel canal alpha
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    scale = min(1.0, MAX_SIDE / max(bgr.shape[:2]))
    if scale < 1:
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    with LOCK:
        result = find_lesions(bgr, LANDMARKER, SEGMENTER, MODEL, CLASSES, conf=conf)
    if result is None:
        return rgb, "Aucun visage détecté. Placez votre visage de face, bien éclairé, et recommencez."
    info, detections = result

    notes = []
    if info["face_h"] / bgr.shape[0] < MIN_FACE_SHARE:
        notes.append("⚠️ Le visage n'occupe pas assez la photo : rapprochez-vous pour qu'il remplisse l'écran, "
                     "sinon les petits boutons ne sont pas visibles.")
    if info["n_windows"] == 0:
        notes.append("Pas assez de peau dégagée pour analyser (cheveux, lunettes, ombres ?).")

    thick = max(2, bgr.shape[1] // 500)
    for x0, y0, x1, y1, name, c in detections:
        cv2.rectangle(bgr, (x0, y0), (x1, y1), COLORS[name], thick)
        cv2.putText(bgr, f"{NAMES[name]} {c:.0%}", (x0, max(y0 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4 * thick, COLORS[name], max(1, thick - 1))

    if detections:
        counts = {}
        for *_, name, _ in detections:
            counts[NAMES[name]] = counts.get(NAMES[name], 0) + 1
        notes.insert(0, f"**{len(detections)} bouton(s) repéré(s)** : " + ", ".join(f"{n} × {k}" for k, n in counts.items()))
    elif info["n_windows"]:
        notes.insert(0, "Aucun bouton repéré avec ce seuil de confiance (vous pouvez le baisser).")
    notes.append(DISCLAIMER)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), "\n\n".join(notes)


with gr.Blocks(title="VisuModel", delete_cache=(600, 600)) as demo:  # photos effacées du serveur après 10 min
    gr.Markdown(
        "# VisuModel : repérer les imperfections du visage\n"
        "1. Prenez un **selfie de face, bien éclairé, le visage qui remplit l'écran** (ou envoyez une photo).\n"
        "2. Cliquez sur **Analyser** : les boutons sont entourés avec leur type probable.\n\n"
        "Votre photo est analysée sur le serveur et supprimée automatiquement au bout de quelques minutes."
    )
    with gr.Row():
        with gr.Column():
            photo = gr.Image(sources=["webcam", "upload"], type="numpy", label="Votre photo")
            conf = gr.Slider(0.3, 0.95, value=0.7, step=0.05, label="Seuil de confiance",
                             info="Plus il est haut, moins il y a de fausses alertes (mais plus de boutons manqués).")
            go = gr.Button("Analyser", variant="primary")
        with gr.Column():
            out = gr.Image(label="Résultat")
            text = gr.Markdown()
    go.click(analyze, [photo, conf], [out, text])
    with gr.Accordion("Fiabilité", open=False):
        gr.Markdown(
            "Mesures sur le jeu de test (photos de visages avec boutons étiquetés) :\n"
            "- bouton ou peau saine : **88 %** des boutons trouvés, **91 %** de précision ;\n"
            "- type exact (point noir, papule, pustule…) : **F1 moyen 0,49** — les types voisins sont souvent "
            "confondus et les nodules et points noirs sont mal reconnus ;\n"
            "- quand la confiance dépasse 90 %, le résultat est juste dans 94 % des cas.\n\n"
            "Ces chiffres viennent de photos haute résolution ; sur une photo de téléphone, les résultats peuvent différer."
        )

if __name__ == "__main__":
    demo.launch()
