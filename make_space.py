"""Prépare le dossier space_build/ à envoyer sur Hugging Face (Space Gradio).

Usage : python make_space.py
Le dossier contient tout ce qu'il faut : appli, modules, modèles, README (avec l'en-tête du Space),
requirements.txt et packages.txt. Il est exclu de Git car il contient les poids du modèle.
"""

import os
import shutil

import gradio
import mediapipe

OUT = "space_build"
FILES = ["app.py", "detect.py", "isolate_skin.py", "prepare_patches.py", "train_patch_cnn.py",
         "model_patches.pt", "face_landmarker.task", "selfie_multiclass.tflite"]

README = f"""---
title: VisuModel
emoji: 🔍
colorFrom: pink
colorTo: red
sdk: gradio
sdk_version: {gradio.__version__}
python_version: "3.12"
app_file: app.py
pinned: false
---

# VisuModel

Prenez un selfie : un réseau de neurones entraîné de zéro repère les boutons et leur type probable
(point noir, point blanc, papule, pustule, nodule, tache).

Outil expérimental : il ne pose aucun diagnostic et ne remplace pas l'avis d'un médecin.
"""

REQUIREMENTS = f"""--extra-index-url https://download.pytorch.org/whl/cpu
torch
torchvision
mediapipe=={mediapipe.__version__}
numpy
"""

PACKAGES = "libgl1\nlibglib2.0-0\n"  # nécessaires à OpenCV sur le serveur


def main():
    missing = [f for f in FILES if not os.path.exists(f)]
    if missing:
        raise SystemExit(f"Fichiers manquants : {', '.join(missing)}")
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT)
    for f in FILES:
        shutil.copy(f, OUT)
    for name, content in (("README.md", README), ("requirements.txt", REQUIREMENTS), ("packages.txt", PACKAGES)):
        with open(os.path.join(OUT, name), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(content)
    size = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT)) / 1e6
    print(f"{OUT}/ prêt ({len(os.listdir(OUT))} fichiers, {size:.0f} Mo) : {', '.join(sorted(os.listdir(OUT)))}")


if __name__ == "__main__":
    main()
