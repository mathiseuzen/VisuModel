# VisuModel

Détecte les boutons du visage avec une caméra (webcam du PC ou iPhone via Camo Studio). Deux modèles au choix.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Utilisation

```bash
python cnn_photo.py --live --cam 1     # réseau de neurones maison (cnn_model.pt), temps réel
python yolo_photo.py --live --cam 1    # modèle YOLO (yolo_model.pt), temps réel
```

Les deux scripts ont les mêmes options :
- `--cam N` : numéro de la caméra (0 = webcam du PC, 1 = souvent Camo / iPhone).
- `--live` : temps réel ; `+` / `-` règlent le seuil de confiance, `q` quitte.
- sans `--live` : ESPACE prend une photo ; `python cnn_photo.py image.jpg` analyse un fichier. Résultat dans `resultat.jpg`.

## Nommage des fichiers

Préfixe = modèle : `cnn_*` (réseau maison), `yolo_*` (YOLO). Sans préfixe : ressources partagées.

| Fichier | Rôle |
|---|---|
| `cnn_photo.py` | Photo ou temps réel avec le CNN |
| `cnn_detect.py` | Fenêtre glissante + classifieur : repère les boutons sur une image |
| `cnn_isolate_skin.py` | Isolation de la peau (MediaPipe) |
| `cnn_prepare_patches.py` | Découpe des imagettes du jeu de données YOLO |
| `cnn_train.py` | Entraînement et évaluation du `PatchCNN` |
| `cnn_model.pt` | Poids du CNN (6 types de boutons + peau saine) |
| `yolo_photo.py` | Photo ou temps réel avec YOLO |
| `yolo_train_kaggle.ipynb` | Entraînement YOLO sur Kaggle ; télécharger `yolo_model.pt` depuis l'Output |
| `yolo_model.pt` | Poids YOLO |
| `face_landmarker.task`, `selfie_multiclass.tflite` | Modèles MediaPipe (visage, segmentation de la peau) |
| `archive/` | Jeu de données de boutons au format YOLO (train / valid / test) |

## Avertissement

Outil expérimental : il ne pose aucun diagnostic et ne remplace pas l'avis d'un médecin. L'origine des données d'entraînement n'est pas confirmée.
