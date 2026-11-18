# État du projet (18/11/2026)

## Où on en est

**Modèle actuel : `yolo_model.pt`** (YOLO11n, une seule classe `imperfection`, entraîné à 1024 px sur Kaggle).

| Modèle | Classes | Taille d'entrée | mAP50 test | Précision | Rappel |
|---|---|---|---|---|---|
| `yolo_model_6classes.pt` | 6 types de boutons | 640 | 0,16 | 0,25 | 0,22 |
| **`yolo_model.pt`** | **imperfection** | **1024** | **0,32** | **0,38** | **0,35** |

Mesuré sur le jeu de test (92 images, 923 boîtes), jamais vu à l'entraînement.
Le CNN fait maison (`cnn_model.pt`) existe toujours, mais n'est plus la piste principale.

**Dataset** : `archive/` (6 classes, 640 / 197 / 92 images en train / valid / test) fusionné en une classe par
`yolo_merge_classes.py` -> `datasets/imperfections/` + `.zip` pour Kaggle. Origine des données non confirmée.

**Outils**
- `python yolo_photo.py --live --cam 1` : temps réel (`+` / `-` seuil, `q` quitte). Sans `--live` : photo (ESPACE).
- `python yolo_wireframe.py photo.jpg --conf 0.25` : rendu anonyme (maillage MediaPipe + détections) dans `output/`.
- Entraînement : `yolo_train_kaggle.ipynb` (GPU T4, 50 epochs, batch 8). Télécharger `yolo_model.pt` depuis l'Output sans le dézipper.

## Constats

- Le modèle détecte bien, mais ses scores restent bas : il faut un seuil faible (~15-25 %) en live.
- À 5 % de seuil : beaucoup de bruit (boîtes superposées, fausses alertes sur la peau normale).
- La courbe de validation montait encore à l'epoch 30 sur 50 : le modèle n'avait peut-être pas fini d'apprendre.
- Causes probables des scores bas : peu de données (640 images), objets très petits, décalage entre le dataset et la webcam.

## À faire la prochaine fois

1. **Relancer l'entraînement plus longtemps** (`epochs=100`, `patience=30`) et voir si le mAP50 dépasse 0,32.
2. **Tester `yolo11s`** à la place de `yolo11n` (plus précis, environ 2,5 fois plus lent en temps réel).
3. **Plus de données** : chercher un autre dataset (Kaggle / Roboflow) à fusionner avec `yolo_merge_classes.py`.
4. **Annoter des photos webcam** : c'est le levier le plus efficace contre le décalage de domaine.
5. **Choisir un seuil par défaut** dans `yolo_photo.py` (0,25 aujourd'hui) après des essais en live avec la webcam.
6. Si on veut retrouver les types de boutons : repartir de `archive/` (6 classes) une fois le dataset agrandi.
7. Décider si on versionne `yolo_model.pt` (5 Mo, ignoré par `.gitignore` via `*.pt`) pour que le dépôt soit utilisable après un clone.
8. Nettoyer : supprimer `yolo_model_640.pt` et `yolo_model_6classes.pt` si on n'en a plus besoin.

## Limites à garder en tête

- Outil expérimental : aucun diagnostic, ne remplace pas l'avis d'un médecin.
- Le détecteur de visage MediaPipe ne trouve pas de visage sur les gros plans du dataset ; `yolo_wireframe.py` et
  `yolo_photo.py` ont besoin d'un visage entier.
