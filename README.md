# VisuModel

Un modèle qui détecte les imperfections du visage (boutons, rougeurs, etc.) à partir de photos.

## Étapes du projet

1. **Capture guidée** (fait) : la caméra photographie le visage sous plusieurs angles pour constituer un panel.
2. **Isoler le visage** : recadrer sur le visage seul.
3. **Détecter les imperfections** : un réseau de neurones créé par nos soins, entraîné sur nos propres images.

## Étape 1 : la capture

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python capture.py
```

1. Placez votre visage dans l'ovale (il passe au vert quand tout est bon).
2. L'écran indique la pose à prendre : face, menton levé, tête baissée, côté gauche, côté droit.
3. Tenez la pose 1 seconde : la photo part toute seule.

Touches : `ESPACE` prend la photo tout de suite, `q` quitte.

Les photos sont enregistrées **sans aucun dessin dessus** dans `captures/<date_heure>/`, en pleine résolution (jusqu'à 1280×720).

## Vie privée

Le dossier `captures/` est exclu de Git (`.gitignore`) : vos photos ne sont jamais envoyées sur GitHub.

## Fichiers

| Fichier | Rôle |
|---|---|
| `capture.py` | Capture guidée du visage |
| `face_landmarker.task` | Modèle MediaPipe de détection du visage, utilisé pour centrer le visage et reconnaître la pose |
| `test_capture.py` | Tests de la logique de cadrage et de pose (`python -m unittest test_capture -v`) |
