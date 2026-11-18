# VisuModel

Un modèle qui détecte les imperfections du visage (boutons, rougeurs, etc.) à partir de photos.

## Étapes du projet

1. **Capture guidée** (fait) : la caméra photographie le visage sous plusieurs angles pour constituer un panel.
2. **Isoler la peau** (fait) : ne garder que la peau, sans fond, cheveux, sourcils, yeux ni lèvres.
3. **Découper en zones** (fait) : front, nez, joues, menton, en rectangles 100 % remplis de peau, en RGB simple (sans alpha).
4. **Préparer les imagettes** (fait) : petits carrés 64×64 de boutons et de peau saine, découpés dans un jeu de données YOLO.
5. **Entraîner le classifieur** (en cours, sur Kaggle) : un CNN écrit de zéro qui reconnaît 6 types de boutons + peau saine.
6. **Détecter sur vos photos** : une fenêtre glissante parcourt les zones de peau et le classifieur dit ce qu'il voit.

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

## Étape 2 : isoler la peau

```bash
python isolate_skin.py                      # dernière capture
python isolate_skin.py captures/<session>   # ou une capture précise
```

Pour chaque photo, écrit `captures/<session>/peau/<nom>.png` : l'image recadrée sur la peau, où tout le reste est transparent **et vidé** (pixels à 0).

Le masque de peau est l'intersection de deux sources :
- l'ovale du visage (points MediaPipe, prolongé de 30 % vers le haut car le maillage s'arrête au milieu du front), moins les yeux, les sourcils et les lèvres ;
- la classe « peau du visage » d'un modèle de segmentation (`selfie_multiclass.tflite`), qui retire cheveux, barbe et fond.

Le masque est ensuite érodé de ~2 % de la largeur du visage pour ne garder aucun pixel de bordure.

## Étape 3 : découper en zones

```bash
python zones.py                      # dernière capture
python zones.py captures/<session>   # ou une capture précise
```

Zones : `front`, `nez`, `joue_gauche`, `joue_droite`, `menton` (gauche/droite = côté de la personne : sur la photo, la joue gauche est à droite de l'image).

Pour chaque zone, le script cherche le **plus grand rectangle sans aucun pixel hors peau** (ni yeux, sourcils, lèvres, cheveux, fond). Un contrôle interne vérifie qu'il n'y a aucun pixel vide dans chaque rectangle.

Résultat dans `captures/<session>/zones/` :
- `<photo>/<zone>.png` : image RGB ordinaire, sans canal alpha, de taille variable selon la zone ;
- `<photo>_apercu.jpg` : la photo avec les rectangles dessinés, pour vérifier le placement.

Une zone dont un côté fait moins de 32 px est ignorée (cas courant du côté caché sur les poses de profil).

## Étape 4 : préparer les imagettes

Source : un jeu de données au format YOLO (dossiers `train/`, `valid/`, `test/` avec `images/` et `labels/`, plus `data.yaml`) placé dans `archive/` (exclu de Git).

```bash
python prepare_patches.py                  # écrit datasets/patches/<split>/<classe>/*.png
```

- Chaque boîte devient une imagette 64×64 : un carré centré sur la boîte, de côté 2× la boîte, pour garder du contexte.
- Autant d'imagettes de **peau saine** sont tirées au hasard, uniquement sur de la peau (modèle de segmentation) et sans toucher aucune boîte.
- Le découpage train / valid / test du jeu de données est conservé tel quel.
- Les boîtes de moins de 6 px sont ignorées.

## Étape 5 : entraîner le classifieur

`train_patch_cnn.py` entraîne un CNN de 288 000 paramètres, écrit de zéro avec PyTorch. Il pondère les classes rares, applique des retournements, rotations de 90° et variations de luminosité, choisit le meilleur modèle sur la validation puis le mesure une seule fois sur le test (précision, rappel, F1 par classe et matrice de confusion).

Sur un processeur, une époque prend environ 150 s : on entraîne donc sur Kaggle (GPU gratuit).

1. Zippez `datasets/patches` et envoyez l'archive sur Kaggle (**Datasets → New Dataset**), en privé.
2. Importez `kaggle_train.ipynb` (**File → Import Notebook**), ajoutez ce dataset en entrée, activez le **GPU** et **Internet**.
3. **Run All** : le modèle `model_patches.pt` est dans l'onglet **Output**.

En local : `python train_patch_cnn.py --epochs 30`.

## Vie privée

Le dossier `captures/` est exclu de Git (`.gitignore`) : vos photos ne sont jamais envoyées sur GitHub.

## Fichiers

| Fichier | Rôle |
|---|---|
| `capture.py` | Capture guidée du visage |
| `face_landmarker.task` | Modèle MediaPipe de détection du visage, utilisé pour centrer le visage et reconnaître la pose |
| `test_capture.py` | Tests de la logique de cadrage et de pose (`python -m unittest test_capture -v`) |
| `isolate_skin.py` | Isolation de la peau sur chaque photo |
| `selfie_multiclass.tflite` | Modèle MediaPipe de segmentation (fond, cheveux, peau, vêtements) |
| `test_isolate_skin.py` | Tests du recadrage et du vidage des pixels hors peau |
| `zones.py` | Découpe de la peau en rectangles pleins par zone du visage |
| `test_zones.py` | Tests de la recherche du plus grand rectangle plein (comparé à une recherche exhaustive) |
| `prepare_patches.py` | Découpe des imagettes (boutons + peau saine) d'un jeu de données YOLO |
| `train_patch_cnn.py` | CNN de zéro : entraînement, évaluation, enregistrement du modèle |
| `kaggle_train.ipynb` | Notebook prêt à importer sur Kaggle pour entraîner sur GPU |
| `test_prepare_patches.py`, `test_train_patch_cnn.py` | Tests de ces deux scripts |
