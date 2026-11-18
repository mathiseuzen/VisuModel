"""Fusionne les 6 classes du jeu de données YOLO en une seule : « imperfection ».

Usage : python yolo_merge_classes.py [--src archive] [--out datasets/imperfections]
Écrit le jeu de données fusionné + un .zip (à envoyer sur Kaggle : New Dataset).
"""

import argparse
import glob
import os
import shutil

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="archive")
ap.add_argument("--out", default="datasets/imperfections")
args = ap.parse_args()

shutil.rmtree(args.out, ignore_errors=True)
for split in ("train", "valid", "test"):
    os.makedirs(f"{args.out}/{split}/images")
    os.makedirs(f"{args.out}/{split}/labels")
    for img in glob.glob(f"{args.src}/{split}/images/*"):
        shutil.copy(img, f"{args.out}/{split}/images")
    n = 0
    for path in glob.glob(f"{args.src}/{split}/labels/*.txt"):
        with open(path) as f:  # on garde la boîte (4 nombres), la classe devient 0
            lines = ["0 " + " ".join(l.split()[1:5]) for l in f if l.strip()]
        n += len(lines)
        with open(f"{args.out}/{split}/labels/{os.path.basename(path)}", "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
    print(f"{split}: {len(os.listdir(f'{args.out}/{split}/images'))} images, {n} boîtes")

with open(f"{args.out}/data.yaml", "w") as f:
    f.write("path: .\ntrain: train/images\nval: valid/images\ntest: test/images\nnames:\n  0: imperfection\n")
shutil.make_archive(args.out, "zip", args.out)
print("zip :", args.out + ".zip")
