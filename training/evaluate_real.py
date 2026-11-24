"""
evaluate_real.py  --  put this in  Driver-Drowsiness-System/training/
Run from the project root:   python training/evaluate_real.py

Replaces the simulated numbers in evaluate_models.py with REAL measurements.
It runs your actual modules (FaceDetector, EyeDetector, YawnDetector,
SunglassesDetector) on labelled FULL-FACE images, then:
  1. splits the images 50% tune / 50% test (fixed seed)
  2. finds the best threshold on the tune half
  3. reports accuracy / precision / recall / F1 / ROC-AUC on the untouched test half
  4. saves CSVs + plots in reports/

Expected folders (full face images, not cropped eyes):
  dataset/eval/eye/open/        dataset/eval/eye/closed/
  dataset/eval/yawn/no_yawn/    dataset/eval/yawn/yawn/
  dataset/eval/glasses/clear/   dataset/eval/glasses/sunglasses/
(any task whose folders are missing is skipped)
"""
import os, sys, glob, itertools, json
import numpy as np, pandas as pd, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from modules.face_detector import FaceDetector
from modules.eye_detector import EyeDetector
from modules.yawn_detector import YawnDetector
from modules.sunglasses_detector import SunglassesDetector

EVAL = os.path.join(ROOT, "dataset", "eval")
OUT = os.path.join(ROOT, "reports"); os.makedirs(OUT, exist_ok=True)
EXT = ("*.jpg", "*.jpeg", "*.png", "*.bmp")

# task -> (negative folder, positive folder)
TASKS = {
    "eye":   ("eye/open", "eye/closed"),
    "yawn":  ("yawn/no_yawn", "yawn/yawn"),
    "glass": ("glasses/clear", "glasses/sunglasses"),
}


def list_images(folder):
    p = os.path.join(EVAL, folder)
    files = []
    for e in EXT:
        files += glob.glob(os.path.join(p, e))
    return sorted(files)


def extract_features(task_files):
    """Run the real detectors once per image; return a DataFrame of raw features."""
    fd = FaceDetector(max_faces=1)
    eye, yawn = EyeDetector(), YawnDetector()
    rows, skipped = [], 0
    for label, files in task_files:
        for f in files:
            img = cv2.imread(f)
            if img is None:
                continue
            r = fd.detect(img)
            if not r["face_found"]:
                skipped += 1
                continue
            lm = r["landmarks"]
            sg = SunglassesDetector(confirm_frames=1, release_frames=1)  # single image: no debounce
            s = sg.detect(img, lm)
            rows.append(dict(file=f, label=label,
                             ear=eye.detect(img, lm)["ear"],
                             mar=yawn.detect(img, lm)["mar"],
                             dark=s["darkness_ratio"], tex=s["eye_texture"]))
    print(f"   face not found in {skipped} images (excluded)")
    return pd.DataFrame(rows)


def metrics(y, p, score):
    return dict(accuracy=accuracy_score(y, p), precision=precision_score(y, p, zero_division=0),
                recall=recall_score(y, p, zero_division=0), f1=f1_score(y, p, zero_division=0),
                auc=roc_auc_score(y, score) if len(set(y)) > 1 else float("nan"))


def best_threshold(y, x, positive_if_below):
    """Sweep thresholds on the tune set, pick the one with the best F1."""
    best = (0, None)
    for t in np.unique(np.quantile(x, np.linspace(0.01, 0.99, 197))):
        p = (x < t) if positive_if_below else (x > t)
        f1 = f1_score(y, p, zero_division=0)
        if f1 > best[0]:
            best = (f1, float(t))
    return best[1]


def report(name, y, p, score, extra=""):
    m = metrics(y, p, score)
    print(f"\n=== {name} (TEST set, n={len(y)}) {extra}")
    print("   " + "  ".join(f"{k}={v:.3f}" for k, v in m.items()))
    cm = confusion_matrix(y, p)
    plt.figure(figsize=(4, 3.4))
    plt.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, v, ha="center", va="center")
    plt.xticks([0, 1], ["neg", "pos"]); plt.yticks([0, 1], ["neg", "pos"])
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(name)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, f"cm_real_{name}.png"), dpi=120); plt.close()
    fpr, tpr, _ = roc_curve(y, score)
    return m, (fpr, tpr)


def main():
    rocs, summary = {}, {}
    for task, (neg, pos) in TASKS.items():
        nf, pf = list_images(neg), list_images(pos)
        if not nf or not pf:
            print(f"[skip] {task}: need images in dataset/eval/{neg} and /{pos}")
            continue
        print(f"\n[{task}] extracting features from {len(nf)} neg + {len(pf)} pos images ...")
        df = extract_features([(0, nf), (1, pf)])
        df.to_csv(os.path.join(OUT, f"features_{task}.csv"), index=False)

        rng = np.random.RandomState(42)
        mask = rng.rand(len(df)) < 0.5
        tune, test = df[mask], df[~mask]

        if task == "eye":      # closed => low EAR
            t = best_threshold(tune.label, tune.ear, True)
            m, roc = report("eye_EAR", test.label, test.ear < t, -test.ear, f"best EAR_THRESHOLD={t:.3f}")
            summary["eye_EAR"] = dict(threshold=t, **m); rocs["EAR"] = roc
        elif task == "yawn":   # yawn => high MAR
            t = best_threshold(tune.label, tune.mar, False)
            m, roc = report("yawn_MAR", test.label, test.mar > t, test.mar, f"best MAR_THRESHOLD={t:.3f}")
            summary["yawn_MAR"] = dict(threshold=t, **m); rocs["MAR"] = roc
        else:                  # sunglasses => dark AND smooth; grid-search both thresholds
            best = (0, None)
            for d, x in itertools.product(np.linspace(0.3, 1.0, 36), np.linspace(2, 80, 40)):
                f = f1_score(tune.label, (tune.dark < d) & (tune.tex < x), zero_division=0)
                if f > best[0]:
                    best = (f, (float(d), float(x)))
            d, x = best[1]
            pred = (test.dark < d) & (test.tex < x)
            score = -(test.dark / max(test.dark.max(), 1e-6)) - (test.tex / max(test.tex.max(), 1e-6))
            m, roc = report("sunglasses", test.label, pred, score,
                            f"best SUNGLASSES_DARKNESS_RATIO={d:.2f}, SUNGLASSES_TEXTURE_THRESHOLD={x:.1f}")
            summary["sunglasses"] = dict(darkness=d, texture=x, **m); rocs["Sunglasses"] = roc

    if rocs:
        plt.figure(figsize=(5.5, 4.5))
        for k, (fpr, tpr) in rocs.items():
            plt.plot(fpr, tpr, label=k)
        plt.plot([0, 1], [0, 1], "k--"); plt.legend(); plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.title("ROC (real test data)"); plt.tight_layout()
        plt.savefig(os.path.join(OUT, "roc_real.png"), dpi=120); plt.close()
    with open(os.path.join(OUT, "evaluation_real.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved CSVs, confusion matrices, ROC and evaluation_real.json in reports/")
    print("Copy the tuned thresholds into CONFIG in app.py and quote THESE numbers in your report.")


if __name__ == "__main__":
    main()
