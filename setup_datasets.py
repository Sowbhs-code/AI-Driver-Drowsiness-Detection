import shutil, pathlib, random

random.seed(42)
D = pathlib.Path("dataset")

def copy_files(files, dst):
    dst.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy(f, dst / f.name)
    print(f"{dst}: {len(files)} images")

# 1. Yawn evaluation set (used only to test the MAR threshold)
for cls in ["yawn", "no_yawn"]:
    copy_files(list((D / "train" / cls).glob("*")), D / "yawn_eval" / cls)

# 2. Webcam-style eye test set (never used for training)
for src, name in [("Open", "open"), ("Closed", "closed")]:
    copy_files(list((D / "train" / src).glob("*")), D / "webcam_eye_test" / name)

# 3. Sunglasses classifier, balanced between the two classes
g = D / "glasses_noGlasses" / "train"
yes = list((g / "with_glasses").glob("*"))
no = list((g / "no_glass").glob("*"))
n = min(len(yes), len(no))
copy_files(random.sample(yes, n), D / "sunglasses_data" / "sunglasses")
copy_files(random.sample(no, n), D / "sunglasses_data" / "no_sunglasses")