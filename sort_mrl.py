import shutil, pathlib, random

PER_CLASS = 12000
random.seed(42)

src = pathlib.Path("dataset/mrleyedataset")
folders = {"closed": src / "Close-Eyes", "open": src / "Open-Eyes"}
exts = {".png", ".jpg", ".jpeg"}

files = {}
for name, folder in folders.items():
    files[name] = [p for p in folder.rglob("*") if p.suffix.lower() in exts]
    print(name, "available:", len(files[name]))

n = min(PER_CLASS, len(files["open"]), len(files["closed"]))
for name, lst in files.items():
    random.shuffle(lst)
    dst = pathlib.Path("dataset/eye_dataset") / name
    dst.mkdir(parents=True, exist_ok=True)
    for p in lst[:n]:
        shutil.copy(p, dst / p.name)
    print(name, "copied:", n)
    print("  example file names:", [p.name for p in lst[:3]])