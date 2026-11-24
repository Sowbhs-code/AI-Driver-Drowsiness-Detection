import random, pathlib
from PIL import Image

root = pathlib.Path("dataset/glasses_noGlasses/train")
random.seed(1)
sheet = Image.new("RGB", (6 * 160, 2 * 160), "white")
for row, cls in enumerate(["with_glasses", "no_glass"]):
    files = list(( root / cls).glob("*"))
    print(cls, len(files), "images")
    for col, f in enumerate(random.sample(files, 6)):
        sheet.paste(Image.open(f).convert("RGB").resize((160, 160)), (col * 160, row * 160))
sheet.save("glasses_peek.png")
print("saved glasses_peek.png")