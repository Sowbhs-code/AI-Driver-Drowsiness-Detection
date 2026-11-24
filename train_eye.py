import pathlib, random, numpy as np, cv2
import tensorflow as tf
from tensorflow.keras import layers, models

SIZE = 64
random.seed(42); np.random.seed(42); tf.random.set_seed(42)

def load(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (SIZE, SIZE))
    return cv2.equalizeHist(img)          # the app must do the same before predicting

cache = pathlib.Path("cache_eye_v2.npz")
if cache.exists():
    d = np.load(cache)
    X, y, subjects, Xw, yw = d["X"], d["y"], d["subjects"], d["Xw"], d["yw"]
else:
    root = pathlib.Path("dataset/eye_dataset")
    imgs, labels, subjects = [], [], []
    for label, name in enumerate(["closed", "open"]):
        for p in (root / name).glob("*.png"):
            imgs.append(load(p)); labels.append(label); subjects.append(p.name.split("_")[0])
    X = np.stack(imgs)[..., None]; y = np.array(labels, dtype="float32")
    subjects = np.array(subjects)
    wimgs, wl = [], []
    for label, name in enumerate(["closed", "open"]):
        for p in (pathlib.Path("dataset/webcam_eye_test") / name).glob("*"):
            wimgs.append(load(p)); wl.append(label)
    Xw = np.stack(wimgs)[..., None]; yw = np.array(wl, dtype="float32")
    np.savez_compressed(cache, X=X, y=y, subjects=subjects, Xw=Xw, yw=yw)

# split MRL by person
people = sorted(set(subjects.tolist())); random.shuffle(people)
test_p, val_p = set(people[:5]), set(people[5:10])
is_test = np.isin(subjects, list(test_p)); is_val = np.isin(subjects, list(val_p))
is_train = ~(is_test | is_val)

# split webcam-style images in half: one half trains, one half tests
perm = np.random.permutation(len(Xw)); half = len(Xw) // 2
w_train, w_test = perm[:half], perm[half:]

Xtr = np.concatenate([X[is_train], Xw[w_train]]); ytr = np.concatenate([y[is_train], yw[w_train]])
print("train:", len(Xtr), "val:", is_val.sum(), "MRL test:", is_test.sum(), "webcam test:", len(w_test))

def block(f):
    return [layers.Conv2D(f, 3, padding="same"), layers.BatchNormalization(),
            layers.Activation("relu"), layers.MaxPooling2D()]

model = models.Sequential([
    layers.Input((SIZE, SIZE, 1)),
    layers.RandomBrightness(0.2), layers.RandomContrast(0.2),
    layers.RandomRotation(0.05), layers.RandomZoom(0.1),
    layers.Rescaling(1 / 255),
    *block(32), *block(64), *block(128),
    layers.GlobalAveragePooling2D(), layers.Dropout(0.4),
    layers.Dense(1, activation="sigmoid"),
])
model.compile(tf.keras.optimizers.Adam(3e-4), "binary_crossentropy", metrics=["accuracy"])
model.fit(Xtr, ytr, validation_data=(X[is_val], y[is_val]), epochs=30, batch_size=64,
          callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6,
                                                      restore_best_weights=True)])
model.save("eye_cnn_v2.keras")

print("\nMRL test (unseen people):", model.evaluate(X[is_test], y[is_test], verbose=0)[1])
print("Webcam-style test (held-out half):", model.evaluate(Xw[w_test], yw[w_test], verbose=0)[1])