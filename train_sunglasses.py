import tensorflow as tf
from tensorflow.keras import layers, models

args = dict(directory="dataset/sunglasses_data", image_size=(96, 96), batch_size=32,
            validation_split=0.2, seed=42, label_mode="binary")
train = tf.keras.utils.image_dataset_from_directory(subset="training", **args)
val = tf.keras.utils.image_dataset_from_directory(subset="validation", **args)
print(train.class_names)   # ['no_sunglasses', 'sunglasses'] -> 0 = no, 1 = sunglasses
model = models.Sequential([
    layers.Input((96, 96, 3)),
    layers.RandomFlip("horizontal"), layers.RandomBrightness(0.2), layers.RandomZoom(0.1),
    layers.Rescaling(1 / 255),
    layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation="relu"), layers.MaxPooling2D(),
    layers.GlobalAveragePooling2D(), layers.Dropout(0.4),
    layers.Dense(1, activation="sigmoid"),
])
model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
model.fit(train, validation_data=val, epochs=12,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])
model.save("sunglasses_cnn.keras")
