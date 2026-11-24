import os
os.environ["KERAS_HOME"] = "D:/keras_home"   # keep the pretrained weights off the nearly full C: drive

import tensorflow as tf
from tensorflow.keras import layers, models

SIZE = 128
args = dict(directory="dataset/yawn_eval", image_size=(SIZE, SIZE), batch_size=32,
            validation_split=0.2, seed=42, label_mode="binary")
train = tf.keras.utils.image_dataset_from_directory(subset="training", **args)
val = tf.keras.utils.image_dataset_from_directory(subset="validation", **args)
print(train.class_names)   # ['no_yawn', 'yawn'] -> 0 = no yawn, 1 = yawn

base = tf.keras.applications.MobileNetV2(input_shape=(SIZE, SIZE, 3),
                                         include_top=False, weights="imagenet")
base.trainable = False

inputs = layers.Input((SIZE, SIZE, 3))
x = layers.RandomFlip("horizontal")(inputs)
x = layers.RandomBrightness(0.2)(x)
x = layers.RandomZoom(0.1)(x)
x = layers.Rescaling(1 / 127.5, offset=-1)(x)      # MobileNetV2 expects values in -1..1
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = models.Model(inputs, outputs)

# Phase 1: train only the new top layer
model.compile(tf.keras.optimizers.Adam(1e-3), "binary_crossentropy", metrics=["accuracy"])
model.fit(train, validation_data=val, epochs=8)

# Phase 2: gently fine-tune the last part of the pretrained network
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False
model.compile(tf.keras.optimizers.Adam(1e-5), "binary_crossentropy", metrics=["accuracy"])
model.fit(train, validation_data=val, epochs=12,
          callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4,
                                                      restore_best_weights=True)])
model.save("yawn_cnn_v2.keras")
print("final validation accuracy:", model.evaluate(val, verbose=0)[1])