"""
Trains a small DS-CNN keyword-spotting model, evaluates it, then
exports .h5, INT8-quantized .tflite, and a .cc C-array for the ESP32.
Run AFTER extract_features.py.
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

FEATURES_DIR = "dataset/features"
MODELS_DIR = "models"

def load_split(name):
    X = np.load(os.path.join(FEATURES_DIR, f"X_{name}.npy"))
    y = np.load(os.path.join(FEATURES_DIR, f"y_{name}.npy"))
    return X, y

def build_ds_cnn(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, (10, 4), strides=(2, 2), padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    for _ in range(3):
        x = layers.DepthwiseConv2D((3, 3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(64, (1, 1), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(2, activation="softmax")(x)
    return models.Model(inputs, outputs)

def representative_dataset_gen(X_train, num_samples=100):
    for i in range(min(num_samples, len(X_train))):
        yield [X_train[i:i+1].astype(np.float32)]

def convert_tflite_to_c_array(tflite_path, c_path, var_name="spectra_model"):
    with open(tflite_path, "rb") as f:
        data = f.read()
    with open(c_path, "w") as f:
        f.write(f"const unsigned char {var_name}[] = {{\n")
        for i, b in enumerate(data):
            f.write(f"0x{b:02x}, ")
            if (i + 1) % 12 == 0:
                f.write("\n")
        f.write(f"\n}};\nconst unsigned int {var_name}_len = {len(data)};\n")

if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")

    model = build_ds_cnn(X_train.shape[1:])
    model.compile(optimizer="adam",
                   loss="sparse_categorical_crossentropy",
                   metrics=["accuracy"])
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True)

    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs=50,
              batch_size=32,
              callbacks=[early_stop])

    # --- Evaluate on held-out test set (never touched until now) ---
    test_loss, test_acc = model.evaluate(X_test, y_test)
    preds = np.argmax(model.predict(X_test), axis=1)
    tp = np.sum((preds == 1) & (y_test == 1))
    fp = np.sum((preds == 1) & (y_test == 0))
    fn = np.sum((preds == 0) & (y_test == 1))
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    print(f"\nTest accuracy: {test_acc:.4f}")
    print(f"Precision: {precision:.4f}  Recall: {recall:.4f}")
    print(f"False positives: {fp}  False negatives: {fn}")

    # --- Save Keras model ---
    h5_path = os.path.join(MODELS_DIR, "spectra_model.h5")
    model.save(h5_path)
    print(f"Saved {h5_path}")

    # --- Convert to INT8-quantized TFLite ---
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset_gen(X_train)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    tflite_path = os.path.join(MODELS_DIR, "spectra_model.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"Saved {tflite_path} ({len(tflite_model)/1024:.1f} KB)")

    # --- Convert to .cc for ESP32 firmware ---
    cc_path = os.path.join(MODELS_DIR, "spectra_model.cc")
    convert_tflite_to_c_array(tflite_path, cc_path)
    print(f"Saved {cc_path}")