import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf

print("Loading .h5 model...", flush=True)

model = tf.keras.models.load_model("AgriVision_XAI_Model.h5", compile=False)

print("Converting to .tflite...", flush=True)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

os.makedirs("models", exist_ok=True)

with open(r"models\agrivision_edge_model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite created successfully!", flush=True)
print("Saved at: models/agrivision_edge_model.tflite", flush=True)