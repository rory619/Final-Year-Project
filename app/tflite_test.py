import os
import cv2
import numpy as np

# Try to use tflite-runtime (lighter for Raspberry Pi etc.),
# fall back to TensorFlow Lite if that's what you installed.
try:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter


def load_labels(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    # --- File paths ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model.tflite")
    labels_path = os.path.join(script_dir, "labels.txt")

    if not os.path.exists(model_path):
        print("model.tflite not found:", model_path)
        return
    if not os.path.exists(labels_path):
        print("labels.txt not found:", labels_path)
        return

    labels = load_labels(labels_path)

    # --- Load TFLite model (standard Interpreter pattern) ---
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Assume first input is [1, height, width, 3]
    input_shape = input_details[0]["shape"]
    ih, iw = int(input_shape[1]), int(input_shape[2])
    input_dtype = input_details[0]["dtype"]

    print("Model expects:", input_shape, input_dtype)

    # --- Capture ONE frame from camera ---
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Could not read from camera.")
        return

    # --- Preprocess: BGR -> RGB, resize, add batch ---
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (iw, ih))
    input_data = np.expand_dims(img, axis=0).astype(input_dtype)

    # If model is float, scale 0–255 to 0–1
    if input_dtype == np.float32:
        input_data = input_data / 255.0

    # --- Run inference (Interpreter pattern from docs) ---
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])[0]

    # --- Get best class ---
    idx = int(np.argmax(output_data))
    score = float(output_data[idx])

    if 0 <= idx < len(labels):
        label = labels[idx]
    else:
        label = f"class_{idx}"

    print(f"Prediction: {label} (score={score:.2f})")


if __name__ == "__main__":
    main()
