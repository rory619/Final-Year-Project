from pathlib import Path
import requests
from datetime import datetime

PI_API = "http://192.168.0.11:8000"  # change the IP based on the Pi's current address

import cv2
import numpy as np


try:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
    
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter #Using tflite instead of tensorflow because its lighter


def load_labels(path: str) -> list[str]:
    
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def grab_webcam_frame(cam_index: int = 0) -> np.ndarray | None:
    cap = cv2.VideoCapture(cam_index)     #opening camera from the webcam and read one frame into BGR format
    if not cap.isOpened():
        print(f"Couldnt open camera {cam_index}")
        return None

    try:
        ret, frame = cap.read()
    finally:
        cap.release()

    if not ret:
        print("Couldnt read from camera.")
        return None
    return frame

def grab_pi_frame() -> np.ndarray | None:
    url = f"{PI_API}/capture"
    print(f"Fetching image from: {url}")

    resp = requests.get(url, timeout=10)  #gets jpeg bytes from the API
    resp.raise_for_status()    #error handling, throw an exception

    # Save the exact JPEG bytes 
    out_dir = Path(__file__).resolve().parent / "captures"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"pi_{datetime.now():%Y%m%d-%H%M%S}.jpg"
    out_file.write_bytes(resp.content)      #responce body
    print(f"Saved capture to: {out_file}")

    # Decode JPEG bytes OpenCV BGR frame
    arr = np.frombuffer(resp.content, np.uint8)    #changing raw bytes into NumPy that suits OpenCV
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)    #changes the buffer into an openCV image
    if frame is None:
        print("Failed to decode image from Pi.")   #print an error in case of decode failure
        return None
    return frame


def preprocess_frame(frame: np.ndarray, input_shape, input_dtype) -> np.ndarray:
   
    _, ih, iw, _ = input_shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   #convert BGR to RGB colour
    rgb = cv2.resize(rgb, (iw, ih))   #resize will match the image to the model input size

    input_data = np.expand_dims(rgb, axis=0).astype(input_dtype)    #make a batch shape and matches the tensor type the model expects
    if input_dtype == np.float32:
        input_data = input_data / 255.0
    return input_data


def main() -> None:
    
    script_dir = Path(__file__).resolve().parent
    model_path = script_dir / "model.tflite"
    labels_path = script_dir / "labels.txt"

    if not model_path.exists():
        print("model.tflite not found:", model_path)
        return
    if not labels_path.exists():
        print("labels.txt not found:", labels_path)
        return

    labels = load_labels(str(labels_path))

    #Load model, allocate sensors, setup input tensor, read output tensor
    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]["shape"]
    input_dtype = input_details[0]["dtype"]
    print("Model expects:", input_shape, input_dtype)

    frame = grab_pi_frame()
    if frame is None:
        return

    input_data = preprocess_frame(frame, input_shape, input_dtype)

    
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])[0]

    
    idx = int(np.argmax(output_data))   #argmax picks highest probability of which class(eg: UnripeBanana) the image resembles based off the image captured

    score = float(output_data[idx])

    label = labels[idx] if 0 <= idx < len(labels) else f"class_{idx}"
    print(f"Prediction: {label} (score={score:.2f})")


if __name__ == "__main__":
    main()