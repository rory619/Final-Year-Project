

import os
import json
import base64
import tempfile
import datetime
import time

import numpy as np
import cv2
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ai_edge_litert.interpreter import Interpreter


# App setup

app = FastAPI()


# File paths

BASE_DIR     = Path("/home/rory/fruitchecker/inference")
CAPTURES_DIR = BASE_DIR / "captures"
MODEL_PATH   = BASE_DIR / "model.tflite"
LABELS_PATH  = BASE_DIR / "labels.txt"
HISTORY_FILE = BASE_DIR / "history.json"


# Load class labels from labels.txt (one label for every line)

labels = [
    line.strip()
    for line in LABELS_PATH.read_text().splitlines()
    if line.strip()
]


# Load the TFLite model using LiteRT

interpreter = Interpreter(model_path=str(MODEL_PATH))
interpreter.allocate_tensors() # allocate memory to input/output tensor

# Get the details of the input and output tensors
input_details  = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

# Read the expected image size and data type from the model
_, input_height, input_width, _ = input_details["shape"]
input_dtype = input_details["dtype"]



# Run inference on the image
# Uses raw image bytes, returns a prediction with the label and confidence score
def run_inference(image_bytes: bytes) -> dict:
    # Decode the image bytes into an OpenCV image array
    
    frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    # Convert from BGR (OpenCV default) to RGB (what the model expects)
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize the image to match the model's expected input size
    rgb_image = cv2.resize(rgb_image, (input_width, input_height))

    # Add a batch dimension so the shape is now (1, height, width, 3)
    tensor = np.expand_dims(rgb_image, axis=0).astype(input_dtype)


    # Plug the image into the model and run prediction
    interpreter.set_tensor(input_details["index"], tensor)
    interpreter.invoke()

    # Read the output scores (one score per class)
    scores = interpreter.get_tensor(output_details["index"])[0]
    print("DEBUG SCORES:", scores)
    print("DEBUG LABELS:", labels)

    # Find the index of the highest score
    best_index = int(np.argmax(scores))
    best_score = float(scores[best_index])

    # Look up the label name, or fall back to a generic class name
    label = labels[best_index] if 0 <= best_index < len(labels) else f"class_{best_index}"

    return {"label": label, "score": round(best_score, 4)}



# Save a prediction result to history.json

def save_to_history(entry: dict, image_bytes: bytes = None):
    # Option to embed the image as a base64 string so the app can display it
    if image_bytes:
        entry["image"] = base64.b64encode(image_bytes).decode("utf-8")

    # Load existing history, or start with new one if the file doesn't exist
    history = []
    if HISTORY_FILE.exists():
        history = json.loads(HISTORY_FILE.read_text())

    # Add the new entry to the top of the list
    history.insert(0, entry)
    # Keep only the 20 most recent entries
    history = history[:20]
    

    # Write the updated history back to the file
    HISTORY_FILE.write_text(json.dumps(history, indent=2))



# health check so the app can confirm the Pi server is running

@app.get("/status")
def status():
    return {"status": "online", "model": "fruit_ripeness"}



# Capturing a photo using the Raspberry Pi camera, then runs fruit prediction
@app.get("/capture")
def capture_and_predict():
    # Create a temporary file to save the image
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.close()

    camera = None

    try:
        from picamera2 import Picamera2

        camera = Picamera2()

        # Set up the camera for a still photo at 720p
        config = camera.create_still_configuration(main={"size": (1280, 720)})
        camera.configure(config)
        camera.start()

       #af mode = 2: continuous auto focus
        camera.set_controls({"AfMode": 2})

        # Give the camera a moment to settle before capturing
        time.sleep(2)

        # Save the photo to the temporary file
        camera.capture_file(tmp.name)

    except Exception as error:
        #  return an error if there is failur
        os.unlink(tmp.name)
        return JSONResponse({"error": f"Camera failed: {str(error)}"}, status_code=500)

    finally:
        # stop and close the camera, even if an error occurred
        if camera:
            try:
                camera.stop()
                camera.close()
            except Exception:
                pass

    # Read the saved image and delete the temporary file
    image_bytes = Path(tmp.name).read_bytes()
    os.unlink(tmp.name)

    # Run the fruit ripeness prediction
    result = run_inference(image_bytes)
    
    result["source"]    = "pi_camera"
    result["timestamp"] = datetime.datetime.now().isoformat()

    save_to_history(result, image_bytes)

    return JSONResponse(result)



# Uses image uploaded from smartphone, then runs the prediction
@app.post("/predict")
async def predict_from_upload(file: UploadFile = File(...)):
    # Read the uploaded image bytes
    image_bytes = await file.read()

    # Run the fruit ripeness prediction
    result = run_inference(image_bytes)
    result["source"]    = "phone_camera"
    result["timestamp"] = datetime.datetime.now().isoformat()

    save_to_history(result, image_bytes)

    return JSONResponse(result)



# Returns the last 20 predictions stored on the Pi

@app.get("/history")
def get_history():
    if not HISTORY_FILE.exists():
        return []

    return json.loads(HISTORY_FILE.read_text())

