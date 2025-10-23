from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from PIL import Image
import numpy as np
import tempfile
import os
import cv2

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load TFLite model using TensorFlow ---
model_path = os.path.join(os.path.dirname(__file__), "plant_disease_model.tflite")
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ['Healthy', 'Powdery', 'Rust']
img_size = (150, 150)  # same as your model input

# --- Helper functions ---
def refine_mask_with_grabcut(img_rgb, mask):
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)

    grabcut_mask = np.where(mask == 0, 0, 3).astype('uint8')
    cv2.grabCut(img_rgb, grabcut_mask, None, bg_model, fg_model, 5, cv2.GC_INIT_WITH_MASK)
    final_mask = np.where((grabcut_mask == 2) | (grabcut_mask == 0), 0, 1).astype('uint8')
    return final_mask * 255

def remove_plant_background(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lower_green = np.array([25, 25, 25])
    upper_green = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    v = hsv[:, :, 2]
    s = hsv[:, :, 1]
    dark_mask = (v < 30) | (s < 30)
    mask[dark_mask] = 0

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    mask = refine_mask_with_grabcut(img_rgb, mask)
    result = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
    result[mask == 0] = [255, 255, 255]

    return Image.fromarray(result)

def preprocess_image(image_path):
    img_no_bg = remove_plant_background(image_path)
    img_resized = img_no_bg.resize(img_size)
    img_array = np.array(img_resized, dtype=np.float32) / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)  # batch dimension
    return img_array

def predict_with_tflite(img_array):
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = int(np.argmax(output_data[0]))
    confidence = float(output_data[0][predicted_class]) * 100
    return predicted_class, confidence

# --- API endpoints ---
@app.get("/")
async def root():
    return {"message": "Plant Disease Detection API is running!", "status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(await file.read())
            temp_path = temp_file.name

        # Preprocess and predict
        img_array = preprocess_image(temp_path)
        predicted_class, confidence = predict_with_tflite(img_array)

        return JSONResponse(content={
            "predicted_class": class_names[predicted_class],
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
