from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tflite_runtime.interpreter as tflite
import tensorflow as tf
from PIL import Image
import numpy as np
import tempfile
import io
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

model_path = os.path.join(os.path.dirname(__file__), "plant_disease_model.tflite")

# Load TFLite model
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ['Healthy', 'Powdery', 'Rust']
batch_size = 64
img_size = (150, 150)

def refine_mask_with_grabcut(img_rgb, mask):
    # Initialize background/foreground models
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)

    # Create initial mask for GrabCut
    grabcut_mask = np.where(mask == 0, 0, 3).astype('uint8')

    cv2.grabCut(img_rgb, grabcut_mask, None, bg_model, fg_model, 5, cv2.GC_INIT_WITH_MASK)
    final_mask = np.where((grabcut_mask == 2) | (grabcut_mask == 0), 0, 1).astype('uint8')
    return final_mask * 255

def remove_plant_background(image_path, show_steps=True):
    """Remove background from plant image using color segmentation"""

    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # covert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define green range
    lower_green = np.array([25, 25, 25])   # Lower bound
    upper_green = np.array([95, 255, 255]) # Upper bound

    # Create mask for green pixels
    mask = cv2.inRange(hsv, lower_green, upper_green)

    v = hsv[:, :, 2]
    s = hsv[:, :, 1]
    dark_mask = (v < 30) | (s < 30)
    mask[dark_mask] = 0

    # Clean up the mask
    kernel = np.ones((5, 5), np.uint8)

    # Remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    mask = refine_mask_with_grabcut(img_rgb, mask)

    # Apply the Mask
    result = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
    result[mask == 0] = [255, 255, 255]

    return Image.fromarray(result), mask

def load_and_preprocess_image(image_path, img_size=img_size, remove_bg=True):
    """Load and preprocess a single image"""

    if remove_bg:
       img_no_bg, mask = remove_plant_background(image_path)

       # Resize
       img = img_no_bg.resize(img_size)
    
    else:
        img = tf.keras.preprocessing.image.load_img(
            image_path,
            target_size=img_size)

    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    return img_array

def predict_single_image(file_bytes):
    """Predict class for a single image"""

    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name
    try:
        # Preprocess
        img_array = load_and_preprocess_image(temp_path, img_size=img_size, remove_bg=True)

        # Ensure float32 if model expects it
        img_array = img_array.astype(np.float32)

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], img_array)

        # Run inference
        interpreter.invoke()

        # Predict
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_class_index = int(np.argmax(output_data[0]))

        return {"predicted_class": class_names[predicted_class_index]}
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/")
async def root():
    return {"message": "Plant Disease Detection API is running!", "status": "healthy"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        result = predict_single_image(file_bytes)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )