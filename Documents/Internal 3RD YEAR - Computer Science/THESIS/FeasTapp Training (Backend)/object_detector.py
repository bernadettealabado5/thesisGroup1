import firebase_admin
from firebase_admin import credentials
from flask import request, Response, Flask
from waitress import serve
from PIL import Image
import json
from ultralytics import YOLO
import mrcnn.model as modellib
from mrcnn.config import Config
import numpy as np
import keras.engine as KE
import tensorflow.keras.engine as KE

cred = credentials.Certificate("feastapp-c4d79-firebase-adminsdk-wygrr-5dee5a902c.json")
firebase_admin.initialize_app(cred)

app = Flask(__name__)

# Sample dish recommendation function
def recommend_dish(ingredient):
    # This is a placeholder, you can replace it with your actual recommendation logic
    dishes = {
        "tomato": "Caprese Salad",
        "apple": "Apple Pie",
        "banana": "Banana Bread",
        "carrot": "Carrot Soup",
        "eggplant": "Eggplant Parmesan",
        "lettuce": "Caesar Salad",
        "onion": "Adobo",
        "chicken": "Chicken Adobo"
    }
    return dishes.get(ingredient, "Unknown Dish")

# List of main meat ingredients
main_meat_ingredients = ["chicken", "pork", "beef", "lamb", "fish", "shrimp", "duck", "turkey"]

# List of condiments
condiments = ["salt", "pepper", "soy sauce", "vinegar", "ketchup", "mayonnaise", "mustard"]

# Mask R-CNN configuration
class IngredientConfig(Config):
    NAME = "ingredient"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 2  # Background + main ingredient + condiment
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.9

# Create Mask R-CNN model
mask_rcnn_model = modellib.MaskRCNN(mode="inference", config=IngredientConfig(), model_dir="./")

# Load pre-trained weights (you need to specify the path to the weights file)
mask_rcnn_model.load_weights("path_to_weights.h5", by_name=True)

# YOLO model initialization
yolo_model = YOLO("best.pt")

@app.route("/")
def root():
    with open("index.html") as file:
        return file.read()

@app.route("/detect", methods=["POST"])
def detect():
    buf = request.files["image_file"]
    boxes_yolo = detect_objects_on_image_yolo(Image.open(buf.stream))
    boxes_mask_rcnn = detect_objects_on_image_mask_rcnn(Image.open(buf.stream))
    return Response(
        json.dumps({
            "yolo": boxes_yolo,
            "mask_rcnn": boxes_mask_rcnn
        }),
        mimetype='application/json'
    )

def detect_objects_on_image_yolo(image):
    results = yolo_model.predict(image)
    result = results[0]
    output = []
    for box in result.boxes:
        x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        ingredient = result.names[class_id]
        dish = recommend_dish(ingredient)
        category = "Main Meat" if ingredient in main_meat_ingredients else "Condiment"
        output.append([x1, y1, x2, y2, ingredient, prob, dish, category])
    return output

def detect_objects_on_image_mask_rcnn(image):
    # Run Mask R-CNN detection
    results = mask_rcnn_model.detect([np.array(image)], verbose=1)
    r = results[0]
    output = []
    for i in range(r['rois'].shape[0]):
        class_id = r['class_ids'][i]
        score = r['scores'][i]
        label = "ingredient" if class_id == 1 else "condiment"
        category = "Main Meat" if label == "ingredient" and r['class_ids'][i] in main_meat_ingredients else "Condiment"
        x1, y1, x2, y2 = r['rois'][i]
        output.append([x1, y1, x2, y2, label, score, recommend_dish(label), category])
    return output

serve(app, host='0.0.0.0', port=8080)
