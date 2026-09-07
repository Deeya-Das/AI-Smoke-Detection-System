import io
from datetime import datetime

import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import models, transforms


# Constants

MODEL_PATH = "model.pth"
CLASS_NAMES = ["fire", "normal"]   
FIRE_INDEX = CLASS_NAMES.index("fire")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing 
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])



# Load Model Once At Startup

def load_detector() -> nn.Module:
    net = models.mobilenet_v2(weights=None)
    net.classifier[1] = nn.Linear(net.classifier[1].in_features, len(CLASS_NAMES))
    net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    net.to(DEVICE)
    net.eval()
    return net


detector = load_detector()



# FastAPI App

app = FastAPI(title="AI-Based Early Fire & Smoke Detection System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Inference Helper

def predict_image(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = detector(tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        predicted_index = int(torch.argmax(probabilities))

    label = "Fire" if predicted_index == FIRE_INDEX else "Normal"
    confidence = probabilities[predicted_index].item() * 100
    return label, confidence



# Prediction Endpoint

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    label, confidence = predict_image(image_bytes)

    return {
        "status": label,
        "confidence": round(confidence, 2),
        "alarm": label == "Fire",
        "time": datetime.now().strftime("%d-%m-%Y %I:%M %p")
    }



# Serve The Existing Frontend

app.mount("/", StaticFiles(directory="static", html=True), name="frontend")
