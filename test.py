import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

print("=" * 50)
print("Fire Detection Test")
print("=" * 50)

# =====================================
# Load MobileNetV2
# =====================================
model = models.mobilenet_v2(weights=None)

# Replace Final Layer
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    2
)

# =====================================
# Load Trained Model
# =====================================
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

print("Model Loaded Successfully!\n")

# =====================================
# Image Preprocessing
# =====================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =====================================
# Image Path
# =====================================
image_path = "test.jpg"      # Change image if needed

# =====================================
# Read Image
# =====================================
image = Image.open(image_path).convert("RGB")
image = transform(image)
image = image.unsqueeze(0)

# =====================================
# Predict
# =====================================
with torch.no_grad():

    outputs = model(image)

    probabilities = torch.softmax(outputs, dim=1)

    prediction = torch.argmax(probabilities, dim=1).item()

# =====================================
# Class Names
# =====================================
classes = ["fire", "normal"]

# =====================================
# Display Result
# =====================================
print("-" * 50)
print("Fire Probability   :", round(probabilities[0][0].item() * 100, 2), "%")
print("Normal Probability :", round(probabilities[0][1].item() * 100, 2), "%")
print("-" * 50)
print("Prediction :", classes[prediction])
print("-" * 50)
