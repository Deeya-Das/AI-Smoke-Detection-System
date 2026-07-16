import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

print("=" * 60)
print("        AI-Based Fire Detection System")
print("=" * 60)

# =====================================================
# Load MobileNetV2 Model
# =====================================================
model = models.mobilenet_v2(weights=None)

# Replace Final Layer
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    2
)

# =====================================================
# Load Trained Model
# =====================================================
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

print("\nModel Loaded Successfully!")

# =====================================================
# Image Transform
# =====================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =====================================================
# Class Names
# =====================================================
classes = ["fire", "normal"]

# =====================================================
# Test Folder
# =====================================================
test_folder = "test_images"

correct = 0
total = 0

print("\nStarting Testing...\n")
print("-" * 60)

# =====================================================
# Test All Images
# =====================================================
for actual_class in classes:

    folder_path = os.path.join(test_folder, actual_class)

    if not os.path.exists(folder_path):
        print(f"Folder not found : {folder_path}")
        continue

    for file_name in os.listdir(folder_path):

        image_path = os.path.join(folder_path, file_name)

        try:

            # Load Image
            image = Image.open(image_path).convert("RGB")
            image = transform(image)
            image = image.unsqueeze(0)

            # Predict
            with torch.no_grad():

                output = model(image)

                probabilities = torch.softmax(output, dim=1)

                fire_prob = probabilities[0][0].item() * 100
                normal_prob = probabilities[0][1].item() * 100

                prediction = torch.argmax(output, dim=1).item()

            predicted_class = classes[prediction]

            # Check Correct Prediction
            if predicted_class == actual_class:
                correct += 1
                result = "Correct"
            else:
                result = "Wrong"

            total += 1

            print(f"Image : {file_name}")
            print(f"Actual      : {actual_class}")
            print(f"Prediction  : {predicted_class}")
            print(f"Fire Prob   : {fire_prob:.2f}%")
            print(f"Normal Prob : {normal_prob:.2f}%")
            print(f"Result      : {result}")
            print("-" * 60)

        except Exception as e:
            print(f"Error reading {file_name}")
            print(e)
            print("-" * 60)

# =====================================================
# Accuracy
# =====================================================
if total > 0:

    accuracy = (correct / total) * 100

    print("\n" + "=" * 60)
    print("Testing Completed")
    print("=" * 60)

    print(f"Total Images        : {total}")
    print(f"Correct Predictions : {correct}")
    print(f"Wrong Predictions   : {total - correct}")
    print(f"Accuracy            : {accuracy:.2f}%")

    print("=" * 60)

else:

    print("No test images found.")
    
