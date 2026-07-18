import cv2
import torch
import torch.nn as nn
import pygame
from torchvision import models, transforms
from PIL import Image

print("=" * 50)
print("AI-Based Fire Detection System")
print("=" * 50)

# =====================================
# Initialize Alarm
# =====================================
pygame.mixer.init()
pygame.mixer.music.load("alarm.wav")

# =====================================
# Load Model
# =====================================
model = models.mobilenet_v2(weights=None)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    2
)

model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

print("Model Loaded Successfully!")

# =====================================
# Image Transform
# =====================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

classes = ["fire", "normal"]

# =====================================
# Open Webcam
# =====================================
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not open webcam.")
    exit()

print("Webcam Started Successfully!")
print("Press Q to Exit")

# =====================================
# Webcam Loop
# =====================================
while True:

    success, frame = camera.read()

    if not success:
        print("Could not read frame.")
        break

    # Convert OpenCV Image to PIL
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)

    # Preprocess
    image = transform(image)
    image = image.unsqueeze(0)

    # Prediction
    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(outputs, dim=1)

        prediction = torch.argmax(outputs, dim=1).item()

    fire_prob = probabilities[0][0].item() * 100
    normal_prob = probabilities[0][1].item() * 100

    label = classes[prediction]

    # =====================================
    # Fire
    # =====================================
    if label == "fire":

        color = (0, 0, 255)
        text = f"FIRE : {fire_prob:.2f}%"

        if fire_prob >= 90:

            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.play()

    # =====================================
    # Normal
    # =====================================
    else:

        color = (0, 255, 0)
        text = f"NORMAL : {normal_prob:.2f}%"

        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()

    # =====================================
    # Display Prediction
    # =====================================
    cv2.putText(
        frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.imshow(
        "AI-Based Fire Detection System",
        frame
    )

    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =====================================
# Release Resources
# =====================================
camera.release()
cv2.destroyAllWindows()
pygame.mixer.quit()

print("=" * 50)
print("Program Closed Successfully!")
print("=" * 50)
