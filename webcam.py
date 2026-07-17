import cv2
import torch
import torch.nn as nn
import threading
from playsound import playsound
from torchvision import models, transforms
from PIL import Image

print("=" * 50)
print("AI-Based Fire Detection System")
print("=" * 50)

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
# Alarm Function
# =====================================
alarm_running = False

def play_alarm():

    global alarm_running

    if alarm_running:
        return

    alarm_running = True

    try:
        playsound("alarm.wav")
    except:
        print("alarm.wav not found!")

    alarm_running = False

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

    # Convert OpenCV image to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image
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
    # Display Result
    # =====================================
    if label == "fire":

        color = (0, 0, 255)
        text = f"🔥 FIRE : {fire_prob:.2f}%"

        # Play Alarm
        if fire_prob >= 90:
            threading.Thread(
                target=play_alarm,
                daemon=True
            ).start()

    else:

        color = (0, 255, 0)
        text = f"✅ NORMAL : {normal_prob:.2f}%"

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

print("=" * 50)
print("Program Closed Successfully!")
print("=" * 50)
