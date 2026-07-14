import cv2
import os

# -----------------------------
# Create dataset folders
# -----------------------------
os.makedirs("dataset/smoke", exist_ok=True)
os.makedirs("dataset/steam", exist_ok=True)
os.makedirs("dataset/normal", exist_ok=True)

# Counters
smoke_count = 1
steam_count = 1
normal_count = 1

# Open Webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera could not be opened!")
    exit()

print("====================================")
print("Press 1 -> Save Smoke Image")
print("Press 2 -> Save Steam Image")
print("Press 3 -> Save Normal Image")
print("Press G -> Show Grayscale")
print("Press Q -> Quit")
print("====================================")

while True:

    success, frame = camera.read()

    if not success:
        print("Could not read frame!")
        break

    cv2.imshow("Smoke Detection System", frame)

    key = cv2.waitKey(1)

    # ---------------- Smoke ----------------
    if key == ord('1'):
        filename = f"dataset/smoke/smoke_{smoke_count}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved : {filename}")
        smoke_count += 1

    # ---------------- Steam ----------------
    elif key == ord('2'):
        filename = f"dataset/steam/steam_{steam_count}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved : {filename}")
        steam_count += 1

    # ---------------- Normal ----------------
    elif key == ord('3'):
        filename = f"dataset/normal/normal_{normal_count}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved : {filename}")
        normal_count += 1

    # ---------------- Grayscale ----------------
    elif key == ord('g'):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Gray Image", gray)

    # ---------------- Quit ----------------
    elif key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()