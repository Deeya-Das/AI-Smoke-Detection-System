# AI-Based Smoke Detection System

## Project Overview

This project aims to develop a real-time AI-based Smoke Detection System using Python, OpenCV, PyTorch, and MobileNetV2.

The system captures live video from the laptop webcam, detects smoke using a trained deep learning model, and triggers an alarm when smoke is detected.

Current Prototype:
- Smoke Detection
- Normal Detection

Future Enhancement:
- Smoke vs Steam Detection to reduce false alarms.

---

# Technologies Used

- Python
- OpenCV
- PyTorch
- TorchVision
- MobileNetV2
- NumPy
- VS Code

---

# Project Structure

```
SmokeDetection/
│
├── main.py              # Webcam and image capture
├── train.py             # Model training (To be implemented)
├── dataset/
│   ├── smoke/
│   └── normal/
```

---

# Dataset Links

## Smoke Dataset

https://www.kaggle.com/datasets/ata999/fire-and-smoke

Use only the **Smoke** class images from this dataset.

---

## Normal Dataset

https://www.kaggle.com/datasets/puneet6060/intel-image-classification

Use images without smoke (for example: buildings, forest, mountain, sea, street).

---

# Work Completed

✅ Project planning

✅ Environment setup

✅ OpenCV installation

✅ PyTorch installation

✅ Webcam module

✅ Live webcam capture

✅ Image saving

✅ Multiple image dataset collection tool

✅ Dataset selection

---

# Next Steps

1. Download both Kaggle datasets.
2. Extract the ZIP files.
3. Copy Smoke images into:

```
dataset/smoke/
```

4. Copy Normal images into:

```
dataset/normal/
```

5. Create `train.py`.
6. Load the dataset using PyTorch.
7. Train MobileNetV2.
8. Save the trained model (`model.pth`).
9. Connect the trained model with the webcam.
10. Trigger an alarm when smoke is detected.

---

# Model Selected

**MobileNetV2**

Reason:
- Lightweight
- Fast
- Works well on CPU
- Suitable for real-time webcam applications
- Good balance between speed and accuracy

---

# Current Status

🟢 Webcam module completed.

🟢 Dataset collection completed.

🟢 Ready to begin model training using PyTorch.
