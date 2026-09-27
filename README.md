# AI-Based Fire & Smoke Detection System

### Summer Internship Project — Indian Oil Corporation Limited (IOCL), Guwahati Refinery

An image-based fire and smoke detection tool built using deep learning. Upload a photo, and the system tells you whether the scene looks safe or shows signs of fire, along with how confident the model is. If fire or smoke is detected, the dashboard turns red and an alarm plays until manually stopped.

This project was built as part of a one-month industrial internship at IOCL Guwahati Refinery, under the Information Systems Department, exploring how a lightweight AI model could support fire safety monitoring in an industrial environment like a refinery.

## Team

- Angkur Kakati
- Chanakya Deka
- Deeya Das
- Disha Handique
- Karan Kakati
- Prastuti Saikia

## The Problem

A fire rarely starts big — it usually begins as a small flame or a wisp of smoke before it becomes dangerous. If caught early, it can be handled quickly and safely. If missed for even a few minutes, it can escalate into serious damage. Fire safety monitoring in most industrial settings still relies heavily on people watching camera feeds or physically checking the area — which works, but depends entirely on someone noticing in time.

This project explores whether a lightweight AI model can act as an additional layer of monitoring — one that never gets tired or distracted, and can flag a fire or early smoke the moment it appears in an image.

## What It Does

- Upload a single image through the web dashboard
- The image is sent to a backend server, where a trained deep learning model analyzes it
- The system returns whether the scene is **Normal** or shows **Fire/Smoke**, along with a confidence score
- If fire/smoke is detected: the dashboard turns red, shows the confidence percentage, and plays an alarm sound until the user clicks "Stop Alarm"
- If the scene is normal: the dashboard shows a green "System Safe" banner and stays quiet

Currently, fire and smoke are treated as a single combined class rather than two separate categories — since a lot of real fires are first noticed through smoke rather than visible flame, separating the two is one of the planned next steps (see **Future Scope** below).

## How It Works

1. A **MobileNetV2** model (pretrained on ImageNet) was fine-tuned using transfer learning on a custom dataset of fire/smoke and normal industrial images.
2. The dataset was cleaned to remove exact and near-duplicate images (many stock photos were reused across multiple sources under different filenames), then split into training and testing sets with **zero overlap** between the two — this was an important fix, since an earlier, unclean version of the dataset was inflating accuracy through data leakage.
3. The trained model is saved as a single file (`model.pth`) and loaded once by the backend server at startup.
4. A **FastAPI** backend receives an uploaded image, processes it the same way it was processed during training, runs it through the model, and returns a prediction with a confidence score (via softmax).
5. A plain **HTML/CSS/JavaScript** dashboard (no frameworks) sends the image to the backend, then updates the screen and triggers the alarm based on the response.

## Tech Stack

- **Model training:** Python, PyTorch, Torchvision
- **Backend:** FastAPI, Uvicorn, Pillow
- **Frontend:** HTML, CSS, JavaScript — built from scratch, no frameworks
- **Training environment:** Google Colab (for free GPU access)

## Dataset

- 377 unique images after removing duplicates and near-duplicates (from an original 418)
- Split: **302 training** images (113 fire/smoke, 189 normal) and **75 testing** images (28 fire/smoke, 47 normal)
- Achieved **98.67% accuracy** on the held-out test set

## Project Structure

fire-detection/
├── app.py # FastAPI backend — loads the model and handles predictions
├── train.py # Script used to train the model
├── test.py # Script used to evaluate model accuracy
├── requirements.txt
├── model.pth # Trained model weights
└── static/
├── index.html
├── style.css
├── script.js
└── alarm.wav


## Running It Locally

```bash
pip install -r requirements.txt
python -m uvicorn app:app --reload
```

Then open `http://127.0.0.1:8000` in your browser.

## Current Limitations

- Works on still images only — no live camera/CCTV integration yet
- No detection history or logging — every check is a one-time result
- Fire and smoke are currently one combined class, not separate categories
- The alarm only sounds within the browser tab — it isn't connected to any real siren or alert system
- Dataset size (377 images) is modest; performance on harder cases (early-stage fires, low light, fog, smoke vs. steam) hasn't been specifically tested

## Future Scope

- **Live CCTV integration** — the biggest next step, moving from manual image upload to continuous live-feed monitoring
- **Separating smoke as its own detection class**, since smoke often appears before visible flame
- **Differentiating smoke from steam** — a real risk in refinery settings where steam is a normal byproduct of operations
- **Detection logging** with timestamps for later review
- **Real alert notifications** (SMS/email/control room alerts) instead of just an on-screen alarm
- **Testing under harder conditions** — night-time, fog, and very early-stage fires
- **Expanding the dataset**, particularly with more early-stage fire/smoke images

## Why MobileNetV2

MobileNetV2 was chosen over larger, more accurate architectures because it's lightweight and fast — a critical requirement for a safety system, where a model that takes several seconds per frame isn't very useful if a fire is actively spreading. The trade-off is reduced sharpness on tricky edge cases (thick smoke, unusual lighting), which is why the system reports a confidence score rather than a plain yes/no, allowing a human to apply judgment on borderline cases.
