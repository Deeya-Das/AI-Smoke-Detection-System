# AI-Based Early Fire & Smoke Detection System

### Summer Internship Project — Indian Oil Corporation Limited (IOCL), Guwahati Refinery

An image-based fire and smoke detection tool. You upload a photo, and it tells you whether the scene looks safe or shows signs of fire, along with how confident the model is. If fire is detected, the dashboard turns red and an alarm starts playing until someone manually stops it.

This was built as a group project. We wanted to explore how a basic AI safety-monitoring tool could work, using industries like oil refineries as the real-world use case — places where even a small delay in spotting a fire can be dangerous.

## Team

- [Your Name]
- [Teammate 2]
- [Teammate 3]
- (add the rest of the team here)

## The problem we're trying to solve

A fire almost never starts big. It usually begins small — a spark, a leak catching flame, or smoke building up quietly before anyone notices. If that small stage gets caught early, it can be handled quickly and safely. But if it goes unnoticed even for a few minutes, it can turn into a major fire that causes serious damage, injuries, or worse.

That's the gap our system is trying to fill. The idea is simple: catch the problem while it's still small. If our system spots even early signs of fire or smoke in an image, it can flag it immediately instead of waiting for a person to notice it visually. That early warning is the difference between someone quickly putting it out and it becoming a major incident. So the real goal of this project isn't just "detect fire" — it's to reduce the chances of a small issue turning into a big, damaging one.

## Why we built this

In places like refineries, fire safety monitoring is still largely dependent on people watching camera feeds or walking around and physically checking. That works, but it's slow and it depends on someone noticing in time. We wanted to see if a lightweight AI model could act as an extra layer of monitoring — something that never gets tired or distracted, and can flag a fire the moment it sees one.

This is a prototype, not a finished product. We are not claiming this is ready to be plugged into an actual refinery today. It's meant to show the core idea working end to end.

## What it actually does right now

- You upload a single image on the website
- The image goes to our backend, where a trained model checks it
- It tells you if the image looks like a normal scene or shows fire
- If fire is detected: the screen turns red, shows the confidence percentage, and an alarm sound starts playing until you press "Stop Alarm"
- If it's a normal scene: the screen stays green and nothing else happens

Right now the model is trained on two categories — fire and normal. Smoke is part of the bigger picture we care about (a lot of real fires are caught early because of visible smoke, not visible flames), but in this version, smoke images are grouped along with fire during training rather than being their own separate category. Making smoke its own detection class is one of the first things we want to improve next, so the system can catch early smoke before flames are even visible.

## How it works (in simple terms)

1. We trained a model called MobileNetV2 on a set of fire and normal images. MobileNetV2 is a smaller, faster version of the kind of models used for image recognition — we picked it because it runs quickly even without a powerful computer.
2. Once trained, the model's knowledge is saved into a file.
3. A small backend server loads that trained model one time when it starts up, so it doesn't have to reload it for every single image.
4. When someone uploads a photo, the server resizes it, feeds it to the model, and gets back a prediction with a confidence score.
5. That result is sent back to the website, which updates the screen and plays the alarm if needed.

## Tech we used

- **Model training:** Python, PyTorch
- **Backend server:** FastAPI (handles the image upload and runs the prediction)
- **Frontend:** plain HTML, CSS, and JavaScript — no frameworks, built from scratch so we understood exactly how everything connects

## Project structure

```
fire-detection/
├── app.py              # backend server, loads the model and handles predictions
├── train.py             # script used to train the model
├── test.py               # script used to test model accuracy
├── requirements.txt
├── model.pth              # our trained model
└── static/
    ├── index.html
    ├── style.css
    ├── script.js
    └── alarm.wav
```

## Running it yourself

```bash
pip install -r requirements.txt
python -m uvicorn app:app --reload
```

Then open `http://127.0.0.1:8000` in your browser.

## What this prototype is not (being upfront about it)

- It does not connect to any live camera or CCTV feed — you have to manually upload a photo
- It only works on still images right now, not video
- Nothing gets saved or logged — every prediction is a one-time check, there's no history
- Smoke is not yet detected as its own separate case
- The alarm only works inside the browser tab — it is not connected to any real siren or alert system

## Challenges we faced

- **Finding good training images was harder than we expected.** There aren't many large, clean, publicly available datasets specifically of refinery or industrial fire scenes. Most fire image datasets online are things like forest fires, house fires, or random stock photos, which don't always represent what fire actually looks like in an industrial setting. We had to spend a good chunk of time collecting and sorting images just to get a dataset that was usable.
- **Keeping the "normal" images realistic.** It's easy to find fire pictures, but we also needed a lot of normal, non-fire images that still resembled a refinery or industrial environment, otherwise the model would just learn to recognize "outdoor scene vs indoor scene" instead of actually learning what fire looks like.
- **False positives from fire-like colors.** Things like orange lighting, sunsets, or bright red/orange equipment sometimes confused the model early on, since the color pattern is visually close to fire. This pushed us to pay more attention to the variety of images in our dataset rather than just the number of images.
- **Balancing the two classes.** If one class (fire or normal) had a lot more images than the other, the model would lean toward predicting whichever class had more examples. We had to actively check and balance the dataset instead of just dumping in whatever images we found.
- **Connecting the trained model to a working website.** Training the model in a notebook is one thing, but getting it to load correctly inside a live backend, handle real uploaded images the same way it was trained, and return a result fast enough to feel real-time took a fair amount of trial and error.

## Where we want to take this next

The biggest goal is to move from "upload one photo manually" to **live CCTV integration**. In a real refinery setting, cameras are already running 24/7. The next real step for this project is to connect this same model to a live camera feed instead of a manual upload, so it can continuously watch the footage and raise an alert the moment it spots fire or smoke — without anyone needing to check it themselves.

Other things on our list:
- Training the model to recognize smoke as its own category, separate from fire
- Saving a log of every detection with the time it happened, so there's a record to look back on
- Sending real alerts (not just an on-screen alarm) to whoever is responsible for safety
- Testing the model on harder cases — night-time footage, foggy conditions, small early-stage fires

## Why MobileNetV2

We picked MobileNetV2 over a bigger, more accurate model because it's much lighter and faster to run. In a real safety system, speed matters — a model that takes several seconds to check one frame isn't very useful if a fire is spreading in that time. The trade-off is that it may not be as sharp on tricky cases like thick smoke or unusual lighting, which is exactly why we show the confidence score instead of just a plain yes or no — so a human can still use their judgment on borderline cases.
