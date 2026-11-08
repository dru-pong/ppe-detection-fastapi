# PPE Detection for Construction Sites (YOLOv8 + FastAPI)

This was my capstone project for the BSc in Computer Science at KNUST.
It detects whether workers are wearing a helmet, gloves, goggles and boots, and it flags anyone who is missing equipment.

## How it works

A YOLOv8 model fine-tuned on hand-labeled site images detects 9 classes.
The classes are helmet, gloves, boots, goggles, person, no_helmet, no_goggle, no_gloves and no_boots.
A FastAPI app serves the model in two ways.
You can upload a single image and get it back with the detections drawn on it.
You can also watch a live webcam stream with detections drawn on every frame.
When a missing-equipment class appears in the stream, an OpenCV face detector saves a timestamped crop of the worker.
Those crops appear in a violations gallery at /faces.

## Data and training

I labeled the images by hand in LabelImg and managed the dataset in Roboflow.
The model was fine-tuned with Ultralytics 8.3 for 20 epochs at 640 pixels with a batch size of 16, on CPU only.

| Metric (validation) | Score |
|---|---|
| mAP@50 | 0.53 |
| mAP@50-95 | 0.30 |
| Precision | 0.65 |
| Recall | 0.49 |

The model was trained for only 20 epochs on a CPU, so recall is the clearest area to improve.
More epochs on a GPU and more examples of the missing-equipment classes would be the next steps.

## Run it locally

You need Python 3.8 or newer and a webcam for the live stream.

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    uvicorn app:app --reload

Then open http://127.0.0.1:8000 in your browser.
The trained weights are included as best.pt.

## Privacy note

The face capture feature stores images of real people.
Anyone deploying this on a site should tell workers, keep the captures secure and delete them on a schedule.

## Licence

This project uses Ultralytics YOLOv8, which is licensed under AGPL-3.0, so this repository uses the same licence.
