# Drowsiness Detection System

# Drowsiness Detection System

## Overview
The **Drowsiness Detection System** is a real-time application designed to monitor a person’s alertness by analyzing eye and mouth regions.  
It combines **YOLOv8 models** and **MediaPipe landmarks** inside a **PyQt5 GUI**, allowing easy visualization and real-time alerts.

This repository provides the full pipeline: dataset capture, auto-labeling, model training, and deployment.

---

## Features
- **Real-Time Detection** – Monitors drowsiness via webcam.
- **Dual YOLOv8 Models** – Detects:
  - **Eye state** (open/closed).
  - **Yawning** (mouth open/closed).
- **MediaPipe Landmarks** – Assists in locating facial ROIs.
- **GUI with PyQt5** – Displays live video feed, confidence scores, and alerts.
- **Data Capture Tools** – Scripts to record and organize datasets.
- **Auto Labeling** – Support for GroundingDINO auto-labeling.

---

## Repository Structure
- `AutoLabelling.py` – Automated bounding box labeling with GroundingDINO.
- `CaptureData.py` – Capture video frames for dataset creation.
- `DrowsinessDetector.py` – Main detection system with GUI.
- `DD_bymyself.py` – Custom/experimental detection script.
- `LoadData.ipynb` – Load and preprocess dataset.
- `RedirectData.ipynb` – Organize dataset for training.
- `train.ipynb` – Notebook for YOLOv8 model training.
- `requirements.txt` – Python dependencies.
- `dataset.yaml` – Dataset configuration for YOLO.

---

## Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/TrungVo57/real-time-drowsy-driving-detection.git
   cd real-time-drowsy-driving-detection

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/tyrerodr/Real_time_drowsy_driving_detection.git
    cd Real_time_drowsy_driving_detection
    ```

2. **Create a virtual environment:**
    ```bash
    python -m venv venv
    venv\Scripts\activate     # On Windows
    source venv/bin/activate  # On Linux/Mac


3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the detection system:**
    ```bash
    python DrowsinessDetector.py
    ```

---

## Usage

- **Real-Time Detection:** Run `DrowsinessDetector.py` with a connected webcam to monitor drowsiness.
- **Data Capture:** Use `CaptureData.py` to collect video frames for training or testing.
- **Training New Models:** Use `train.ipynb` to retrain the models on your custom datasets.

---

## How It Works

How It Works

Face Detection & Landmarks: MediaPipe FaceMesh extracts facial landmarks (eyes & mouth regions).

YOLOv8 Inference:

Eye model → detects open/closed eyes.

Yawn model → detects yawning.

Logic Engine:

Count blinks.

Detect microsleeps if eyes closed for >2s.

Detect prolonged yawns.

Alert System: Show warnings on GUI and play alert sounds.

Technologies Used

Python 3.10+

YOLOv8 (Ultralytics) – Object detection.

OpenCV – Image processing.

MediaPipe – Facial landmarks.

PyQt5 – Graphical user interface.

NumPy, Pandas, Matplotlib, Seaborn – Data analysis & visualization.

Future Improvements

Add driver head pose estimation.

Multi-person detection in the same frame.

Deploy on mobile/edge devices (e.g. Jetson Nano, Raspberry Pi).

Author

Trung Vo