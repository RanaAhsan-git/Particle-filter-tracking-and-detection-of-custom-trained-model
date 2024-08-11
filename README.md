# Particle Filter Tracking of custom trained model with YOLOv8

This project implements an object tracking system using a Particle Filter and YOLOv8  model. The Particle Filter is employed to track objects in video frames, while YOLOv8 is used for initial object detection.

# Overview


**Particle Filtering**


Particle Filtering is a method used for tracking objects in a video stream. It involves representing the state of an object by a set of particles, each representing a possible state. The particles are updated over time based on their predicted motion and the likelihood of their state given the observed data. The key steps are:

**Initialization:**

Generate particles around the initial position of the object.

**Prediction:** 

Update the particles based on a motion model, adding some noise to simulate movement.

**Update:**

Adjust the weights of the particles based on how well they match the observed data.

**Resampling:**

Select particles with higher weights to form the next set of particles.

**Estimation:** 

Compute the estimated state of the object from the particles.

# How It Works


**Initialization:**

The YOLOv8 model is used to detect the initial object in the video frame. The Particle Filter is initialized with this detection.

**Tracking Loop:**

The system predicts the new positions of particles based on their previous states and some added noise.

It then updates these predictions based on the new frame and adjusts particle weights according to how well they match the observed object.

The particles are resampled to focus on more likely states.

The estimated position of the object is drawn on the frame.

# System Requirements

Python 3.7 or higher

OpenCV

NumPy

YOLOv8 (Ultralytics package)

# Setting Up the Environment

**Create a Virtual Environment (optional but recommended):**
```
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

**Install Dependencies:**
```
pip install -r requirements.txt
```

**Download YOLOv8 Model:**

Ensure you have the YOLOv8 model file (Ahsan Pen.pt). Place it in the directory specified in the code or update the path accordingly.

**Prepare Video File:**

Ensure you have a video file (inputvideo.mp4) for testing. Place it in the directory specified in the code or update the path accordingly.

# Usage

Run the script using:
```
python tracker.py
```
# Script Description

**ParticleFilterTracker Class:**

Manages the particle filter for tracking objects. It initializes with a set number of particles, predicts their new states, updates their weights based on observed data, and resamples particles.

# main() Function:

Loads the YOLOv8 model.

Captures video frames.

Initializes the Particle Filter when an object is detected.

Continuously predicts, updates, and draws the tracked object on each frame.

Displays the result in a window and exits when 'q' is pressed.

# Results 

**Tracking**

https://github.com/user-attachments/assets/1e7ce912-afc2-4f76-ba10-61caff2061d7

**Detection**

![image](https://github.com/user-attachments/assets/7b9dad71-ebe3-4af0-856c-bf75730e4dc1)

# Flowchart

![XLHDSzCm4BthLoozD3qmmmmtUmOA2M4wKsf8-1YEmbet0baoqchIykjPIrKSuW8vo8vstzFRzqQ-32GznTw89xJ37g7NFRfRBuUTI2KXgYkhIKkZVo5yRQuRUESKcbfKpO0MFck53fPoeEYn5hg3HTvQQuCmjg1wUG5CR](https://github.com/user-attachments/assets/d26e6b3e-f482-4476-94b0-8c741bc8053c)


# Notes

Adjust the noise scale and alpha in the Particle Filter according to the specific application for better performance.

Ensure that the video file and model file paths are correctly set in the script.

The code assumes you are running in an environment with GUI support. If running in a headless environment, modify the script to save frames or output to files instead.
