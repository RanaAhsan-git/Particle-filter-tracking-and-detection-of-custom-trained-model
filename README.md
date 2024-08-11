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
# Labeling Dataset with LabelImg

LabelImg is an open-source graphical image annotation tool that is used to label objects in images, typically for object detection tasks. It supports various annotation formats such as YOLO and Pascal VOC.

**Installing LabelImg**

LabelImg can be installed using Python and is compatible with Windows, macOS, and Linux. 

**Install LabelImg:**

You can install LabelImg via pip. Open a terminal or command prompt and run:
```
pip install labelimg
```

Alternatively, you can install LabelImg directly from the source if you need the latest version or wish to contribute to development:
```
git clone https://github.com/tzutalin/labelImg.git
cd labelImg
pip install -r requirements/requirements.txt
python labelImg.py
```
# Using LabelImg to Label Your Dataset

**Open LabelImg:**

After installation, you can start LabelImg by running:
```
labelimg
````
If you installed from the source, run:
```
python labelImg.py
```
**Set Up Directories:**

**Open Directory:** Click on "Open Dir" to select the directory where your images are stored.

**Save Directory:** Click on "Change Save Dir" to select the directory where you want to save your annotations.

# **Label Images:**

**Create Annotation:** Click on the “Create RectBox” button (or press W) to start annotating. Draw a rectangle around the object you want to label.

**Enter Label:** After drawing the rectangle, a dialog will appear asking you to enter a label. Type the name of the object (e.g., "cat", "dog", "car").

**Save Annotations:** The annotations are automatically saved in the format you selected (YOLO or Pascal VOC). You can change the format by going to “View” > “Change Output Format” and selecting your preferred format.

**Navigate Through Images:** Use the arrow keys or the navigation buttons to move to the next or previous image in the folder.

**Finish Labeling:**

Continue labeling each image in your dataset. Once done, all annotations will be saved in the specified directory.
# Annotation Formats

**YOLO Format:**

Annotations are saved in .txt files, where each line represents an object in the format: <class_id> <x_center> <y_center> <width> <height>.

Coordinates are normalized to [0, 1].

# Launch LabelImg.

Open the directory containing your images.

Choose a save directory for annotations.

Start annotating each image by drawing bounding boxes around the objects and assigning labels.

Save annotations in your desired format.

# Additional Tips

**Class Names:** Maintain a consistent list of class names for labeling.

**Consistency:** Ensure annotations are accurate and consistent across your dataset for better model performance.

By following these steps, you can efficiently label your dataset for training object detection models using tools like YOLO.

# Training a Dataset with YOLOv8

# 1. Prepare Your Dataset

Before training, you need to prepare your dataset. YOLOv8 requires a specific format for annotations, which can be generated using tools like LabelImg.

**Dataset Structure**

**YOLOv8 typically requires the following directory structure:**
```
/dataset
    /images
        /train
            image1.jpg
            image2.jpg
            ...
        /val
            image1.jpg
            image2.jpg
            ...
    /labels
        /train
            image1.txt
            image2.txt
            ...
        /val
            image1.txt
            image2.txt
            ...
```

**Images:** JPEG or PNG files for training and validation.

**Labels:** YOLO format text files (.txt) for annotations where each line contains:

<class_id> <x_center> <y_center> <width> <height>

Coordinates are normalized to [0, 1].
Example of YOLO Format (.txt)
```
0 0.5 0.5 0.2 0.3
1 0.7 0.8 0.1 0.2
```
Here, 0 and 1 are class IDs, and the following values are normalized coordinates.

# 2. Install YOLOv8

YOLOv8 can be installed via pip. Make sure you have Python installed, then use:
```
pip install ultralytics
```
# 3. Prepare Configuration File

You need to create a configuration file to specify your dataset paths and hyperparameters. Create a YAML file (e.g., data.yaml) with the following content:
```
path: /path/to/your/dataset  # Path to your dataset
train: images/train
val: images/val

nc: 2  # Number of classes
names: ['class1', 'class2']  # List of class names
```
Replace /path/to/your/dataset with the actual path to your dataset directory, adjust nc (number of classes), and provide the names of your classes.

# 4. Train the Model

With YOLOv8 installed, you can train your model using the following command:
```
yolo train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640
```
**Parameters Explained:**

**model=yolov8n.pt:** The pre-trained YOLOv8 model to start with (YOLOv8 has different versions like yolov8n for nano, yolov8s for small, etc.).

**data=data.yaml:** Path to your YAML configuration file.

**epochs=50:** Number of training epochs.

**imgsz=640:** Image size (resolution) for training.

You can adjust these parameters based on your needs and available computational resources.

# 5. Monitor Training

The training process will generate logs and save checkpoints. Monitor the output for metrics like loss, precision, recall, and mAP (mean Average Precision). You can visualize the training progress using tools like TensorBoard or directly from the log files.

# 6. Evaluate and Test

After training, evaluate the model on your validation set to check its performance. YOLOv8 will save the best model weights based on the validation metrics.

# 7. Inference

To run inference on new images or videos, use the trained model with the following command:
```
yolo predict model=path/to/best_model.pt source=path/to/image_or_video
```

Replace path/to/best_model.pt with the path to your trained model and path/to/image_or_video with the path to the image or video file you want to test.

**Example Command for Training:**
```
yolo train model=yolov8s.pt data=data.yaml epochs=100 imgsz=640
```

This command trains the yolov8s model for 100 epochs with images resized to 640x640 pixels.

# Additional Tips for Yolo Training
**Data Augmentation:** Use data augmentation techniques to improve model robustness and generalization.

**Hyperparameter Tuning:** Adjust learning rates, batch sizes, and other hyperparameters based on your dataset and hardware.

**Pre-trained Models:** Using pre-trained models as a starting point can significantly reduce training time and improve performance.

By following these steps, you should be able to train a YOLOv8 model on your custom dataset and deploy it for object detection tasks.

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
