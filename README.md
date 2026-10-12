
# Sign Language Detection System

A real-time Sign Language Detection System built using Python, OpenCV, MediaPipe, and Scikit-learn. The project uses computer vision and machine learning techniques to process hand gestures and classify them into corresponding labels, with the goal of converting recognized gestures into text.

## Overview

Sign language is an important communication method for people with hearing or speech impairments. This project explores how computer vision and machine learning can help recognize hand gestures through a camera.

The system follows a machine learning pipeline that includes image collection, dataset preparation, model training, and real-time inference.

## Features

* Real-time hand gesture recognition using a camera
* Image collection for custom gesture classes
* Dataset preparation and feature extraction
* Machine learning model training
* Gesture classification using a trained model
* Text-based output for recognized gestures, depending on the implemented labels

## Technologies Used

| Technology   | Purpose                                            |
| ------------ | -------------------------------------------------- |
| Python       | Core programming language                          |
| OpenCV       | Camera access and image processing                 |
| MediaPipe    | Hand detection and landmark tracking               |
| Scikit-learn | Machine learning model training and classification |

## Project Structure

```text
Sign-Language-detection-Python/
│
├── collect_imgs.py
├── create_dataset.py
├── train_classifier.py
├── inference_classifier.py
├── requirements.txt
├── README.md
│
├── data/                 # Collected images or dataset
└── model.p               # Trained model, if generated
```

*Note: The dataset and model filenames shown above are examples of generated outputs. Their actual names and locations depend on the implementation.*

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/usmanbashir16/Sign-Language-detection-Python.git
cd Sign-Language-detection-Python
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

The project uses these library versions:

```text
opencv-python==4.7.0.68
mediapipe==0.9.0.1
scikit-learn==1.2.0
```

**Compatibility note:** These are older library versions. Use a compatible Python version, such as Python 3.10, if installation fails with a newer Python release.

## How to Run

Run the scripts from the project root directory in the following order.

### Step 1: Collect Images

```bash
python collect_imgs.py
```

This script is intended to collect hand gesture images for the classes used by the project. Follow any prompts or keyboard instructions displayed by the script.

### Step 2: Create the Dataset

```bash
python create_dataset.py
```

This step prepares the collected data for machine learning, according to the preprocessing implemented in the script.

### Step 3: Train the Classifier

```bash
python train_classifier.py
```

This step trains the machine learning classifier using the prepared dataset. The trained model should be saved for use during inference.

### Step 4: Run Real-Time Detection

```bash
python inference_classifier.py
```

This script runs the prediction pipeline using the trained classifier and camera input, if configured in the implementation.

**Important:** Complete the data collection, dataset preparation, and training steps before running inference. Use the exact output paths and filenames defined in the Python scripts.

## Workflow

```text
Hand Gesture Images
        ↓
Image Collection
        ↓
Dataset Preparation
        ↓
Feature Extraction
        ↓
Model Training
        ↓
Trained Classifier
        ↓
Real-Time Camera Input
        ↓
Gesture Prediction
        ↓
Text Output
```

## Future Improvements

* Expand the dataset with more gesture classes and samples.
* Improve recognition under different lighting conditions.
* Add confidence scores and prediction smoothing.
* Build a user-friendly desktop or web interface.
* Extend the system to recognize complete words and dynamic gestures.
* Evaluate model performance using suitable classification metrics.

## Limitations

* Recognition depends on the gestures represented in the training dataset.
* Lighting, camera quality, and hand positioning may affect predictions.
* The system may require additional training data to support more gestures.
* Recognition of complete sign-language sentences requires additional modeling beyond individual gesture classification.

## Author

**Muhammad Usman**

GitHub: [usmanbashir16](https://github.com/usmanbashir16)

## License

No license has been specified for this repository yet. Please add a suitable license if you intend to permit reuse or redistribution.
