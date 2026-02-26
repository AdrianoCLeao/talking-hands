# Talking Hands - Sign Language Recognition System

<img src="https://github.com/user-attachments/assets/36c6c711-d5f1-48e0-ae6f-c5ddf566440c" alt="Alt text" width="450">

A complete sign language recognition system using MediaPipe for hand tracking and TensorFlow/Keras for gesture classification.

This project aims to create a machine learning model that can identify, translate, and speak from live video of Libras (Brazilian Sign Language). It uses Google's Mediapipe to detect key points and Keras from TensorFlow to create the model.

## Project Structure

```
talking-hands/
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── capture/
│   │   │   ├── __init__.py
│   │   │   ├── capture_samples.py
│   │   │   └── create_keypoints.py
│   │   ├── models/
│   │   │   └── actions_18.keras
│   │   ├── train/
│   │   │   ├── __init__.py
│   │   │   ├── evaluate_model.py
│   │   │   ├── model.py
│   │   │   └── training_model.py
│   │   └── tts/
│   │       ├── __init__.py
│   │       └── text_to_speech.py
│   └── utils/
│       ├── __init__.py
│       ├── constants.py
│       └── utils.py
├── data/
│   └── keypoints/
├── main.py
├── mainwindow.ui
├── run.py
├── setup.py
├── requirements.txt
└── README.md
```

## Installation

### Method 1: Using pip (Recommended)

```bash
# Install the package in development mode
pip install -e .
```

### Method 2: Manual installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Main Application

```bash
# Using the run script
python run.py main

# Or directly
python main.py
```

### Training a Model

```bash
# Using the run script
python run.py train

# Or using the module
python -m src.core.train.training_model
```

### Creating Keypoints

```bash
# Using the run script
python run.py keypoints

# Or using the module
python -m src.core.capture.create_keypoints
```

### Capturing Samples

```bash
# Using the run script
python run.py capture

# Or using the module
python -m src.core.capture.capture_samples
```

## Package Usage

You can also import and use the modules directly:

```python
from src.core.train import training_model, get_model
from src.core.capture import capture_sample, create_keypoints
from src.core.tts import text_to_speech
from src.utils import constants, utils

# Train a model
model_path = "path/to/model.keras"
training_model(model_path, model_num=18)

# Create keypoints
create_keypoints("word_id", "frames_path", "output.h5")

# Text to speech
text_to_speech("Hello world")
```

## Dependencies

See `requirements.txt` for the complete list of dependencies. Main dependencies include:

- TensorFlow/Keras for machine learning
- MediaPipe for hand tracking
- OpenCV for computer vision
- PyQt5 for GUI
- NumPy and Pandas for data processing
- scikit-learn for machine learning utilities

## Features

- Real-time sign language recognition
- Model training with data augmentation
- Keypoint extraction from video frames
- Text-to-speech output
- GUI interface for easy interaction
- Modular package structure

## Development

To develop this package:

1. Clone the repository
2. Install in development mode: `pip install -e .`
3. Make your changes
4. Run tests and ensure everything works
  - `utils/utils.py`
  - `utils/constants.py`
  - `utils/text_to_speech.py`
- `models/`: Pre-trained models for gesture translation.
- `train/`: Scripts for model training.
  - `training_model.py`
  - `evaluate_model.py`
- `capture/`: Scripts for capturing and processing new data.
  - `capture_samples.py`
  - `create_keypoints.py`

## How to Use

1. **Create a Virtual Environment**

To isolate the project's dependencies, it is recommended to create a virtual environment using `venv`.

```bash
python -m venv venv
```
2. Activate the Virtual Environment
Activate the created virtual environment:
- On Windows:
```bash
venv\Scripts\activate
```
- On macOS and Linux:
```bash
source venv/bin/activate
```
3. Install the Requirements
With the virtual environment activated, install the project's requirements:
```bash
pip install -r requirements.txt
```
4. Run the Application
Execute the main.py file to start the GUI application:
```bash
python main.py
```
## Contributions
Feel free to contribute to this project. You can open issues and pull requests to discuss and implement improvements.

## License
This project is under the MIT license. Check LICENSE.md for more details.
