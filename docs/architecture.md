
# Requirements Document for LIBRAS to Audio Transcription Project

## Project Overview
This project leverages neural networks and computer vision to interpret LIBRAS signs in real-time and transcribe them into audio in Portuguese. Modern frameworks such as MediaPipe for body landmark extraction and TensorFlow/Keras for sign classification will be used. The objective is to create a native application that is OS-agnostic, ensuring compatibility with Windows, macOS, and Linux.

---

## Functional Requirements
1. **Real-Time Recognition:**
   - The system must capture LIBRAS signs in real-time using the device’s webcam.
   - Process and classify signs within 200 ms.

2. **Transcription to Text and Audio:**
   - Generate text transcription for recognized signs.
   - Convert text transcription to Portuguese audio using Text-to-Speech (TTS).

3. **Multimodal Integration:**
   - Support for input via hand signs, facial expressions, and body movements.

4. **Model Training and Updates:**
   - Enable incremental training with new sign data.

5. **User Interface:**
   - Provide a graphical interface for system configuration, new sign training, and monitoring.

6. **Export and Sharing:**
   - Allow exporting transcribed text and audio in standard formats (TXT, MP3).

7. **Compatibility:**
   - Support for Windows, macOS, and Linux operating systems.

---

## Non-Functional Requirements
1. **Performance:**
   - Maximum latency of 200 ms per iteration.
2. **Security:**
   - Ensure that captured data is not shared without user consent.
3. **Scalability:**
   - The system should support adding new signs without rewriting the base model.
4. **Portability:**
   - Modular structure for deployment in edge computing environments.

---

## Technology Stack

1. **Graphical User Interface:**
   - Framework: Electron for cross-platform development.
   - GUI Toolkit: GTK+ or Qt for native graphical interfaces.

2. **Data Processing:**
   - Language: Python for data manipulation and machine learning models.
   - Frameworks: MediaPipe for landmark extraction and TensorFlow/Keras for classification.

3. **TTS (Text-to-Speech):**
   - Framework: Google TTS or Speech Synthesis Markup Language (SSML).

4. **Infrastructure:**
   - Dependency Management: Conda or Virtualenv.
   - Build and Packaging: PyInstaller for generating native executables.

---

## Monthly Deliverables Plan (6 Months)

### Month 1: Initial Setup
- Set up the development environment.
- Define project structure and primary libraries.
- Implement initial video capture system using MediaPipe.

### Month 2: Recognition Pipeline
- Develop the landmark extraction pipeline.
- Train an initial CNN model for basic sign classification.
- Evaluate model performance using small datasets.

### Month 3: Transcription System
- Integrate the classification system with text transcription.
- Implement modules for managing transcription data.

### Month 4: Audio Conversion
- Integrate TTS libraries for audio generation.
- Optimize transcription pipeline to reduce latency.

### Month 5: User Interface
- Develop graphical interface with Electron and GTK+/Qt.
- Create a dashboard for transcription visualization and export.

### Month 6: Testing and Distribution
- Perform integration and load testing across different operating systems.
- Generate native executables for Windows, macOS, and Linux.
- Finalize documentation and user training materials.

---

## Directory Structure

```
project-root/
├── src/
│   ├── core/
│   │   ├── models/  # Machine Learning models and definitions
│   │   ├── processing/  # Data processing logic
│   │   └── tts/  # Text-to-Speech logic
│   ├── gui/  # Graphical User Interface
│   │   ├── components/  # Reusable components
│   │   ├── windows/  # Main application windows
│   │   └── assets/  # Visual resources
│   └── utils/  # Helper functions
├── data/
│   ├── datasets/  # Training datasets
│   └── preprocessed/  # Processed data
│   └── keypoints/
├── docs/
├── tests/
│   ├── unit/  # Unit tests
│   └── integration/  # Integration tests
├── build/  # Generated files for distribution
└── scripts/  # Auxiliary scripts
```
