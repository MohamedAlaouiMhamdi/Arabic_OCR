# Optical Character Recognition (OCR) Project

This project is a deep learning-based Optical Character Recognition (OCR) system designed to extract and transcribe text from images, with a particular emphasis on supporting Arabic script transcription. It leverages convolutional neural networks (CNNs) for image feature extraction and sequence prediction techniques for text recognition.

## Table of Contents

- [Overview](#overview)
- [Project Objectives](#project-objectives)
- [Technical Approach](#technical-approach)
- [Model Architecture](#model-architecture)
- [Data Preparation](#data-preparation)
- [Training Process](#training-process)
- [Evaluation](#evaluation)
- [GUI Implementation](#gui-implementation)
- [Challenges and Solutions](#challenges-and-solutions)
- [Future Directions](#future-directions)
- [Conclusion](#conclusion)

## Overview

The OCR project aims to build a robust system capable of recognizing and transcribing text from images, particularly focusing on Arabic language support. This system could be used in various applications, from digitizing printed documents to enabling text recognition in real-time camera feeds.

## Project Objectives

- **Text Extraction**: Develop a model that can accurately extract text from images.
- **Arabic Script Support**: Provide accurate transcription of recognized text into Arabic script.
- **Real-Time Prediction**: Enable the system to make predictions in real-time through a user-friendly interface.
- **Model Generalization**: Ensure the model performs well across different fonts, sizes, and styles of text.

## Technical Approach

### Data Preprocessing

The input images undergo several preprocessing steps to prepare them for training and prediction:

- **Resizing**: Images are resized to a standard size to ensure consistency in the input shape.
- **Normalization**: Pixel values are normalized to fall within a specific range, typically between 0 and 1, to stabilize the training process.
- **Data Augmentation**: Techniques such as rotation, translation, and scaling are applied to augment the training data, improving the model's robustness.

### Model Architecture

The OCR model is designed to balance complexity and efficiency. The architecture consists of the following components:

- **Convolutional Layers (CNN)**: These layers are responsible for extracting high-level features from the input images, such as edges, textures, and character shapes.
- **Recurrent Layers (RNN/LSTM/GRU)**: Following the CNN layers, recurrent layers are employed to capture the sequential nature of the text, handling the varying lengths of text sequences.
- **CTC (Connectionist Temporal Classification) Layer**: The final output layer uses CTC loss, which is ideal for sequence prediction tasks where the alignment between input and output is unknown.
  
<img width="578" alt="Screenshot 2024-09-07 113111" src="https://github.com/user-attachments/assets/f1d0fbe8-47db-4ccb-91ab-4c519105a5c4">

<img width="614" alt="Screenshot 2024-09-07 113141" src="https://github.com/user-attachments/assets/450f1c9c-9906-417d-a09c-7362e3be6b32">


### Data Preparation

IFN/ENIT-database

<img width="562" alt="Screenshot 2024-09-07 114423" src="https://github.com/user-attachments/assets/1b10a94d-7817-491f-b65e-fe12a83e9066">

<img width="483" alt="Screenshot 2024-09-07 113750" src="https://github.com/user-attachments/assets/e3403124-3e07-4d23-bde6-2e1dffe491c4">

for more information check out the dataset pdf or visit this link : http://www.ifnenit.com/
### Training Process

The training process involves:

1. **Batch Processing**: Training data is processed in batches to efficiently utilize computational resources.
2. **Loss Calculation**: The CTC loss function is used, which allows the model to predict sequences of characters without needing a predefined alignment between input images and output text.
3. **Optimization**: An optimizer like Adam or RMSprop is employed to minimize the loss, adjusting the weights of the network through backpropagation.
<img width="735" alt="Screenshot 2024-09-07 115353" src="https://github.com/user-attachments/assets/841dd8de-8f45-4e01-b011-c0f9ffee8897">

<img width="601" alt="Screenshot 2024-09-07 115447" src="https://github.com/user-attachments/assets/e3376c0e-0f80-4426-9860-1427e0d99af9">

### Evaluation

The model's performance is evaluated based on its ability to correctly predict the text in the test images. Metrics such as accuracy, character error rate (CER), and word error rate (WER) are used to quantify performance.

<img width="369" alt="Screenshot 2024-09-07 120420" src="https://github.com/user-attachments/assets/ac2990ed-cec3-4dc3-a6ca-c2be6cc4a47a">

<img width="306" alt="Screenshot 2024-09-07 120441" src="https://github.com/user-attachments/assets/e659467f-6b7f-445a-9b16-e318eb35ab5a">

### GUI Implementation

A graphical user interface (GUI) is developed using Tkinter to allow users to interact with the OCR model:

- **Image Upload**: Users can upload images containing text.
- **Prediction Display**: The predicted text and its Arabic transcription are displayed.
- **Real-Time Feedback**: The system provides immediate feedback on the uploaded image, showing the recognized text.

## Challenges and Solutions

- **Character Overlap**: One of the challenges was dealing with characters that overlap or touch in cursive fonts, particularly in Arabic. This was mitigated by refining the preprocessing steps and tuning the CNN architecture.
- **Sequence Length Variation**: The model needed to handle varying lengths of text sequences, which was addressed using the CTC loss function, allowing the model to flexibly predict sequences without requiring fixed alignment.

## Future Directions

- **Multi-Language Support**: Expanding the OCR system to support multiple languages beyond Arabic.
- **Improved Accuracy**: Experimenting with more complex architectures, such as Transformer-based models, to improve recognition accuracy.
- **Deployment**: Packaging the model and GUI into a deployable application for wider use.
