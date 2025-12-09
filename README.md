Text Detection and Recognition with YOLO + CRNN
A complete OCR (Optical Character Recognition) system that combines YOLO for text detection and CRNN for text recognition to extract text from images.
Overview
This project implements a two-stage pipeline:

Text Detection: Uses YOLO11 to detect text regions in images
Text Recognition: Uses CRNN (Convolutional Recurrent Neural Network) to recognize text within detected regions

Project Structure
.
├── combining_module.py              # Main pipeline combining YOLO + CRNN
├── Text_detection_module.ipynb      # YOLO training and detection notebook
├── Text_recognition_module.ipynb    # CRNN training notebook
├── Text_recognition_module.py       # CRNN implementation script
├── yolo11m.pt                       # YOLO11 medium model weights
├── yolo11n.pt                       # YOLO11 nano model weights
├── ocr_crnn.pt                      # Trained CRNN model weights
├── yolo_data/                       # YOLO training dataset
├── ocr_dataset/                     # CRNN training dataset
├── SceneTrialTrain/                 # Training data directory
└── runs/                            # Training runs and results
Features

Text Detection: Fast and accurate text region detection using YOLO11
Text Recognition: Character-level recognition using CRNN with CTC loss
End-to-End Pipeline: Complete solution from image input to text output
Multiple Model Sizes: Support for both YOLO11-nano and YOLO11-medium models
