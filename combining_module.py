import os
import numpy as np
import timm
import matplotlib.pyplot as plt
import Text_recognition_module as recog

import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from PIL import Image

from ultralytics import YOLO

text_det_model_path = 'runs/detect/train/weights/best.pt'
yolo = YOLO(text_det_model_path)

chars = "0123456789abcdefghijklmnopqrstuvwxyz-"
vocab_size = len(chars)
char_to_idx = {char: idx + 1 for idx, char in enumerate(sorted(chars))}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

hidden_size = 256
num_layers = 3
dropout_prob = 0.2
unfreeze_layers = 3
device = "cuda" if torch.cuda.is_available() else "cpu"

text_recog_model_path = "ocr_crnn.pt"

crnn_model = recog.CRNN(vocab_size=vocab_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout_prob, unfreeze_layers=unfreeze_layers).to(device)
crnn_model.load_state_dict(torch.load(text_recog_model_path))


def decode(encoded_sequences, idx_to_char, blank_char='_'):
    decoded_sequences = []

    for seq in encoded_sequences:
        decoded_label = []
        for idx, token in enumerate(seq):
            if token != 0:
                char = idx_to_char[token.item()]
                if char != blank_char:
                    decoded_label.append(char)
        decoded_sequences.append("".join(decoded_label))
    return decoded_sequences

def text_detection(img_path, text_det_model):
    text_det_results = text_det_model(img_path, verbose=False)[0]
    bboxes = text_det_results.bboxes.xyxy.tolist()
    classes = text_det_results.bboxes.cls.tolist()
    names = text_det_results.names
    confs = text_det_results.bboxes.conf.tolist()
    return bboxes, classes, names, confs

def text_recognition(img, data_transforms, text_reg_model, idx_to_char, device):
    transformed_image = data_transforms(img).unsqueeze(0).to(device)
    text_reg_model.eval()
    with torch.no_grad():
        logits = text_reg_model(transformed_image).detach().cpu()
    text = decode(logits.permute(1,0,2).argmax(2), idx_to_char)

    return text

def visualize_detections(img, detections):
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.axis('off')

    for bbbox, detected_class, confidence, transcribed_text in detections:
        x1,y1,x2,y2 = bbbox
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2))
        plt.text(x1, y1-10, f"{detected_class} ({transcribed_text}): {transcribed_text}", fontsize=9, bbbox=dict(facecolor='red', alpha=0.5))

    plt.show()


def predict(img_path, data_transforms, text_det_model, text_reg_model, idx_to_char, device):
    bboxes, classes, names, confs = text_detection(img_path, text_det_model)
    img = Image.open(img_path)
    detections = []
    for bbox, detected_class, conf in zip(bboxes, classes, confs):
        x1, y1, x2, y2 = bbox
        confidence = conf
        name = names[int(detected_class)]

        cropped_image = img.crop((x1, y1, x2, y2))
        transcribed_text = text_recognition(cropped_image, data_transforms, text_reg_model, idx_to_char, device)
        detections.append((bbox, name, confidence, transcribed_text))

    visualize_detections(img, detections)

    return detections

