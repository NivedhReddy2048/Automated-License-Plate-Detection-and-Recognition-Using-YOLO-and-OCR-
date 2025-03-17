# Automated-License-Plate-Detection-and-Recognition-Using-YOLO-and-OCR-


This project focuses on detecting and recognizing license plates from images and videos using YOLOv8 and Optical Character Recognition (OCR).

🚀 Features

License Plate Detection using YOLOv8

OCR (Optical Character Recognition) using Tesseract and EasyOCR

Preprocessing for better text recognition

Bounding Box Visualization for detected plates

🛠 Installation

Run the following commands to install the required dependencies:

pip install ultralytics opencv-python numpy torch torchvision torchaudio
pip install easyocr pytesseract matplotlib

📂 Dataset Setup

Dataset:
https://www.kaggle.com/datasets/ronakgohil/license-plate-dataset
Upload your dataset ZIP file to Colab.

Extract it using:

import zipfile
zip_path = "/content/Number plate dataset.zip"
extract_dir = "/content/number_plate_dataset"
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
print("✅ Dataset extracted successfully!")

Modify dataset.yaml with correct paths:

yaml_content = f"""
train: {extract_dir}/archive/images/train
val: {extract_dir}/archive/images/val

nc: 1  # Number of classes
names: ['license_plate']
"""
yaml_path = f"{extract_dir}/archive/dataset.yaml"
with open(yaml_path, "w") as file:
    file.write(yaml_content)
print("✅ Updated dataset.yaml with correct paths.")

🎯 Training YOLOv8

from ultralytics import YOLO
model = YOLO("yolov8n.yaml")
model.train(data=f"{extract_dir}/archive/dataset.yaml", epochs=50, imgsz=640)

🛠 Inference and Visualization

import cv2
import matplotlib.pyplot as plt

model = YOLO("/content/runs/detect/train/weights/best.pt")
test_image_path = "/content/number_plate_dataset/archive/images/train/sample.jpg"
results = model(test_image_path)
res_plotted = results[0].plot()
plt.imshow(res_plotted)
plt.axis("off")
plt.show()

🔍 OCR (Text Extraction)

import cv2
import pytesseract
from PIL import Image

image = cv2.imread(test_image_path)
results = model(test_image_path)

for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        license_plate = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(thresh, config='--psm 7')
        print("🔹 Detected License Plate Number:", text.strip())

📌 Future Improvements

Improve OCR accuracy using more advanced preprocessing techniques

Train with a larger dataset for better performance

Implement real-time detection from live video streams



