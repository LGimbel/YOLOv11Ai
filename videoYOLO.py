from ultralytics import YOLO #Ultralytics is where you get the YOLO11 model

#The following commands are in case you do not have these libraries installed
import cv2 #cv2 is a library which has functions for computer vision and image processing
from PIL import Image #Stands for Python Imaging Library, used for opening and editing various image formats
import torch

# Check for CUDA device and set it
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

#load YOLO11 model file from current directory
model = YOLO("yolo11n.pt").to(device)

#Creates an object called "results" to store the information outputted from YOLO
#accepts all formats - image/dir/Path/URL/video/PIL/ndarray. source=0 for webcam
results = model.predict(source=0, show=True)  # Display preds. Accepts all YOLO predict arguments, can add show=true