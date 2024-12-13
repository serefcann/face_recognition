# Face Recognition with Pre-Trained Dataset

This project demonstrates real-time face recognition using a webcam. The `test.py` script detects faces in the video feed and identifies individuals that have been pre-trained in the system. When a recognized face is detected, the corresponding name is displayed above the bounding box around the face.

## Features
- Real-time face detection using a webcam.
- Recognition of pre-trained individuals.
- Displays the name of the recognized person on the video feed.

## How It Works
1. **Pre-Training**: The system is trained with images of individuals using their labeled data.
2. **Face Detection**: The webcam captures the video feed, and faces are detected in real-time.
3. **Face Recognition**: Detected faces are compared with the pre-trained dataset, and the names of recognized individuals are displayed.

## Requirements
- Python 3.8 or later
- Libraries:
  - OpenCV
  - NumPy
  - MTCNN
  - keras-facenet

## How to Use
   git clone https://github.com/yourusername/your-repo.git
   cd face_recognation_1
   pip install -r requirements.txt
   python test.py
   (dont forget you must create test_images folder and upload images into the folder)

