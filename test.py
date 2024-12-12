import cv2
import os
from mtcnn import MTCNN
from keras_facenet import FaceNet
import numpy as np

os.chdir("C:\\Users\\şerefcanmemiş\\Documents\\Projects\\face_recognation_1")
embedder = FaceNet()
detector = MTCNN()

folderpath = "test_images"
image_list = os.listdir(folderpath)

known_embeddings = []
Ids = []
for image in image_list:
    img = cv2.imread(os.path.join(folderpath,image))

    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result = detector.detect_faces(img_rgb)
    if result:
        face_encoding = embedder.embeddings([img_rgb])[0]
        known_embeddings.append(face_encoding)
        Ids.append(os.path.splitext(image)[0])



cap = cv2.VideoCapture(0)
cap.set(3,680)
cap.set(4,460)

while True:
    ret, frame =cap.read()
    
    rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb_frame)

    for face in faces:
        x,y,w,h = face['box']
        face_encoding = embedder.embeddings([rgb_frame[y:y+h,x:x+h]])[0]

        name = "unkown"
        min_dist = float("inf")
        for id,known_embedding in zip(Ids,known_embeddings):
            dist = np.linalg.norm(face_encoding - known_embedding)
            if dist < min_dist:
                min_dist = dist
                name = id

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        
    cv2.imshow("Frame",frame)

    key =  cv2.waitKey(1)

    if key == 27: #esc
        break


cap.release()
cv2.destroyAllWindows()










