import cv2
import os


os.chdir("C:\\Users\\şerefcanmemiş\\Documents\\Projects\\face_recognation_1")


cap = cv2.VideoCapture(0)
cap.set(3,680)
cap.set(4,460)
os.chdir("C:\\Users\\şerefcanmemiş\\Documents\\Projects\\face_recognation_1")
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
profileFaceCascade=cv2.CascadeClassifier("haarcascade_profileface.xml")

while True:
    ret, frame =cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    profileFaces = profileFaceCascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=6,minSize=(100,100))
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=6,minSize=(100,100))

    for x,y,w,h in profileFaces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,0),thickness=2)
        cv2.putText(frame,"can",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,2)
        
    for x,y,w,h in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,0),thickness=2)
        cv2.putText(frame,"can",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,2)
        
    cv2.imshow("Frame",frame)

    key =  cv2.waitKey(1)

    if key == 27: #esc
        break


cap.release()
cv2.destroyAllWindows()


