import cv2 as cv
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from keras_facenet import FaceNet
import joblib

facenet = FaceNet()
faces_embedding = np.load(r"C:\Users\Administrator\Desktop\DetectionModels\SVM_FACE_DETECTION_REALTIME\model\faces_embeddings_done_4classes.npz")
Y = faces_embedding['arr_1']

encoder = LabelEncoder()
encoder.fit(Y)
haarcascade = cv.CascadeClassifier(r"C:\Users\Administrator\Desktop\DetectionModels\SVM_FACE_DETECTION_REALTIME\model\haarcascade_frontalface_default.xml")
model = joblib.load(r"C:\Users\Administrator\Desktop\DetectionModels\SVM_FACE_DETECTION_REALTIME\SVM_model.joblib")

cap = cv.VideoCapture(0)

while cap.isOpened():
    _,frame = cap.read()
    rgb_img = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    gray_img = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    faces = haarcascade.detectMultiScale(gray_img,1.3,5)
    for x,y,w,h in faces:
        img = rgb_img[y:y+h,x:x+w]
        img = cv.resize(img,(160,160))
        img = np.expand_dims(img,axis=0)
        ypred = facenet.embeddings(img)
        face_name = model.predict(ypred)
        final_name = encoder.inverse_transform(face_name)
        print(final_name)
        # cv.rectangle(frame,(x,y-10),(x+w,y+h),(255,0,255),10)
        border_thickness = 2
        cv.rectangle(frame,(x,y-10),(x+w,y+h),(255,0,255),border_thickness)
        # cv.putText(frame,str(final_name),(x,y-20),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),cv.LINE_AA)
        font_scale = 0.8
        font_thickness = 1
        (text_width, text_height), baseline = cv.getTextSize(str(final_name), cv.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        cv.putText(frame, str(final_name), (x, y - 20), cv.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness)

    cv.imshow("Face Recognition:",frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()