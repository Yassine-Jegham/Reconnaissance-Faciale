import cv2
import numpy as np
import time
import face_recognition
import os
from datetime import datetime


from gtts import gTTS
import playsound


"""import picamera                                 
# from PIL import ImageGrab
camera = picamera.PiCamera()
 # Ajustement des paramètres de capture (facultatif)
camera.resolution = (640, 480)  # Résolution de l'image
camera.framerate = 30  # Taux de rafraîchissement

    # Capture d'une image
path = 'ImagesAttendance'
camera.capture(path)

print("Image capturée : ", path) """

path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
## IMPORT OF IMAGES 
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)
 #ECODING IMAGES 
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
 
def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
 
#### FOR CAPTURING SCREEN RATHER THAN WEBCAM
##def captureScreen(bbox=(300,300,690+300,530+300)):
     #capScr = np.array(ImageGrab.grab(bbox))
     #capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
     #return capScr
 
encodeListKnown = findEncodings(images)
print('Encoding Complete')
 #USE CAMERA TO DETTCTED 
cap = cv2.VideoCapture(0)
 
while True:
    success, img = cap.read()
    #img = captureScreen()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
 
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
 
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        #print(faceDis)
        matchIndex = np.argmin(faceDis)
 
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)
            
            # Le message de code à convertir en voix
            code_message = "{} Devant toi ".format(name if name else '(no detections)')
            # Convertir le message de code en voix
            tts = gTTS(text=code_message, lang='fr')  # 'fr' pour le français
            tts.save('code_message.mp3')  # Enregistrer le fichier audio
            # Lire le fichier audio à travers le casque
            playsound.playsound('code_message.mp3')
 
    cv2.imshow('Webcam',img)
    cv2.waitKey(1)
    
