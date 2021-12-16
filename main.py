import numpy as np
import cv2
import time

lbpcascade = cv2.CascadeClassifier('lbpcascades/cascade3.xml')
# haarcascade = cv2.CascadeClassifier('/haarcascade_frontalface_default.xml')
vid = cv2.VideoCapture(0)

t1 = time.time()

while(True):
    ret, img = vid.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    license = lbpcascade.detectMultiScale(gray, 1.3, 5)
    # license = haarcascade.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in license:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_color = img[y:y+h, x:x+w]
        cv2.imshow('roi',roi_color)
    t2 = time.time()
    fps = int(1/(t2 - t1))
    
    cv2.putText(img, 'FPS: {0:.2f}'.format(fps), (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)
    cv2.imshow('img', img)
    print('FPS: {}'.format(fps))
    t1 = time.time()
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()