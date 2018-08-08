import numpy as np
import cv2
import os

print('loading cascades')
cascades = {}

for file in [f for f in os.listdir("./cascades/data/") if '.xml' in f][0:5]:
    print('loading '+file)
    cascades[file]=cv2.CascadeClassifier('./cascades/data/'+file)


print('loading camera stream')
cap = cv2.VideoCapture(0)

print('starting loop')
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for (name,cascade) in cascades.items():
        regions = cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        for (x,y,w,h) in regions:
            #print(x,y,w,h)
            color = (255,100,100) #BGR 0-255,
            stroke = 3
            font                   = cv2.FONT_HERSHEY_SIMPLEX
            fontScale              = 1
            fontColor              = (255,255,255)
            lineType               = 2
            #haarcascade_frontalcatface_extended.xml

            cv2.putText(frame,name[12:-4], (x,y), font, fontScale,fontColor,lineType)
            cv2.rectangle(frame, (x,y),(x+w,y+h),color, stroke)



    # Display the resulting frame
    cv2.imshow('frame',frame)
    #cv2.imshow('frame',cv2.flip(frame,1))
    #cv2.flip(img,1)
    #cv2.imshow('gray',gray)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
print('shutting down')
cap.release()
cv2.destroyAllWindows()
