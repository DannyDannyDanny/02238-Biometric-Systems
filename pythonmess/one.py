import numpy as np
import cv2

#####
#
#
#
#####
#print('loading face cascade')
face_cascade = cv2.CascadeClassifier('./cascades/data/haarcascade_frontalface_alt2.xml')

print('loading camera stream')
cap = cv2.VideoCapture(0)

print('starting loop')
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x,y,w,h) in faces:
        print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        color = (255,100,100) #BGR 0-255,
        stroke = 3
        cv2.rectangle(frame, (x,y),(x+w,y+h),color, stroke)

        #img_item = "my-image.png"
        #cv2.imwrite(img_item,roi_gray)


    # Display the resulting frame
    cv2.imshow('frame',frame)
    #cv2.imshow('gray',gray)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
print('shutting down')
cap.release()
cv2.destroyAllWindows()
