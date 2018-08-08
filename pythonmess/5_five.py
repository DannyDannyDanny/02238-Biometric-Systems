import numpy as np
import cv2
import os
import json

a = {}

for subject in [s for s in os.listdir("./orl_faces/") if "READ" not in s]:
    print(subject)
    a[subject] = []
    for image in os.listdir("./orl_faces/"+subject):
        # image in and grayscale
        frame = cv2.imread("./orl_faces/"+subject+"/"+image)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # face detection
        cascade = cv2.CascadeClassifier('./cascades/data/haarcascade_frontalface_default.xml')
        regions = cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

        a[subject].append((image,str(len(regions))))
        # if str(len(regions)) != '1':
        #     print(subject+"/"+image+":"+str(len(regions)))

print(a)

with open('./data.txt', 'w') as outfile:
    json.dump(a, outfile)
