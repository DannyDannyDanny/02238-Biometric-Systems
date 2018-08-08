import numpy as np
import cv2
import os

print('loading cascades')
cascades = {}

for file in [f for f in os.listdir("./cascades/data/") if '.xml' in f]:
#    print('loading '+file)
    cascades[file]=cv2.CascadeClassifier('./cascades/data/'+file)

# SINGLE PHOTOS
print('counting hits for cascades')
cascade_hits = dict([(name,0) for (name,_) in cascades.items()])

somepic = "./orl_faces/s9/2.pgm"

frame = cv2.imread(somepic)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

for (name,cascade) in cascades.items():
    regions = cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    cascade_hits[name]=len(regions)
    #print(name + ' finds: ' + str(len(regions)))
    # for (x,y,w,h) in regions:
    #     #print(x,y,w,h)
    #     color = (255,100,100) #BGR 0-255,
    #     cv2.rectangle(frame, (x,y),(x+w,y+h),color, 3)


for (k,v) in cascade_hits.items():
    print(k+":"+str(v))

# def test():
#     for subject in [s for s in os.listdir("./orl_faces/") if "READ" not in s]:
#         for image in os.listdir("./orl_faces/"+subject):
#             print("loading ./orl_faces/"+subject+"/"+image)
#             img = cv2.imread("./orl_faces/"+subject+"/"+image)
#             cv2.imshow('image',img)
#             cv2.waitKey(20)
