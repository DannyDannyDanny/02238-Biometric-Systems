import numpy as np
import cv2
import os
import json

def getCascadesDict():
    # (cascadeName, cascadeObj) in dict
    names_cascades=['haarcascade_frontalface_alt.xml','haarcascade_frontalface_alt2.xml','haarcascade_frontalface_default.xml','haarcascade_profileface.xml']

    cascades = {}

    for file in [f for f in os.listdir("./cascades/data/") if f in names_cascades]:
    #    print('loading '+file)
        cascades[file]=cv2.CascadeClassifier('./cascades/data/'+file)

    return cascades

path = "./MIT-CBCL-facerec-database/training-synthetic/"

print('loading cascades')

cascades = getCascadesDict()

cascade_hits = dict([(name,0) for (name,_) in cascades.items()])
faces_regions = {}

for s in os.listdir(path):
    print(s)
    frame = cv2.imread(path+'/'+s)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces_regions[s] = []

    for (name,cascade) in cascades.items():
        regions = cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        faces_regions[s].append((name,len(regions)))

        cascade_hits[name]=cascade_hits[name]+len(regions)

with open('./dateset2_which_faces.txt', 'w') as outfile:
    json.dump(faces_regions, outfile)

exit()

with open('./dateset2_which_cascades_2.txt', 'w') as outfile:
    json.dump(cascade_hits, outfile)

exit()

### NICE AND SIMPLE CHECK ALL WITH SINGLE CASCADE

path = "./MIT-CBCL-facerec-database/training-synthetic/"

for s in os.listdir(path):
    #print(s)
    frame = cv2.imread(path+'/'+s)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # face detection
    cascade = cv2.CascadeClassifier('./cascades/data/haarcascade_frontalface_default.xml')
    regions = cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    if len(regions) != 1:
        print(s+":"+str(len(regions)))
