import numpy as np
import cv2
import os
import csv

path = "./MIT-CBCL-facerec-database/training-synthetic/"

def get_cascades_dict():
    # (cascadeName, cascadeObj) in dict
    names_cascades=['haarcascade_frontalface_alt.xml','haarcascade_frontalface_alt2.xml','haarcascade_frontalface_default.xml','haarcascade_profileface.xml']

    cascades = {}

    for file in [f for f in os.listdir("./cascades/data/") if f in names_cascades]:
    #    print('loading '+file)
        cascades[file]=cv2.CascadeClassifier('./cascades/data/'+file)

    return cascades
def get_image_path_list():
    return [path+s for s in os.listdir(path)]
def detection_quality_of_face_in_image_from_path(path,percentage):
    # returns between [0,1,2,3,4]
    image_frame = cv2.imread(path)
    image_quality = cv2.resize(image_frame, (0,0), fx=percentage, fy=percentage)
    image_gray = cv2.cvtColor(image_quality, cv2.COLOR_BGR2GRAY)

    detection_quality = 0
    for (name,cascade) in cascades.items():
        regions = cascade.detectMultiScale(image_gray, scaleFactor=1.5, minNeighbors=5)
        if len(regions) == 1:
            detection_quality += 1

    return detection_quality

cascades = get_cascades_dict()
image_paths = get_image_path_list()

results = {}
for quality in [i/100 for i in range(100,0,-5)]:
    q_list = []
    for image_path in image_paths:
        d_quality = detection_quality_of_face_in_image_from_path(image_path,quality)
        q_list.append(d_quality)

    detection_qualities = [4,3,2,1,0]

    results[quality] = [len([c for c in q_list if c==dq]) for dq in detection_qualities]


for (qual,freqs) in results.items():
    csvme = [qual] + freqs
    print(csvme)

with open('pythonmess/seven.csv', 'w') as csvfile:
    csvw = csv.writer(csvfile, delimiter=' ')
    headers = ['quality','dq4','dq3','dq2','dq1','dq0']
    csvw.writerow(headers)
    for (qual,freqs) in results.items():
        csvme = [qual] + freqs
        csvw.writerow(csvme)


exit()
for q in set(q_list):
    print(str(q)+':'+str(len([c for c in q_list if q==c])))
exit()
