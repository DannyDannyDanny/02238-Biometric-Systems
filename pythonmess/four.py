import numpy as np
import cv2
import os

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

print('loading cascades')
cascades = {}

for file in [f for f in os.listdir("./cascades/data/") if '.xml' in f]:
#    print('loading '+file)
    cascades[file]=cv2.CascadeClassifier('./cascades/data/'+file)

cascade_hits = dict([(name,0) for (name,_) in cascades.items()])

count = 0
for subject in [s for s in os.listdir("./orl_faces/") if "READ" not in s]:
    for image in os.listdir("./orl_faces/"+subject):

        # making scannable
        frame = cv2.imread("./orl_faces/"+subject+"/"+image)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for (name,cascade) in cascades.items():
            regions = cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
            cascade_hits[name]=cascade_hits[name]+len(regions)


            ## MAKE IT WORK

            for (x,y,w,h) in regions:
                #print(x,y,w,h)
                roi_gray = gray[y:y+h, x:x+w]

                color = (0,0,0) #BGR 0-255,
                stroke = 2
                cv2.rectangle(gray, (x,y),(x+w,y+h),color, stroke)

                font                   = cv2.FONT_HERSHEY_SIMPLEX
                fontScale              = 0.5
                lineType               = 2
                cv2.putText(gray,name[12:-4], (x,y), font, fontScale,color,lineType)


        ensure_dir('./sdags/'+subject+'/')
        cv2.imwrite('./sdags/'+subject+'/'+image,gray)


print('results:')
for (k,v) in cascade_hits.items():
    print(k+":"+str(v))

print(count)
