# import csv
# with open('pythonmess/results.csv', 'w') as csvfile:
#     csvw = csv.writer(csvfile, delimiter=' ')
#     headers = ['filename','resolution',]
#     csvw.writerow(['1','0','3'])
#     csvw.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])

import json

if True:
    data = ''
    with open('./dateset2_which_faces.txt', 'r') as file:

        data = file.read().strip()

    a = json.loads(data)


    for (k,v) in [(k,v) for (k,v) in a.items()]:
        number_of_detections = sum([val[1] for val in v])
        number_of_well_detected_features = len([val for val in v if val[1]==1])

        #if number_of_detections != number_of_well_detected_features:
        if number_of_well_detected_features == 0:
            print(k)
            # for val in v:
            #     print('--'+str(val))
        #break


if False:
    data = ''
    with open('./dateset2_which_cascades.txt', 'r') as file:

        data = file.read().strip()

    a = json.loads(data)


    for (k,v) in [(k,v) for (k,v) in a.items() if v > 1000]:
        print(str(v)+':'+str(k))


exit()
data = ''
with open('./orl_faces_that_couldnt_be_detected.txt', 'r') as file:
    data = file.read().strip()

a = json.loads(json.loads(data))

for (k,v) in a.items():
    print(k)
    for value in [val for val in v if val[1]!='1']:
        # print("-"+str(value))
        print("-"+str(value))
