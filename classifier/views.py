import pickle
from django.shortcuts import render
from  django.core.files.storage import FileSystemStorage
from django.views.generic import TemplateView
from matplotlib import pyplot as plt 


#===============================================================================================
import os
import numpy as np 
import matplotlib.pyplot as plt 
import pickle
import random 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import cv2
import pandas as pd 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
# data = []
# categories = ["tumor","no_tumor"]
# for item in categories:
#     path = os.path.join(item)
#     label = categories.index(item)
#     for img in os.listdir(path):
#         imgpath = os.path.join(path,img)
#         cancer_image = cv2.imread(imgpath,0)
#         try:
#            cancer_image = cv2.resize(cancer_image, (50,50))
#            image = np.array(cancer_image).flatten()
#            data.append([image,label])
#         except Exception as err:
#             passpy manage.py

# pick_data = open("data1.pickle","wb")
# pickle.dump(data, pick_data)
# pick_data.close()



pick_data = open("data1.pickle","rb")
data = pickle.load(pick_data)
pick_data.close()
random.shuffle(data)
features = []
labels = []

for featuer,label in data:
    features.append(featuer)
    labels.append(label)
xtrain,xtest,ytrain,ytest = train_test_split(features,labels,test_size=0.2)
# model = SVC(C= 1,kernel='poly', gamma= 'auto')
# model.fit(xtrain,ytrain)

save_model = open(os.path.join('model.sav'),'rb')
# pickle.dump(model, save_model)
saved_model = pickle.load(save_model)
save_model.close()



#===============================================================================================

def index(request ):
    if request.method =='POST':
        
        image = request.FILES['image']
        fs = FileSystemStorage()
        fs.save(image.name, image)
        test_image = os.path.join('media',image.name)
        x = cv2.imread(test_image,0)
        x= cv2.resize(x,(50,50))
        ar = np.array(x).flatten()
        try:
            categories = ["tumor","no_tumor"]
            x = cv2.imread(test_image,0)
            x= cv2.resize(x,(50,50))
            ar = np.array(x).flatten()
            label = saved_model.predict([ar])
            global result 
            if label == 0:
                result = "This is a cancer tumor image"
            elif label ==1:
                result = "this is a benign tumor image"
            fs.delete(test_image)
            return render(request, 'index.html',{'result':result,"image":test_image})
           
        except Exception as err:
            fs.delete(test_image)
            return render(request, 'index.html',{'result':"bad image","image":test_image})

        

    return render(request, 'index.html')
# Create your views here.
