import cv2
import glob
import dlib
import numpy as np
import pandas as pd

global j
j=0
global totalx
totalx=[]

global totaly

totaly=[]

global totald

totald=[]

location = 'shape_predictor_68_face_landmarks.dat_2'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(location)
data=[]
def convertingTo_numpyarray(shape, dtype="int"):

    coordinates = np.zeros((68, 2), dtype=dtype)

    for i in range(0, 68):
        coordinates[i] = (shape.part(i).x, shape.part(i).y)

    return coordinates


def get_landmarks(image):
    global j
    frame_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    w, h=frame_grey.shape
    width=230
    height=int(h*(width/w))
    shrink=(width/w)
    frame_resized = cv2.resize(frame_grey,(height,width),interpolation=cv2.INTER_AREA)

    
    dets = detector(frame_resized)
    if len(dets) > 0:
        for k, d in enumerate(dets):
            shape = predictor(frame_resized,d)
            #print(k)
            #print(d)
            
            listx=[]
            listy=[]
            distance=[]
            for i in range(0,68):
                
                listx.append(float(shape.part(i).x))
                
                listy.append(float(shape.part(i).y))
            
            #print('x coordinate')
            meanpx=np.mean(listx)
           
            meanpy=np.mean(listy)
            #print(listx)
            #print('y coordinate')
            #print(listy)
            #print(meanpx)
            #print(meanpy)
            totalx.append(listx)
            totaly.append(listy)
            

            cv2.circle(image, (int(meanpx/shrink), int(meanpy/shrink)), 3, (0, 0, 0), -1)
            meannp=np.asarray((meanpx,meanpy))
           # print(meannp)
            for z,w in zip(listx,listy):
                cord=np.asarray((z,w))
                #print(cord)
                dist=np.linalg.norm(cord-meannp)
                #print(dist)
                distance.append(float(dist))
            #print('distances')
            #print(distance)
            totald.append(distance)
            
            for h,m in zip(listx,listy):
                h=h+1
                m=m+1
                cv2.line(image,(int(h/shrink), int(m/shrink)),(int(meanpx/shrink), int(meanpy/shrink)),(0, 0, 255),1)
                  
              
            
            shape = convertingTo_numpyarray(shape)
            for (x, y) in shape:
                cv2.circle(image, (int(x/shrink), int(y/shrink)), 3, (255, 255, 255), -1)
    
    print(listx)
        
    if(j==4):
        print(totalx)
        df=pd.DataFrame(columns='emotion x y d'.split())
        for i in range(0,len(totalx)):
            df=df.append({'emotion':'a','x':totalx[i]},ignore_index=True)
            df.to_csv('Book1.csv',encoding='utf-8',index=True)    
                                          
    j=j+1    
      
  

    

def main():

    images=glob.glob("*.png")
    count=0       
    z=[]
    w=[]
    dist=[]   
    
    for image in images:
        count=count+1
        img=cv2.imread(image)
        get_landmarks(img)
      

      
        
main()        
        
        
        
    
    
    
    
    
    


    
