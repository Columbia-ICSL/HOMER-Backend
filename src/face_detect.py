import cv2
import os
import numpy as np
import time
from settings import Export


def pre_process(data_path, clf_file, results_path, mode='photo'):
    
    face_cascade = cv2.CascadeClassifier(clf_file) 
    
    tik = time.time()

    if mode == 'photo':
        for i, filename in enumerate(os.listdir(data_path)):  
            if filename.split(sep='.')[-1] != 'png':
                continue
            img = load_image(data_path, filename)
            face_img = face_detection(img, face_cascade)
            face_img_resized = resize_images(face_img, 224, 224)
            cv2.imwrite(results_path+'test{:d}.jpg' .format(i),face_img_resized)
        print("Face detection time: ", time.time()-tik)
   
    else:
        raise ValueError("The chosen mode is invalid")



    






def main():
    data_path = 'Dataset/IMG_0725.mov' #if mode == photo it's a folder, otherwise it's the file 
    #data_path = 'Dataset/photos'
    clf_file = 'haarcascade_frontalface_default.xml'
    results_path='Dataset/results/'
    
    pre_process(data_path, clf_file, results_path, mode='video')

    
        
    
if __name__ == "__main__":
    main()