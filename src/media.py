import os
import cv2
from settings import Export
import numpy as np
from emo_detect import EmoDetector
#import matplotlib.pyplot as plt
from img_processing import face_detection
import matplotlib.pyplot as plt
import subprocess
from openpyxl import load_workbook



class Video(object):
    def __init__(self, frames, duration, path=None, name='video', fps=5):
        self.frames=frames
        self.name=name
        self.duration = duration
        self.successful_loading = 1
        if path is not None:
            self.name = path.split('/')[-1].split('.')[0]
            self.successful_loading = self.load_frames(path, fps)
   
        elif frames:
            self.width=frames[0].pix_vals.shape[1]
            self.height=frames[0].pix_vals.shape[0]
        self.emotions = EmoDetector()


    


    def export(self, path, width=None, height=None, codec='mp4v', exp_format='mp4', fps=20, start=-1, end=-1):
        
        if width is None or height is None:
            width=self.width
            height=self.height
        else:
            self.resize(width, height)
        
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        file_path = os.path.join(path, self.name)+'.'+exp_format

        out = cv2.VideoWriter(file_path, fourcc, fps, (width,height))
        
       # 
        if start == -1 and end == -1:
            frames = self.frames
        else:
            frames = self.frames[start-1:end]
            
        for i, frame in enumerate(frames):
            out_img = frame.pix_vals
            out.write(out_img)
            
        out.release()
        cv2.destroyAllWindows()


    def frames_export(self, path):
        path=os.path.join(path, self.name)
        if os.path.isdir(path) == False:
            os.mkdir(path)
        
        for i, frame in enumerate(self.frames):
            frame.export(path)

    
    def load_frames(self, path, fps, record_start_time=0, time_limit=None):
        cap = cv2.VideoCapture(path)
        sec=0
        i=0
        if time_limit is None:
            time_limit = self.duration
        while(cap.isOpened()):
            sec = sec + 1/fps
            sec = round(sec, 2)
            cap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
            ret, frame = cap.read()
            #print(sec)
            if ret==True and sec <= time_limit:
                if sec > record_start_time:
                    i += 1
                    if i == 1:
                        
                        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
                        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
                    frame_name = "frame_{:d}" .format(i)
                    self.frames.append(Image(frame, name=frame_name))
                    #print(frame)

            else:
                break
            
        if i == 0:
            return 0
        else:
            print("Video "+self.name+" successfully imported.")
        cap.release()
        cv2.destroyAllWindows()
        return 1
    
    def face_detect(self, mode='crop', model='both'):
        for i in range(len(self.frames)):
            self.width, self.height = self.frames[i].face_detect(mode, model)


    
    def emotions_prediction(self):
        for frame in self.frames:
            if frame.face_detected == True:
                frame.resize(48, 48)
                #frame.resize(64, 64)
                frame.convert_to_gray()
                self.emotions.predict_emotion(frame.pix_vals[np.newaxis, :, :, np.newaxis])
                
                #self.emotions.predict_emotion(frame.pix_vals[np.newaxis, :, :, :])
            else:
                #self.emotions.preds7.append(np.asarray([-0.1]*7))
                self.emotions.preds7.append(np.asarray([-0.1]*7))
                self.emotions.best_guess7.append("No face")
        

                
        
    def resize(self, width, height):
        for i in range(len(self.frames)):
            self.frames[i].resize(width, height)
            
    def rotate(self, angle):
        if angle%90 != 0:
            raise ValueError("The rotation angle must be a multiplier of 90")
        for i in range(len(self.frames)):
            self.frames[i].rotate(angle) 
            
            
    def split(self, interclips=False):
        last_frame_was_b_and_w=True
        clips_start = []
        clips_end = []
        
        for frame_nb, frame in enumerate(self.frames):
            #if not clips_start:
             #   if np.sum(frame.pix_vals) > 20000000:
              #      clips_start.append(frame_nb)
            h=frame.pix_vals.shape[0]
            w=frame.pix_vals.shape[1]
            img_norm = frame.pix_vals/255
            nb_pixels= h*w
            white_sum=(np.sum(img_norm[:int(h/4)])+np.sum(img_norm[int(3*h/4)+1:]))/3
            black_sum=np.sum(img_norm[int(h/4)+1:int(3*h/4)])/3
            
            if black_sum < nb_pixels*0.1/2 and white_sum > nb_pixels*0.9/2 and last_frame_was_b_and_w == False:
                if interclips == False:
                    clips_end.append(frame_nb)
                last_frame_was_b_and_w=True
            if (black_sum > nb_pixels*0.1/2 or white_sum < nb_pixels*0.9/2) and last_frame_was_b_and_w == True:
                clips_start.append(frame_nb)
                if frame_nb and interclips == True:
                    clips_end.append(frame_nb)
                last_frame_was_b_and_w=False
        if len(clips_end) != len(clips_start):
            clips_end.append(len(self.frames)-1)
            
        return clips_start, clips_end


    def remove_black_frame(self):
        middle_frame = self.frames[int(len(self.frames)/2)].pix_vals
        start = 0
        end=self.width-1
        black_pix = [0, 0, 0]
        middle_index=int(self.height/2)
        
        while start < self.width and middle_frame[middle_index][start].tolist() == black_pix:
            start+=1

        while end >= 0 and middle_frame[middle_index][end].tolist() == black_pix:
            end-=1
        if not (end < 0 or start >= self.width or end > self.width-10 or start < 10):
            for i in range(len(self.frames)):
                self.frames[i].crop(0, self.height-1, start, end, inplace=True)
            self.width = self.frames[0].pix_vals.shape[1]
            self.height = self.frames[0].pix_vals.shape[0]
            
            
        
    
    
def video_import_split_export(path, fps_in=15, fps_out=15):
    cap = cv2.VideoCapture(path)
    sec=0
    i=0
    frames=[]
    last_frame_was_b_and_w=True
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    clip_nb=1
    
    while(cap.isOpened()):
        sec = sec + 1/fps_in
        sec = round(sec, 2)
        cap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        ret, frame = cap.read()
        if ret==True:
            i += 1
            if i == 1:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
                print('export clip {:d}...' .format(clip_nb))
                file_path = os.path.join(Export.video_export_path, "clip_{:d}" .format(clip_nb))+'.mp4'
                out = cv2.VideoWriter(file_path, fourcc, fps_out, (width,height))
            frames.append(frame)
            h=frame.shape[0]
            w=frame.shape[1]
            img_norm = frame/255
            nb_pixels= h*w
            white_sum=(np.sum(img_norm[:int(h/4)])+np.sum(img_norm[int(3*h/4)+1:]))/3
            black_sum=np.sum(img_norm[int(h/4)+1:int(3*h/4)])/3
            if black_sum < nb_pixels*0.1/2 and white_sum > nb_pixels*0.9/2:
                if last_frame_was_b_and_w == False:
                    clip_nb += 1
                    print('export clip {:d}...' .format(clip_nb))
                    file_path = os.path.join(Export.video_export_path, "clip_{:d}" .format(clip_nb))+'.mp4'
                    out = cv2.VideoWriter(file_path, fourcc, fps_out, (width,height))
                    last_frame_was_b_and_w=True
            else:
                if last_frame_was_b_and_w == True:
                    last_frame_was_b_and_w=False
                out.write(frame)

        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

        
            


class Image(object):
    def __init__(self, img=None, path=None, name=None):
        if img is not None:
            self.pix_vals = img
        elif path is not None:
            self.pix_vals = cv2.imread(path)
        else:
            raise ValueError("Provide either an image array (img) or a file (path)")
        if name is not None:
            self.name = name
        else:
            self.name = "IMG"


    def face_detect(self, mode, model):
        img, self.face_detected = face_detection(self.pix_vals.copy(), mode, model)
        
        if self.face_detected == False and model=='both':
            img, self.face_detected = face_detection(self.pix_vals.copy(), mode, 'dlib')
            
        self.pix_vals = img
            
            
        if Export.no_face_detected_export == True and self.face_detected == False:
            self.export(Export.no_face_detected_path)
            
        
            
        return img.shape[1], img.shape[0]



    
    def convert_to_gray(self): 
        self.pix_vals = cv2.cvtColor(self.pix_vals, cv2.COLOR_BGR2GRAY)

    def export(self, path, exp_format="png"):
        export_path=os.path.join(path, self.name) + '.' + exp_format
        
        #if self.pix_vals.shape[0] < self.pix_vals.shape[1]:
         #   export_img = self.pix_vals
          #  cv2.imwrite(export_path, export_img)
        #else:
         #   print('coucou')
        cv2.imwrite(export_path, self.pix_vals)


   

    def resize(self, w, h):
        self.pix_vals = cv2.resize(self.pix_vals,(w,h))

    def rotate(self, angle, keep_same_dim=False):
        if keep_same_dim == False:
            k = angle/90
            rotated = np.rot90(self.pix_vals, k)
            self.pix_vals = rotated

        else:
            (h, w) = self.pix_vals.shape[:2]
            scale=h/w
            center = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(center, angle, scale)
            self.pix_vals = cv2.warpAffine(self.pix_vals, M, (w, h))
        
        
        
    def crop(self, yi, yf, xi, xf, inplace=True):
        img = self.pix_vals[yi:yf, xi:xf]
        
        if inplace == True:
            self.pix_vals = img
        else:
            return img
        return 1
    
    def emotions_prediction(self):
        
        self.emotions=EmoDetector()
        #self.resize(48, 48)
        self.resize(64, 64)
        #self.convert_to_gray()
        self.emotions.predict_emotion(self.pix_vals[np.newaxis, :, :, np.newaxis])
        
        
        
        
