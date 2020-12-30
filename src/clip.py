#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:44:10 2019

@author: hugomeyer
"""


from emotion import Emotion
from similarity import Similarity
from img_processing import from_img_to_bins, from_binned_img_to_tridiag, min_max_tridiags_pair
from media import Video
from sound import Sound
from events_timeline import Events_timeline

import pandas as pd
import os
import math
import numpy as np
from scipy.io.wavfile import read
import subprocess


class Trim(object):
    def __init__(self, start, end, fps):
        self.start=start
        self.end=end
        self.fps = fps
        self.time=(self.end-self.start)/fps
        
    def info(self):
        return {
            'start':self.start,
            'end': self.end,
            'time': self.time,
            'fps': self.fps
        }
    



class Clip(object):
    def __init__(self, ID=None, back_video_path='', front_video_path='', 
                 trimmed_video_export='', target_cut=False):
        #if target_cut==True:
         #   self.targ_cuts=self.load_target_cuts(ID, data_path, 'cuts_fps4.csv')
        #else:
            #self.targ_cuts=None
            
        #if ID is not None and name is None:
         #   name = self.load_name(ID, data_path)
            
        self.ID = ID
        self.duration = None
        #self.name = name
         
        self.emotion=None
        self.similarity=None
        self.sound=None
        
        self.events_timeline = None
        
        self.emotion_trigger=False
        
        self.front_video=front_video_path
        self.back_video=back_video_path
        self.trim_export=trimmed_video_export
        self.trim = None
        
        
    def compute_events_timeline(self, fps):
        signals = dict()
        #signals['happiness'] = self.emotion.preds['p7_hap'].values
        #signals['surprise'] = self.emotion.preds['p7_surp'].values
        print("Sound!")
        print(self.sound.y)
        signals['sound'] = self.sound.y
        print("Similarity!")
        print(self.similarity.values)
        signals['similarity'] = self.similarity.values
        features = dict()
        print("Happy!")
        print(self.emotion.preds['p7_hap'].values)
        print(self.emotion.preds['p7_surp'].values)
        if self.emotion is not None:
            adjusted_fps_ratio = self.duration/self.emotion.preds.shape[0]
        
            features['happiness'] = [
                                        {
                                                'start': peak.start*adjusted_fps_ratio,
                                                'end': peak.end*adjusted_fps_ratio,
                                                'rise_start': peak.rise_start,
                                                'fall_end': peak.fall_end,
                                                'score': peak.avg
                                        } 
                                        for peak in self.emotion.peaks if peak.emotion == 'p7_hap'
                                    ]
            
            features['surprise'] = [
                                        {
                                                'start': peak.start*adjusted_fps_ratio,
                                                'end': peak.end*adjusted_fps_ratio,
                                                'rise_start': peak.rise_start,
                                                'fall_end': peak.fall_end,
                                                'score': peak.avg
                                        } 
                                        for peak in self.emotion.peaks if peak.emotion == 'p7_surp'
                                    ]
        else:
            features['happiness'] = []
            features['surprise'] = []
        
        features['sound'] = [
                                    {
                                            'start': event.start_t,
                                            'end': event.end_t,
                                            'score': event.score,
                                            'label': event.label
                                    } 
                                    for event in self.sound.events
                            ]
        
        features['similarity'] = [
                                    {
                                            'time': feat.time,
                                            'score': feat.score,
                                            'label': feat.label,
                                    } 
                                    for feat in self.similarity.features
                            ]
        for emo_label in ['happiness', 'surprise']:
            for i in range(len(features[emo_label])):
                if features[emo_label][i]['rise_start'] is not None:
                    features[emo_label][i]['rise_start'] *= adjusted_fps_ratio
                if features[emo_label][i]['fall_end'] is not None:
                    features[emo_label][i]['fall_end'] *= adjusted_fps_ratio
        print("Emotions!")
        print(features) 
    
        
        self.events_timeline = Events_timeline(signals, features, self.duration, fps)
        
        
        
        
    def compute_audio_signal(self, input_path):
        
        try: 
            fs, track = read(input_path)
        except:      
            command = "ffmpeg -i "+self.back_video+" -ab 160k -ac 2 -ar 44100 -vn "+input_path
            subprocess.call(command, shell=True)
            try:
                fs, track = read(input_path)
            except:
                return 0
            
        
        self.sound = Sound(track[:, 0], fs, self.ID, input_path)
        self.duration = self.sound.L
        return 1
        

    def compute_emotions(self, emo_labels, fps=4):

        #determine_emotion_fps(os.path.join(self.front_video), self.max_HL_time)
     
        video = Video([], self.duration, path=self.front_video, name=self.ID, fps=fps)
        
        if video.successful_loading:
        #video.rotate(180)
            video.face_detect(mode='crop', model='both')
            video.emotions_prediction()
            
            preds7 = np.asarray(video.emotions.preds7)
            best_guess7 = np.asarray(np.expand_dims(video.emotions.best_guess7, axis=1))
            
            concat = np.concatenate((preds7, best_guess7), axis=1)
            df = pd.concat([pd.Series(x) for x in concat], axis=1).T
            
                    
            self.emotion = Emotion(df, emo_labels, fps=fps)
            return 1
        else:
            print('Front video of clip {} not found.' .format(self.ID))
            return 0
        
        
        
        
    def compute_img_similarity(self, simil_fps, importing_time_limit=30, ratio=0.1, bin_size=2):  
        
        tot_simil_vect=[]
        for start in range(0, int(self.duration), importing_time_limit):
            video=Video(frames=[], path=None, name='clip_{}'.format(self.ID), fps=simil_fps, duration=self.duration) 
            if start:
                start -=1
                importing_time_limit += 1
            end = min(start+importing_time_limit, self.duration)
                        
            successful_loading = video.load_frames(path=self.back_video, fps=simil_fps, record_start_time=start, time_limit=end)
            if not successful_loading:
                raise ValueError('The video file was not found. Check the path again')

                    
            video.remove_black_frame()
            imgs = [frame.pix_vals for frame in video.frames]             
            simil_vect=[]
            
            for i in range(1, len(imgs)-1):
                triplet_imgs = [imgs[j] for j in [i-1, i, i+1]]
                binned_imgs = [from_img_to_bins(img, ratio, bin_size) for img in triplet_imgs]
                diags = [from_binned_img_to_tridiag(binned_imgs[i], binned_imgs[i+1], bin_size) for i in range(2)]
                min_, max_ = min_max_tridiags_pair(diags[0], diags[1])
                simil_vect.append(min_/max_)
                
            simil_vect = [simil_vect[0]] + simil_vect

            
            if start == 0:
                tot_simil_vect = simil_vect
            else:
                tot_simil_vect += simil_vect[9:]

        self.similarity = Similarity(1, len(tot_simil_vect), tot_simil_vect, simil_fps)
            
            
        

    
        
        
    
        
        
        
            
    def load_name(self, ID, path):
        names = pd.read_csv(os.path.join(path, 'clip_names.csv'))
        #if isinstance(ID, int):
        #name = names.Name[ID-1]
        #else:
         #   name = ID
          #  if ID in names['Name'].values:
           #     ID = names['Name'].index[names['Name'].values == ID][0]+1
            #else:
             #   ID = None
        #return ID, name 
        return names.Name[ID-1]
    
            
            
    def load_target_cuts(self, ID, path, file):
        fps=4
        df = pd.read_csv(os.path.join(path, file))
        starts, ends = df.iloc[ID-1, 0], df.iloc[ID-1, 1]
        try:
            starts, ends = [int(starts)], [int(ends)]
        except:
            if not isinstance(starts, str):
                if math.isnan(starts):
                    starts = [-1]
                    ends = [-1]
        if isinstance(starts, str):
            starts = [round(int(el)*fps/4) for el in starts[1:-1].split(',')]
            ends = [round(int(el)*fps/4) for el in ends[1:-1].split(',')]        
        
        return [Trim(start, end, fps) for start, end in zip(starts, ends)]
        
