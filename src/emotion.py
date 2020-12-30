#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 22:20:24 2019

@author: hugomeyer
"""


import pandas as pd
import numpy as np
from peak_processing import peak_detection, check_if_peak_in_peak
import matplotlib.pyplot as plt    
import os


class Emotion_cut(object):
    def __init__(self, index, time, score=-1):
        self.index=index
        self.time=time
        self.score=score
    
    def info(self):
        return {'index': self.index, 'score': self.score}


class Emotion(object):
    def __init__(self, preds, emo_labels, fps, interlude=False):
        self.labels_3 = ['p3_neg', 'p3_neu', 'p3_pos']
        self.labels_7 = ['p7_ang', 'p7_disg', 'p7_fear', 'p7_hap', 'p7_sad', 'p7_surp', 'p7_neu']
        self.tresholds = {'p7_ang': None, 
                          'p7_disg': None, 
                          'p7_fear': None, 
                          'p7_hap': 0.8,
                          'p7_sad': None, 
                          'p7_surp': 0.3, 
                          'p7_neu': None}
        self.best_cut = {'p7_ang': None, 
                          'p7_disg': None, 
                          'p7_fear': None, 
                          'p7_hap': None,
                          'p7_sad': None, 
                          'p7_surp': None, 
                          'p7_neu': None}
        self.emo_dict = {'p7_ang': 'Anger', 
                        'p7_disg': 'Disgust', 
                        'p7_fear': 'Fear', 
                        'p7_hap': 'Happiness', 
                        'p7_sad': 'Sad', 
                        'p7_surp': 'Surprise', 
                        'p7_neu': 'Neutral',
                        'p3_neg': 'Negative', 
                        'p3_neu': 'Neutral', 
                        'p3_pos': 'Positive'}
        self.max_peak_time={'p7_ang': None,   #In seconds
                          'p7_disg': None, 
                          'p7_fear': None, 
                          'p7_hap': 4,
                          'p7_sad': None, 
                          'p7_surp': 1, 
                          'p7_neu': None}
        self.preds = self.init_preds(preds)
        self.emo_labels = emo_labels
        self.best_emo_label = None
        self.no_face = self.no_face_interpolation(0.5)
        self.add_features()
        self.fps=fps
        self.stats = self.compute_stats(self.preds)
        self.ratios = self.compute_ratios()
        self.peaks = []
        self.preds, self.preds_after = self.cut_interlude_at_the_end(interlude)
        self.T = self.preds.shape[0]
        
        
        
    def init_preds(self, preds):
        labels=  self.labels_7 + ['7_best']
        preds.columns=labels
        for label in preds.columns:
            if label[0] == 'p':
                preds[label] = preds[label].astype(float)
              
        preds.index = range(1, preds.shape[0]+1)
        
        return preds
        
    
    
    
    def extract_features(self):

        if not self.no_face:
            self.signals = dict()
            for emo_label in self.emo_labels:
                max_peak_size=self.max_peak_time[emo_label]*self.fps
                peaks, best_peak = peak_detection(self.preds, self.T, emo_label, self.emo_dict, 
                                                  max_peak_size, self.tresholds[emo_label])
               
                self.peaks = self.peaks + peaks
        
      
    def cut_interlude_at_the_end(self, interlude):
        if interlude == True:
            df = self.preds.copy()
            end_clip_boundary=df.shape[0]-(4*self.fps-1)
            return df.loc[:end_clip_boundary], df.loc[end_clip_boundary:]
        else:
            return self.preds, None

        
    def compute_stats(self, data):
        stat_labels=['variance', 'mean', 'max', 'min', 'max_min_diff']
        stat_col = data.columns.values
        stat_col = [label for label in stat_col if label not in ['3_best', '7_best']]
        variances=[]
        means=[]
        maxis=[]
        mins=[]
        max_min_diff=[]
        list_stats=[]
        for col in stat_col:
            variances.append(data[col].var())
            means.append(data[col].mean())
            maxis.append(data[col].max())
            mins.append(data[col].min())
            max_min_diff.append(maxis[-1]-mins[-1])
            
        list_stats = [variances, means, maxis, mins, max_min_diff] 
        stats = pd.concat([pd.Series(x) for x in list_stats], axis=1).T
        stats.columns=stat_col
        stats.index=stat_labels
        return stats
    
    def compute_ratios(self):
        no_face_count = 0
        pos_count = 0
        hap_count = 0
        neg_count = 0
        neu_count = 0
        if 'No face' in self.preds['3_best'].values:
            no_face_count = self.preds['3_best'].value_counts()['No face']
        if 'Positive' in self.preds['3_best'].values:
            pos_count = self.preds['3_best'].value_counts()['Positive']
        if 'Happiness' in self.preds['7_best'].values:
            hap_count = self.preds['7_best'].value_counts()['Happiness']
        if 'Negative' in self.preds['3_best'].values:
            neg_count = self.preds['3_best'].value_counts()['Negative']
        if 'Neutral' in self.preds['3_best'].values:
            neu_count = self.preds['3_best'].value_counts()['Neutral']
        tot_count = self.preds['3_best'].count()
        neg_ratio = neg_count/(tot_count-no_face_count)
        neu_ratio = neu_count/(tot_count-no_face_count)
        pos_ratio = pos_count/(tot_count-no_face_count)
        hap_ratio = hap_count/(tot_count-no_face_count)
        return {'pos': pos_ratio, 'neu':neu_ratio , 'neg':neg_ratio , 'hap':hap_ratio}

    
    def add_features(self):
        emo3_list = ['Negative', 'Neutral', 'Positive']
        self.preds['p3_neg'] = self.preds[['p7_sad', 'p7_fear', 'p7_ang', 'p7_disg']].values.max(axis=1)
        self.preds['p3_neu'] = self.preds['p7_neu'].copy()
        self.preds['p3_pos'] = self.preds['p7_hap']+self.preds['p7_surp']
        p_sum = self.preds['p3_pos'] + self.preds['p3_neg'] + self.preds['p3_neu']
        for label in ['p3_pos', 'p3_neg', 'p3_neu']:
            self.preds[label] /= p_sum
        preds3 = self.preds[['p3_neg', 'p3_neu', 'p3_pos']]
        self.preds['3_best'] = [emo3_list[np.argmax(preds3.iloc[i].values)] for i in range(preds3.shape[0])]
        #reg_model = train_model(path)
        #self.preds['reg_score'] = reg_model.predict(self.preds[['p3_neg', 'p3_neu', 'p3_pos']])
                
                
    
    def no_face_interpolation(self, discard_clip_treshold):
        no_face_ratio = self.preds['7_best'][self.preds['7_best']=='No face'].count()/self.preds.shape[0]
        df = self.preds.copy()

        if no_face_ratio < discard_clip_treshold:
            if no_face_ratio == 0:
                return False
            
            indices_not_missing=df.index[df['7_best']!='No face'].copy().values
            indices_to_interp=df.index[df['7_best']=='No face'].copy().values
            values_not_missing=df[df['7_best']!='No face'].copy()
            for col in df.columns:
                if col != '7_best':
                    self.preds.loc[indices_to_interp, col]=np.interp(indices_to_interp, indices_not_missing, values_not_missing[col].values)
            preds = self.preds.iloc[:, :-1].copy()
            max_preds = preds.loc[indices_to_interp].idxmax(axis=1)
            self.preds.loc[indices_to_interp, '7_best'] = [self.emo_dict[max_pred]  for max_pred in max_preds]
            ratio = indices_to_interp.shape[0]/self.preds.shape[0]
            print("{:.3f}% of emotion predictions were interpolated" .format(ratio))
            
            return False
        else:
            return True
        
        
    


       
    
    def determine_best_emo_label(self):
        best_cuts = [(label, cut.score, cut.index) for label, cut in self.best_cut.items() if cut is not None]
        if best_cuts:
            indices = np.asarray(best_cuts)[:, 2].astype(int)
            sorted_ind = np.argsort(indices)
            indices = indices[sorted_ind]
            cuts_scores = np.asarray(best_cuts)[:, 1].astype(float)
            cuts_scores = cuts_scores[sorted_ind]
            labels = np.asarray(best_cuts)[:, 0]
            labels = labels[sorted_ind]
            
            best_peak_index = np.argmax(cuts_scores)
            
            # Check if 2 peaks with different emotions are close in time and score
            # If yes, select always the first of the two peaks for the cut
            if len(best_cuts)>1:
                score_diff = (max(cuts_scores) - cuts_scores[0]) 
                time_diff = (indices[best_peak_index]-indices[0])/self.fps
                if score_diff < 0.5 and time_diff < 5:
                    self.best_emo_label = labels[0]
            else:
                self.best_emo_label = labels[best_peak_index]
        
                



    def lineplot_analog(self, labels, clip_name, duration, title='', markers='time', peaks=False, target=False, output=False, 
                        save=False, path='../'):
        #plt.figure(num=None, , dpi=80, facecolor='w', edgecolor='k')
        if self.preds_after is not None:
            df = pd.concat([self.preds, self.preds_after])
        else:
            df = self.preds.copy()
        data = df[labels].copy()

        if save:
            dpi = 80
        else:
            dpi = 80
        
        fig, ax = plt.subplots(figsize=(15, 4), dpi=dpi)
        
        if markers=='time':
            data.index = np.linspace(0, duration, data.shape[0])
            X_ticks = np.arange(0, duration, 1)
        else:
            data.index = range(1, data.shape[0]+1)
            X_ticks = np.arange(0, data.shape[0]+1, 5)
            X_ticks[0]=1
        
        y_ticks = np.arange(0, 1.1, 0.1)
        nb_break_frames = 16
        
        
        ax.spines['top'].set_color('white')
        ax.spines['right'].set_color('white')
        
        maxi = data.values.max()
        mini = data.values.min()
        
        ax.set_xticks(X_ticks)
        ax.set_yticks(y_ticks)
        if markers == 'time':
            xlabel = 'Time (s)'
        else:
            xlabel = 'Frames'
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Probabilities')
        ax.yaxis.grid(False)
        if data.shape[0] > 100:
            marker=''
        else:
            marker='o'
            
        if peaks == True:
            for peak in self.peaks:
                emo_data=data[peak.emotion]
                if peak.rise_start != None:
                    plt.scatter(emo_data.index.values[peak.rise_start-1:peak.start], emo_data.values[peak.rise_start-1:peak.start], color='b', s=130)
                if peak.fall_end != None:
                    plt.scatter(emo_data.index.values[peak.end-1:peak.fall_end], emo_data.values[peak.end-1:peak.fall_end], color='b', s=130)
                plt.scatter(emo_data.index.values[peak.start-1:peak.end], emo_data.values[peak.start-1:peak.end], color='r', s=100)
            
            
            
        if target == True:
            for label in labels:
                cut = self.best_cut[label]
                if cut is not None:
                    plt.axvline(x=cut.index, linewidth=2, linestyle='--',  color='g')
    
        if self.preds_after is not None:         
            ax.fill_between([data.shape[0], data.shape[0]-nb_break_frames+1], [mini, mini], [maxi, maxi], facecolor='black', alpha=0.1)  
            plt.axvline(x=data.shape[0]-nb_break_frames+1, linewidth=1, color='k')
            #plt.axvline(x=data.shape[0], linewidth=1, color='k')
            
            
        data.rename(columns = {'p7_hap':'Happiness', 'p7_surp':'Surprise'}, inplace = True)
        
        
        if output:
            print(self.best_cut[self.best_emo_label])
            plt.axvline(x=self.best_cut[self.best_emo_label].time, linewidth=2, linestyle='--',  color='g')
            pass
            
        #ax.fill_between([data.shape[0], data.shape[0]-nb_break_frames+1], [mini, mini], [maxi, maxi], facecolor='black', alpha=0.1)  
        #plt.axvline(x=data.shape[0]-nb_break_frames+1, linewidth=1, color='r')
        #plt.axvline(x=data.shape[0]-nb_break_frames+1, linewidth=1, color='r')

        data.plot(kind='line',ax=ax, grid=True, marker=marker, title=title, legend=True)
        
        
        if target==True:
            labels_file = '../Data/Experiment3/cuts_fps10.xlsx'
            labels = pd.read_excel(labels_file)
            clip_labels = labels[labels.clip_nb == 18].copy()
            cut_starts = str(clip_labels.cut_start.values[0]).split(',')
            cut_ends = str(clip_labels.cut_end.values[0]).split(',')
            clip_labels = np.asarray([[start.strip(), end.strip()] for start, end in zip(cut_starts, cut_ends)])
            clip_labels[clip_labels=='end'] = -1
            clip_labels = clip_labels.astype('float')
            clip_labels[clip_labels==-1] = (len(self.preds)-1)/self.fps
            starts, ends = np.round(clip_labels[:, 0]*self.fps).astype(int), np.round(clip_labels[:, 1]*self.fps).astype(int)
            for start, end in zip(starts, ends):
                plt.axvline(x=start, linewidth=1, color='r')
                plt.axvline(x=end, linewidth=1, color='r')
                ax.fill_between([start, end], [mini, mini], [maxi, maxi], facecolor='r', alpha=0.1)  
                
    
        if save == True:
            plt.savefig(os.path.join(path, 'Clip_{}_emotion' .format(clip_name)+'.png'))
        else:
            plt.show()



def determine_emotion_fps2(filename, max_HL_time):
    time = get_video_duration(filename)
    if time > 60:
        emo_fps = 2
    elif time > 30:
        emo_fps = 3
    else:
        if max_HL_time > 4:
            emo_fps = 4
        else:
            emo_fps = 9-max_HL_time

    return emo_fps

def determine_emotion_fps(filename, max_HL_time):
    return 4
