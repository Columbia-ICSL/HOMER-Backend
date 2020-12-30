#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 15:24:55 2019

@author: hugomeyer
"""


import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.scimath import logn
from math import e

from cut import Cut





class End_cut(Cut):
    def __init__(self, features, time, fps, sound_dict, hl_max_size=-1, hl_min_size=-1):
        Cut.__init__(self, features, time, fps, sound_dict, hl_max_size, hl_min_size)
        
        self.best_intervals = []
        self.simil_labels = ['plateau_end', 'valley_end', 'hill_top', 'plateau_start']
        self.simil_ranges = []
        self.intervals_on_similarity_zone = []
    
    
    def find_n_best_end_cut_intervals(self):
        self.emotions_scoring()
        self.sound_scoring()
        self.similarity_scoring()
        #self.plot_score_fct()
        self.find_best_intervals()
        
        
    
    
    
    def find_best_intervals(self):
        function = self.score_fct.copy()
        print(self.score_fct)
        print(self.time_score_fct)
        print(self.sound_score_fct)
        print(self.emo_score_fct)
        print(self.simil_score_fct)
        #self.plot_score_fct()
        interval_middle = np.argmax(function)
        median = np.median(function)
        
        function[:self.hl_min_size] = median-1
        left, right = self.find_plateau(function)
        interval_middle = int((left + right)/2)
        
        while function[int(interval_middle)] >= median:
            print("Got here")
            #Prevent to get interval with score == 0
            interval_on_similarity_zone = False

            for (start, end) in self.simil_ranges:
                if interval_middle >= start and interval_middle <= end:
                    interval_on_similarity_zone = True

            self.intervals_on_similarity_zone.append(interval_on_similarity_zone)

            interval = [{'time': self.time[el], 'index': el, 'score': function[el]} for el in range(left, right+1)]
            self.best_intervals.append(interval)

            function[left:right+1] = median-1
            
            
            left, right = self.find_plateau(function)
            interval_middle = int((left + right)/2)
     
            #self.plot_score_fct()
            
            
    
    def similarity_scoring(self):
        factor = self.params['similarity']['downscale_factor_end']

        relevant_simil = [feat for feat in self.features['similarity'] 
                          if feat['label'] in self.simil_labels and feat['time'] > self.hl_min_size]
        
        relevant_simil = self.merge_double_points(relevant_simil)
        
        for feat in relevant_simil:
            #feat['score'] = self.reajust_scores_given_labels(feat['label'], feat['score'])
            #gaussian_vals = norm.pdf(np.arange(self.L), loc=feat['time'], scale=feat['score']*3)
            score = feat['score'] if feat['time'] > 1*self.fps else feat['score']/2
            ratio1 = self.params['similarity']['upscale_valley']
            ratio2 = self.params['similarity']['upscale_plateau']

            if not self.features['sound'] and len([simil['label'] for simil in relevant_simil if simil['label']=='hill_top'])==len(relevant_simil):
                ratio3 = self.params['similarity']['upscale_hill']
            else:
                ratio3 = self.params['similarity']['downscale_hill']
            valley_end_or_plateau_start = feat['label'] == 'valley_end' or feat['label'] == 'plateau_start'
            score = score**(1/ratio1) if valley_end_or_plateau_start else score**(1/ratio2) if feat['label'] == 'plateau_end' else score*ratio3
            half_t_range = max(int(round(self.params['similarity']['max_time_range']*self.fps*feat['score']/2)), 1)
            start = feat['time']-half_t_range
            end = feat['time']+half_t_range+1
            self.simil_score_fct[start:end] += score*factor#gaussian_vals*feat['score']*0.5/gaussian_vals.max()
            self.simil_ranges.append([start, end])
        self.score_fct += self.simil_score_fct
    
    
    def reajust_scores_given_labels(self, label, score):
        ratio = self.params['similarity']['labels_rescale']
        if label == 'valley_end' or label == 'hill_end':
            return (score+ratio)/(1+ratio)
        elif label == 'plateau_end' or label == 'plateau_start':
            return (score+ratio/2)*(2-ratio)/(2+ratio)
        else:
            return score*(1-ratio)
    
    
    
    def merge_double_points(self, simil_pts):
        to_rm = list()
        for i in range(len(simil_pts)-1):
            if simil_pts[i]['time'] == simil_pts[i+1]['time']:
                indices = [i, i+1]
                best_index = np.argmax([simil_pts[i]['score'], simil_pts[i+1]['score']])
                simil_pts[indices[best_index]]['score'] = max(simil_pts[i]['score'], simil_pts[i+1]['score'])
                to_rm.append(indices[1-best_index])
                
        return np.delete(simil_pts, to_rm)
    
        
    def sound_scoring(self):
        events_labels = [self.sound_dict[feat['label']] for feat in self.features['sound']]
        rescale_laugh_and_speech = 'laugh' in events_labels and 'speech' in events_labels
        #downscale_misc = True #'laugh' in events_labels or 'speech' in events_labels
        upscale_similarity = not 'laugh' in events_labels and not 'speech' in events_labels
        
        if upscale_similarity:
            self.params['similarity']['downscale_factor_end'] = 1
            
        
        for feat in self.features['sound']:
            sound_label = self.sound_dict[feat['label']]
            if feat['end'] > self.hl_min_size:
                #sound_scale_up = (sound_label == 'laugh')^(sound_label == 'speech')
                if sound_label == 'laugh':
                    score = feat['score']#*(feat['end']-feat['start'])/(3*self.fps)
                    #Upscale laugh score as long as speech is present in the clip
                    score = score**(1/2) if rescale_laugh_and_speech else score
                    score *= 1.5
                    event_vals = np.linspace(-score, score, feat['end']-feat['start'])
                    self.sound_score_fct[feat['start']:feat['end']] += event_vals

                    pattern_end = self.score_pattern('constant', 'laugh', feat['end'], score)
                    self.score_pattern('linear', 'laugh', pattern_end, score)
                    
                elif sound_label == 'speech':
                    score = feat['score']#*(feat['end']-feat['start'])/(3*self.fps)
                    #Downscale speech score as long as laugh is present in the clip
                    score = logn(e, 1+score) if rescale_laugh_and_speech else score
                    score *= 1.5
                    #Penalize more for cutting during speech and scale up the low values with square function
                    neg_score = score*self.params['sound']['during_speech_penalty']

                    self.sound_score_fct[feat['start']:feat['end']] -= neg_score
                    self.test[feat['start']:feat['end']] -= neg_score
                    pattern_end = self.score_pattern('constant', self.sound_dict[feat['label']], feat['end'], score)
                    self.score_pattern('linear', self.sound_dict[feat['label']], pattern_end, score)
                    
                else:
                    score = feat['score']*(feat['end']-feat['start'])/(3*self.fps)
                    score = min(score, self.params['sound']['max_misc_score_end_cut'])
                    #score = score*self.params['sound']['misc_scale_down'] if downscale_misc else score
                    self.sound_score_fct[feat['start']:feat['end']] -= score/2
                    self.test[feat['start']:feat['end']] -= score
                    duration = (feat['end']-feat['start'])/self.fps
                    pattern_end = self.score_pattern('constant', self.sound_dict[feat['label']], feat['end'], score, duration)
                    self.score_pattern('linear', self.sound_dict[feat['label']], pattern_end, score, duration)
          
                
          
                
        self.score_fct += self.sound_score_fct
        
        
        
        
    def score_pattern(self, shape, event, last_pattern_end, score, duration=None):
        if event == 'misc':
            ratio = self.params['sound']['misc_cue_ratio'] 
            param = duration*ratio if shape == 'constant' else duration*(1-ratio)
            #print(shape, param, duration)
        else:
            param = self.params['sound']['shape'][shape]['speech']
            
        width = int(round(min(param*self.fps, self.L - last_pattern_end)))
        if shape == 'constant':
            vals = score
        else:
            end_val = score*(1-width/(param*self.fps))
            vals = np.linspace(score, end_val, width)
        end = last_pattern_end+width 
        self.sound_score_fct[last_pattern_end:end] += vals
        if event == 'speech':
            self.test[last_pattern_end:end] += vals
        
        return end
        
    
    
    
        
   
    def emotions_scoring(self):
        
        
        #Score the emotion peaks with their respective score as squares 
        #with a cue at the end proportional to the peak width        
        for emotion in ['happiness', 'surprise']:
            last_emo_peak_start = self.L
            for feat in self.features[emotion][::-1]:
                if feat['fall_end'] > self.hl_min_size:
                    score = feat['score']*self.params['emotion']['upscale_factor']
                    rise_vals = np.linspace(0, score, feat['start']-feat['rise_start']+1)[:-1]
                    ratio_cue = max(1-(feat['end'] - feat['start'])/(self.params['emotion']['max_peak_duration']*self.fps), 0)
                    #ratio_const = (feat['end'] - feat['start'])/(feat['fall_end']-feat['start'])
                    const_width = int(round(ratio_cue*self.params['emotion']['max_cue_width']*self.fps))
                    #const_end = int(round(ratio_const*cue_width + feat['fall_end']))
                    #const_end = min(int(round(ratio_const*(cue_width+feat['fall_end']-feat['start']) + feat['start'])), self.L)
                    const_end = min(const_width+feat['fall_end'], last_emo_peak_start)
                    print(feat['rise_start'], feat['start'])
                    #fall_width = int(round(ratio_cue*self.params['emotion']['max_cue_width']*self.fps))
                    #fall_width_adjusted = min(fall_width, self.L-const_end)
                    #y = score*(fall_width-fall_width_adjusted)/max(fall_width, 1)
                    #fall_vals = np.linspace(score, y, fall_width_adjusted) 
                    #fall_end = const_end + fall_width_adjusted
                    self.emo_score_fct[feat['rise_start']:feat['start']] = rise_vals
                    self.emo_score_fct[feat['start']:const_end] += score
                    last_emo_peak_start = feat['rise_start']
                    #self.emo_score_fct[const_end:fall_end] += fall_vals
        self.score_fct += self.emo_score_fct
    
    
    
    
    
    def plot_score_fct(self, cut=None, save=False):
        title = 'Score function for the end cut of the highlight'
        x_label = 'Time'
        y_label='Score'
            
            
        x = self.time#+self.offset
        y = np.asarray(self.score_fct)
    
        fig, ax = plt.subplots(figsize=(10, 4), dpi=80)
        
        factor = 1/self.fps
        
        if x.shape[0] > 400:
            step = round(50*factor)
        elif x.shape[0] > 200:
            step = round(20*factor)
        elif x.shape[0] > 100:
            step = round(10*factor)
        elif x.shape[0] > 30:
            step = max(round(5*factor, 1), 0.1)
        else:
            step = max(round(1*factor, 1), 0.1)
            
        glob_min = min(self.simil_score_fct.min(), self.emo_score_fct.min(), 
                       self.sound_score_fct.min(), y.min())
        
        glob_max = max(self.simil_score_fct.max(), self.emo_score_fct.max(), 
                       self.sound_score_fct.max(), y.max())
        
        ystep = round((glob_max-glob_min)/10, 1)
        
        X_ticks_maj = np.arange(max(int(x[0])-factor, 0), int(x[-1])+2, step)

        y_ticks = np.arange(round(y.min(), 1)-0.1, round(y.max(), 1)+0.1, max(ystep, 0.1))
        X_ticks_min = np.arange(x[0]-factor, x[-1]+1, step*0.2)
        X_ticks_min[0] = int(x[0])

        
        ax.spines['top'].set_color('white')
        ax.spines['right'].set_color('white')
        
        ax.set_xticks(X_ticks_maj)
        ax.set_xticks(X_ticks_min, minor=True)
        ax.set_yticks(y_ticks)
        ax.grid(which='both')
        ax.grid(which='minor', alpha=0.4)
        ax.grid(which='major', alpha=1)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        plt.title(title)
            
        
        
        if cut is None:
            ax.axvline(x=self.time[np.argmax(y)], linewidth=1, color='g')
        else:
            ax.axvline(x=cut, linewidth=1, color='g')
        plt.plot(x, self.emo_score_fct, label='Emotion')
        plt.plot(x, self.sound_score_fct, label='Sound')
        plt.plot(x, self.simil_score_fct, label='Similarity')
        
        plt.plot(x, y, label='Total')
        
        plt.legend()
            
        if save == True:
            plt.savefig('Clip_{}_end_score_fct.png')
        else:
            plt.show()
    
        
