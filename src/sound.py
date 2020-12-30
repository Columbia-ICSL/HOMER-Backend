#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 12:05:53 2019

@author: hugomeyer
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import pywt
from sklearn.cluster import KMeans 
from scipy.integrate import simps

from sound_recognition import predict_wrapper


        
class Sound_event(object):
    def __init__(self, start_i, end_i, start_t, end_t, score):
        self.start = start_i
        self.end = end_i
        self.start_t = start_t
        self.end_t = end_t
        self.score = score
        self.label = None
        
    def info(self):
        print({'start': self.start, 'end': self.end, 
               'start_t': round(self.start_t,2), 'end_t': round(self.end_t, 2), 
               'score': round(self.score,2),
               'label': self.label})



class Sound(object):
    def __init__(self, signal, fs, index, path):
        self.index = index
        self.y = np.asarray(signal)
        self.fs = fs
        self.L = signal.shape[0]/fs
        self.t = np.linspace(0, self.L, len(signal))
        self.T = 1/fs
        self.N = self.y.shape[-1]
        self.only_noise = False
        self.treshold = None
        self.events = []
        self.origY = self.y
        self.origT = self.t
        self.path = path
        self.event_labels = None
        
    def subsample(self, f):
        self.y = self.y[::f]
        self.t = self.t[::f]
        self.fs /= f
        self.T = 1/self.fs
        self.N = self.y.shape[-1]
        
        
    def fft(self, db=False):
        signal = self.y.copy()
        yf = fft(signal)
        self.xf = np.linspace(0.0, 1.0/(2.0*self.T), self.t.shape[0]//2)
        self.yf = 2.0/self.N * np.abs(yf[0:self.N//2])
        if db == True:
            yf = 20 * np.log10(yf / np.max(yf))
            
            
            
    def events_recognition(self):
        self.event_labels = predict_wrapper(self.path, '../Models/Sound/laughter', '../Models/Sound/speech')
        #print(self.event_labels)
            
            
    def segmentation_preprocess(self, subsamp_freq=1000, nb_clusters=3, wavelet_lvl=7):        
        self.subsample(int(self.fs/subsamp_freq))
        self.origY = self.y
        self.origT = self.t
        self.y = (np.abs(self.y)/np.max(np.abs(self.y)))**2
        self.y = self.wavelet_filtering(self.y, nb_coeffs=1, lvl=wavelet_lvl)
        

        self.N = self.y.shape[-1]
        self.t = np.linspace(0, self.L, self.N)
        self.fs = self.N/self.L
        self.T = 1/self.fs
        
        integral = simps((self.y/self.y.max())**2, self.t)
        
        #Remove audio tracks that are too noisy -> big integral
        
        
        if integral/self.L >= 0.3:
            self.only_noise = True
            
        
        self.y = (self.y/self.y.max())-np.percentile(self.y, 50)
        self.y = np.where(self.y<0, 0, self.y)
        
        #self.tresholds.append(self.y.mean())
        
        
        
    

        
    def find_treshold(self, signal, nb_cluster=3, class_min=2):
        kmeans = KMeans(n_clusters=nb_cluster, n_init=10, tol=1e-4).fit(signal[:, np.newaxis])
        centroids = kmeans.cluster_centers_.reshape(-1)
        labels=kmeans.labels_
        class_label = np.argsort(centroids)[class_min]
        threshold = min(signal[labels==class_label])
        return threshold

    def wavelet_filtering(self, signal, nb_coeffs, lvl):
        coeffs = pywt.wavedec(signal, 'db1', level=lvl)
        return pywt.waverec(coeffs[:nb_coeffs], 'db1')
    
    def events_segmentation(self):
        self.treshold = self.adaptive_treshold(2)#min(self.tresholds)
        
        
        
        if self.treshold < 0.01:
            self.only_noise = True
       # print("treshold: ", self.treshold)
        if self.only_noise:
            #self.events = None
            return
        events_boundaries = self.find_events_boundaries()
        #print("event boundaries: ", events_boundaries)
        coverage = sum([bound[1]-bound[0] for bound in events_boundaries])/self.N
        if coverage > 0.7:
         #   print('Max coverage reached: > 70%')
            self.events = []
            self.treshold = self.adaptive_treshold(2, nb_pts=4, elbow_constraint_free=True)
         #   print("treshold: ", self.treshold)
            events_boundaries = self.find_events_boundaries()
            
 
   

        events_score = self.compute_event_score(events_boundaries)
        self.events = [Sound_event(start, end, (start+1)*self.T, (end+1)*self.T, score) 
                       for score, (start, end) in zip(events_score, events_boundaries)]
        
        if self.event_labels is not None:
            self.merge_segmentation_and_recognition()
            
        self.signal_simplification(events_score, events_boundaries)
        
        #self.info()
        #print([event.info() for event in self.events])
        
        
        
        
    def signal_simplification(self, events_score, events_boundaries):
        feat_signal = np.zeros(len(self.y))
        for score, (start, end) in zip(events_score, events_boundaries):
            feat_signal[start:end+1] = score
        self.y = feat_signal
        
        
        
        
    def merge_segmentation_and_recognition(self):
        for i in range(len(self.events)):
            start = self.events[i].start_t
            end = self.events[i].end_t
            
            durations = [0, 0, 0]
            event_indices = np.arange(int(start), min(int(end)+1, len(self.event_labels)), 1) 
            event_indices
            for j in event_indices:
                if start > j:
                    durations[self.event_labels[j]] += (1-start+j)
                elif abs(j-end) < 1:
                    durations[self.event_labels[j]] += end-j
                else:
                    durations[self.event_labels[j]] += 1
    
            if durations[0] + durations[1] > 0.25*(end-start):
                self.events[i].label = np.argmax(durations[:2])
            else:
                self.events[i].label = len(durations)-1
            
        
        
        
    def find_events_boundaries(self):
        peaks_pts_ind = np.squeeze(np.argwhere(self.y > self.treshold))
        if peaks_pts_ind.shape:
            indices_shifted = np.append(peaks_pts_ind[1:], peaks_pts_ind[-1]+1)
            bounds = [0] + list(np.nonzero(indices_shifted-peaks_pts_ind-1)[0]+1) + [len(peaks_pts_ind)]
            peaks_intervals = [(peaks_pts_ind[bounds[i]], peaks_pts_ind[bounds[i+1]-1]) for i in range(len(bounds)-1)]
        else:
            peaks_intervals = [(peaks_pts_ind, peaks_pts_ind)]
        raw_boundaries = self.find_subevents_boundaries(peaks_intervals, momentum=0.5)
        #print("raw bounds: ", raw_boundaries)
        return self.merge_subevents_into_event(raw_boundaries)

        
    def compute_event_score(self, boundaries):
        scores=[]
        for start, end in boundaries:                
            event_integral = simps(self.y[start:end+1], self.t[start:end+1])
            event_duration = self.t[end]-self.t[start]
            score = 2*event_integral/event_duration
            scores.append(score)
            
        return scores
        
    
            
        
    def find_subevents_boundaries(self, peaks_intervals, momentum=0.5, eps=0.02): #momentum in seconds
        boundaries = []
        for interval in peaks_intervals:
            (start, end) = interval
            momentum_ind = int(momentum/self.T)+1
            stop_condition = False
            i = start
            while not stop_condition and i >= 0:
                curr_val = self.y[i]
                proj_val = self.y[max(i-momentum_ind, 0)]
                if curr_val - proj_val < eps:#and prec_val-curr_val < eps:
                    while i > 0 and self.y[i]-self.y[i-1] > 0.01:
                        i -= 1
                    stop_condition = True
                i -= 1
            i += 1
            j = end
            stop_condition = False
            while not stop_condition and j<len(self.y):
                curr_val = self.y[j]
                proj_val = self.y[min(j+momentum_ind, len(self.y)-1)]
                if curr_val - proj_val < eps:# and prec_val-curr_val < eps:
                    while j < len(self.y)-1 and self.y[j]-self.y[j+1] > 0.01:
                        j += 1
                    stop_condition = True
                j += 1
            j -= 1
            if i!=j:
                boundaries.append([i, j])
            else:
                boundaries.append([i-1, j+1])
        return boundaries
    
    
    def adaptive_treshold(self, nb_crosses_limit=2, nb_pts=2, elbow_constraint_free=False):
        nb_crosses=[]
        tresh_vals = np.arange(0, 0.5, 0.01)
        for i in tresh_vals:
            nb_crosses.append(self.nb_of_treshold_crosses(self.y, i)/self.L)
        '''
        fig, ax = plt.subplots(figsize=(15, 4), dpi=80)
        ax.spines['top'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.set_xticks(np.arange(0, 0.5, 0.01), minor=True)
        ax.grid(which='both')
        ax.set_xlabel('Treshold value')
        ax.set_ylabel('Nb of threshold crosses over video duration')
        
        plt.scatter(np.arange(0, 0.5, 0.01), nb_crosses)
        plt.plot(np.arange(0, 0.5, 0.01), nb_crosses)
        plt.plot(np.arange(0, 0.5, 0.01), np.ones(len(nb_crosses))*2, color='r', linestyle='--')
        path='/Users/hugomeyer/Desktop/PDM/Smartphone-app-multimedia-smart-selection/Paper/'
        plt.savefig(path + 'threshold.png')
        '''
        if nb_crosses:
            index_max = np.argmax(nb_crosses)
            
            if index_max != 0:
                elbow=self.check_if_elbow(nb_crosses, index_max, 0.2, nb_pts, elbow_constraint_free)
                if elbow is None:
                    if nb_crosses[index_max] > nb_crosses_limit:
                        i = index_max
                        while nb_crosses[i]>=2 and i<20:
                            i += 1

                        return tresh_vals[i]
                    else:
                        return 0.01
                else:
                    return tresh_vals[min(elbow, 20)]
        return 0


    def nb_of_treshold_crosses(self, data, treshold=None):
        if treshold is None:
            treshold = data.mean()
        return len([i for i in range(data.shape[0]-1) if (data[i+1] - treshold)*(data[i] - treshold)<0])


    def check_if_elbow(self, data, start, end, nb_pts, elbow_constraint_free):
        i = start
        slope=0
        while i < len(data)-1 and data[i+1] == data[i] and i<20:
            i += 1
        while i < len(data)-2 and data[i]-data[i+nb_pts] > 0.1 and i<20:
            i+=1
        if i!=start:
            slope = 100*(data[start]-data[i])/(i-start)
        if elbow_constraint_free or (slope > 15 and data[i] < 0.5*data[start]):
            return i
        return None
    
    
    
    def comp_mean_maxs(self, data):
        if len(data)>2:
            tops = [data[i] for i in range(1, len(data)-1) if data[i]>data[i-1] and data[i]>data[i+1]]
        if len(data)<=2 or not tops:
            return data.max()
        rank = max(round(len(tops)/3), 1)
        relevant_tops = np.sort(tops)[-rank:]
        return np.asarray(relevant_tops).mean()
    
    
    def merge_subevents_into_event(self, boundaries, eps=0.6, delta_t=1):
        eps = round(eps/self.T)
        #print("eps: ", eps)
        merged_boundaries = [boundaries[0]]
        max_vals = []
        cumul_max = self.comp_mean_maxs(self.y[merged_boundaries[0][0]: merged_boundaries[0][1]+1])
        cumul_width = merged_boundaries[0][1] - merged_boundaries[0][0] 
        counter=1
        for i in range(1, len(boundaries)):
            start1, end1 = merged_boundaries[-1]
            start2, end2 = boundaries[i]
            max_diff = abs(cumul_max - self.comp_mean_maxs(self.y[start2: end2+1]))
            treshold = 0.2+0.2*max(cumul_max, (self.y[start2: end2+1]).max())
            merging = start2-end1 <= eps and (start2-end1 < 0 or max_diff <= treshold)

            if merging:
                merged_boundaries[-1][1] = end2
                treshold = 0.2+0.2*max(cumul_max, (self.y[start2: end2+1]).max())
                if max_diff < treshold:
                    coeff = (end2-start2)/(cumul_width+end2-start2)
                    cumul_max = cumul_max*(1-coeff)+self.comp_mean_maxs(self.y[start2: end2+1])*coeff
                else:
                    cumul_max = max(cumul_max, self.comp_mean_maxs(self.y[start2: end2+1]))
                cumul_width += end2-start2
                counter+=1
            else:
                merged_boundaries.append([start2, end2])
                max_vals.append(cumul_max)
                cumul_max = self.comp_mean_maxs(self.y[start2: end2+1])
                cumul_width = end2-start2
                counter=1
                
        max_vals.append(cumul_max)
        
        #print('merged boundaries1: ', merged_boundaries)
                    
        return self.merge_unsimilar_profiles(merged_boundaries, max_vals, eps, delta_t)
    
    def merge_unsimilar_profiles(self, boundaries, max_vals, eps, delta_t):
        delta_ind = int(delta_t/self.T)+1
        if len(boundaries)>1:
            dists = np.asarray([boundaries[i+1][0]-boundaries[i][1] for i in range(len(boundaries)-1)])
            dist_ind = np.argsort(dists)
            dists = dists[dist_ind]

            i=0
            while i<len(dists) and dists[i] < eps:
                bounds = [boundaries[dist_ind[i]], boundaries[dist_ind[i]+1]]
                maxis = [max_vals[dist_ind[i]], max_vals[dist_ind[i]+1]]
                
                low_peak_ind = np.argmin(maxis)
                max_diff = abs(maxis[0] - maxis[1])
                treshold = 0.2+0.2*max(maxis)
                if max_diff < treshold or bounds[low_peak_ind][1] - bounds[low_peak_ind][0] <= delta_ind:
                    width0 = bounds[0][1]-bounds[0][0]
                    width1 = bounds[1][1]-bounds[1][0]
                    coeff = width0/(width0+width1)
                    max_vals[dist_ind[i]+1] = max_vals[dist_ind[i]]*coeff+max_vals[dist_ind[i]+1]*(1-coeff)
                    boundaries[dist_ind[i]+1][0] = boundaries[dist_ind[i]][0]
                    
                    del boundaries[dist_ind[i]]
                    del max_vals[dist_ind[i]]
                    dists = np.delete(dists, i)
                    index=dist_ind[i]
                    dist_ind = np.where(dist_ind>index, dist_ind-1, dist_ind)
                    dist_ind = np.delete(dist_ind, i)
                    i-=1
                i+=1
        return boundaries
        
    
    
    
        
    def lineplot2(self):
        fig, axes = plt.subplots(2, figsize=(15, 8), dpi=1000, sharex=True)
        
        for i in range(len(axes)):
            
            if i == 0:
                x = self.origT
            else:
                x = self.t

            if i==0:
                y = self.origY
            else:
                y = self.y
                
            
            axes[i].set_xlabel('Time (s)')
            axes[i].set_ylabel('Amplitude')
    
            X_ticks_minor = np.arange(0, x.shape[-1], 0.1)
            X_ticks_major = np.arange(0, x.shape[-1], 1)
            axes[i].set_xticks(X_ticks_minor, minor=True)
            axes[i].set_xticks(X_ticks_major, minor=False)
                
     
            if i==1:  
                y_ticks = np.arange(0, 1.1, 0.1)
                axes[i].set_yticks(y_ticks)
            
            axes[i].spines['top'].set_color('white')
            axes[i].spines['right'].set_color('white')
            axes[i].grid(which='both')
            axes[i].grid(which='minor', alpha=0.4)
            axes[i].grid(which='major', alpha=1)
            
        
        #for event in self.events:
         #   print(event.info())
        
            for event in self.events:
                #if original_signal:
                mini, maxi = y.min(), y.max()
                axes[i].fill_between([self.t[event.start], self.t[event.end]], [mini, mini], [maxi, maxi], facecolor='red', alpha=0.1)  
                axes[i].axvline(x=self.t[event.start], linewidth=1, color='r')
                axes[i].axvline(x=self.t[event.end], linewidth=1, color='r')
    
        
        
            axes[i].plot(x, y)
            
            if i == 1:
                t_tresh = np.ones(x.shape[-1])*self.treshold
                axes[i].plot(x, t_tresh, color='g', linewidth=1)
                
        path='/Users/hugomeyer/Desktop/PDM/Smartphone-app-multimedia-smart-selection/Paper/'
        plt.savefig(path + 'Clip_{}.png'.format(self.index))
        
        
        
    
        
    def lineplot(self, x=None, y=None, clip_nb=0, title=None, markers='indices', original_signal=False, save=False):
        if title is None:
            title = ''#'Clip {}'.format(clip_nb)
        if x is None:
            if markers == 'time':
                if original_signal:
                    x = self.origT
                else:
                    x = self.t
            else:
                x = np.arange(len(self.y))
        if y is None:
            if original_signal:
                y = self.origY
            else:
                y = self.y
                
        if save:
            dpi = 1000
        else:
            dpi = 80
                        
        fig, ax = plt.subplots(figsize=(15, 4), dpi=dpi)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        
        if self.L < 30:
            if markers == 'time' or original_signal:
                X_ticks_minor = np.arange(0, x.shape[-1], 0.1)
                X_ticks_major = np.arange(0, x.shape[-1], 1)
                ax.set_xticks(X_ticks_minor, minor=True)
                ax.set_xticks(X_ticks_major, minor=False)
            else:
                X_ticks_minor = np.arange(0, x.shape[-1], 1)
                X_ticks_major = np.arange(0, x.shape[-1], 5)
                ax.set_xticks(X_ticks_minor, minor=True)
                ax.set_xticks(X_ticks_major, minor=False)
        elif self.L < 60:
            if markers == 'time' or original_signal:
                X_ticks = np.arange(0, x.shape[-1], 1)
            else:
                X_ticks = np.arange(0, x.shape[-1], 10)
            ax.set_xticks(X_ticks)
        elif self.L < 180:
            if markers == 'time' or original_signal:
                X_ticks = np.arange(0, x.shape[-1], 5)
            else:
                X_ticks = np.arange(0, x.shape[-1], 20)
            ax.set_xticks(X_ticks)
        else:
            if markers == 'time' or original_signal:
                X_ticks = np.arange(0, x.shape[-1], 10)
            else:
                X_ticks = np.arange(0, x.shape[-1], 50)
            ax.set_xticks(X_ticks)   
 
        if not original_signal:  
            y_ticks = np.arange(0, 1.1, 0.1)
            ax.set_yticks(y_ticks)
        
        ax.spines['top'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.grid(which='both')
        ax.grid(which='minor', alpha=0.4)
        ax.grid(which='major', alpha=1)
        
        
        #for event in self.events:
         #   print(event.info())
        
        for event in self.events:
            if markers == 'time' or original_signal:
                #if original_signal:
                mini, maxi = y.min(), y.max()
                ax.fill_between([self.t[event.start], self.t[event.end]], [mini, mini], [maxi, maxi], facecolor='red', alpha=0.1)  
                plt.axvline(x=self.t[event.start], linewidth=1, color='r')
                plt.axvline(x=self.t[event.end], linewidth=1, color='r')
            else:
                ax.fill_between([event.start, event.end], [0, 0], [1, 1], facecolor='red', alpha=0.1)  
                plt.axvline(x=event.start, linewidth=1, color='r')
                plt.axvline(x=event.end, linewidth=1, color='r')
        
        
        plt.plot(x, y)
        plt.title(title)
        if self.L < 30 and not original_signal:
            plt.scatter(x, y, color='b', s=10, alpha=0.5)
        
        if self.treshold is not None and not original_signal:
            t_tresh = np.ones(x.shape[-1])*self.treshold
            plt.plot(x, t_tresh, color='g', linewidth=1)
        if save:
            path='/Users/hugomeyer/Desktop/PDM/Smartphone-app-multimedia-smart-selection/Paper/'
            plt.savefig(path + 'Clip_{}.png'.format(self.index))
        plt.show()
        
    
    
    
    def info(self):
        print({'Duration': round(self.L,2), 'fs': int(self.fs), 'N': self.N, 'T': round(self.T, 2)})
        
        

    
        
