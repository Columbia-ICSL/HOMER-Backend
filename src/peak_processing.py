#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:47:07 2019

@author: hugomeyer
"""



import numpy as np
from sklearn.cluster import KMeans 



class Peak(object):
    def __init__(self, emotion, start, end, avg, maxi, treshold):
        self.emotion = emotion
        self.start = start
        self.end = end
        self.rise_start = None
        self.fall_end = None
        self.T = end-start+1
        self.max = maxi
        self.avg = avg
        self.treshold = treshold
        self.only_a_part = False
        
    def info(self):
        return {
            'emotion':self.emotion,
            'rise start': self.rise_start,
            'start': self.start,
            'end': self.end,
            'fall end': self.fall_end,
            'length': self.T,
            'max': self.max,
            'average': self.avg,
            'treshold': self.treshold
        }
        
        
        
        


def peak_detection(preds, emo_length, emo_label, emo_dict, 
                   max_peak_size, peak_detect_treshold=0.8):
    peaks=[]
    emo_best_pred = preds['7_best'].values
    count_consec_emo = longest_consec_subseq(emo_best_pred, emo_dict[emo_label], emo_best_pred.shape[0] , 0, 0)
    if count_consec_emo/emo_length < 0.8:
        starts, ends, avgs, maxis = find_start_and_end(preds, emo_label, peak_detect_treshold)
        
        
        for i in range(len(starts)):
            #ratio_peak_clip = (ends[j]-starts[j])/emo_length
            peaks.append(Peak(emo_label, starts[i], ends[i], avgs[i], maxis[i], peak_detect_treshold))
     
        
        peaks=merge_neighbor_peaks(peaks, emo_label)
     
        rise_starts = find_rise_start(preds, peaks, emo_label)
        
        fall_ends = find_fall_end(preds, peaks, emo_label)
        rise_starts, fall_ends = remove_any_overlap(rise_starts, fall_ends, preds, peaks, emo_label)
        counter=0
        for i in range(len(peaks)):
            if peaks[i].emotion == emo_label:
                peaks[i].rise_start = rise_starts[counter]
                peaks[i].fall_end = fall_ends[counter]
                counter+=1

            
        #lineplot_analog(clips[i], [emo_label], 'probabilities', peaks=True, save=False)
        
    return compute_peak_scores(peaks, emo_label, max_peak_size)




def remove_any_overlap(rise_starts, fall_ends, preds, peaks, emo_label):
    data = preds[emo_label]
    for i in range(len(rise_starts)-1):
        prev_end, next_rise = fall_ends[i], rise_starts[i+1]
        next_rise = 0 if next_rise is None else next_rise
        prev_end = data.shape[0]-1 if prev_end is None else prev_end
        if next_rise < prev_end:
            min_index = np.argmin(data[peaks[i].end-1:peaks[i+1].start].values) + peaks[i].end
            fall_ends[i] = min_index
            rise_starts[i+1] = min_index
    return rise_starts, fall_ends




def compute_peak_scores(peaks, emotion, max_peak_size):
    if len(peaks):
        best_peak, scores = find_best_peak(peaks, emotion, max_peak_size)
        if sum(scores) == 0:
            best_peak = None
        for j in range(len(peaks)):
            if peaks[j].emotion == emotion:
                peaks[j].score = scores[j]
    #elif len(peaks)==1 and peaks[0].emotion == emotion:
     #   best_peak = peaks[0]
      #  peaks[0].score = 1
    else:
        best_peak = None
    return peaks, best_peak
    
  
    
  
def check_if_peak_in_peak(peak, data):
    if data.max()-data.min()<0.2 or data.std()*100<8:
        return None
    high_val = find_high_val_kmean(data)
    indices = high_val.index.values
    high_val = high_val.values
    indices_shifted = np.append(indices[1:], indices[-1]+1)
    boundaries = list(np.nonzero(indices_shifted-indices-1)[0]+1)
    grps, grps_indices = np.split(high_val, boundaries), np.split(indices, boundaries)
    cut_index=grps_indices[np.argmax([grp.max() for grp in grps])][0]
    return cut_index



    
    
def find_start_and_end(preds, emo_label, peak_detect_treshold):
    peak_starts = []
    peak_ends = []
    peak_avgs=[]
    data = preds[emo_label]

    tops, tops_ind = find_peaks_tops(data, peak_detect_treshold)
    #print("tops: ", tops_ind)
    for top, top_ind in zip(tops, tops_ind):
        left_edge = find_left_peak_edge(data, top, top_ind, peak_starts, peak_ends)
        right_edge = find_right_peak_edge(data, top, top_ind, peak_starts, peak_ends)

        #Prevent single value peak
        if left_edge is not None and right_edge is not None: #left_edge != right_edge and 
            peak_starts.append(left_edge)
            peak_ends.append(right_edge)
            peak_avgs.append(np.average(data[left_edge-1:right_edge].values))

    return peak_starts, peak_ends, peak_avgs, tops
    




        
def find_peaks_tops(data, treshold):
    
    
    high_val = find_high_val_kmean(data)
    indices = high_val.index.values
    high_val = high_val.values

    
    if high_val.shape[0]>1:
        indices, high_val = remove_single_val_peaks(indices, high_val)
    
    
    indices_shifted = np.append(indices[1:], indices[-1]+1)
    boundaries = list(np.nonzero(indices_shifted-indices-1)[0]+1)
    
    tops = [max(grp) for grp in np.split(high_val, boundaries) if max(grp)>treshold]
    tops_ind = [np.argwhere(data == top)[0][0]+1 for top in tops]
        
    return tops, tops_ind
        
        
def remove_single_val_peaks(indices, high_val):
    indices_to_rm = []
    ind_size=indices.shape[0]
    #Peaks with a single value a removed
    if indices[1]-indices[0] != 1:
        indices_to_rm.append(0)
    if indices[ind_size-1]-indices[ind_size-2] != 1:
        indices_to_rm.append(ind_size-1)
    if ind_size > 2:
        for i in range(2, ind_size-1):
            if indices[i]-indices[i-1] != 1 and indices[i+1]-indices[i] != 1:
                indices_to_rm.append(i)
    if len(indices_to_rm) < ind_size:
        indices = np.delete(indices, indices_to_rm)
        high_val = np.delete(high_val, indices_to_rm)
    return indices, high_val



def no_peaks_overlap(index, peak_starts, peak_ends, side):
    if side == 'left':
        for start, end in zip(peak_starts, peak_ends):
            if index >= start and index <= end + 1:
                return False
    else:
        for start, end in zip(peak_starts, peak_ends):
            if index >= start-1 and index <= end:
                return False
    return True
    

def find_high_val_kmean(signal):
    kmeans = KMeans(n_clusters=3, n_init=10, tol=1e-4).fit(signal.values[:, np.newaxis])
    centroids = kmeans.cluster_centers_.reshape(-1)
    labels=kmeans.labels_
    class_label = np.argsort(centroids)[2]
#    threshold = min(signal[labels==class_label])
    return signal[labels==class_label]

    
       
def find_left_peak_edge(data, top, top_ind, peak_starts, peak_ends):
    i = top_ind
    top_at_one_peak_side=True    
    if i>1:
        if i==2:
            ratio = data[i]-data[i-1]
        else:
            ratio = data[i]-data[i-2]
        #print(i, 0.2*top, ratio, no_later_decrease(data, i, 'left'))
        if no_peaks_overlap(i, peak_starts, peak_ends, 'left'):
            
            while (ratio<0.2*top or no_later_decrease(data, i, 'left')) and no_peaks_overlap(i, peak_starts, peak_ends, 'left'):
                top_at_one_peak_side=False
                i -= 1
                if i < 2:
                    break
                if i==2:
                    ratio = data[i]-data[i-1]
                else:
                    ratio = max(data[i]-data[i-2], data[i-1]-data[i-2])
                #print(i, 0.2*top, ratio, no_later_decrease(data, i, 'left'))
            if i>1:
                if top_at_one_peak_side == False or (data[i]-data[i-1])<0.2*top:
                    i -= 1
        else:
            return None
    return i


def find_right_peak_edge(data, top, top_ind, peak_starts, peak_ends):
    i = top_ind
    top_at_one_peak_side=True

    if i<data.shape[0]:
        if i==data.shape[0]-1:
            ratio = data[i]-data[i+1]
        else:
            ratio = data[i]-data[i+2]
        if no_peaks_overlap(i, peak_starts, peak_ends, 'right'):
            while (ratio<0.2*top or no_later_decrease(data, i, 'right')) and no_peaks_overlap(i, peak_starts, peak_ends, 'right'):

                top_at_one_peak_side=False
                i+=1
                if i > data.shape[0]-1:
                    break
                if i==data.shape[0]-1:
                    ratio = data[i]-data[i+1]
                else:
                    ratio = max(data[i]-data[i+2], data[i+1]-data[i+2])

            if i<data.shape[0]:
                if top_at_one_peak_side == False or data[i]-data[i+1]<0.2*top:
                    i += 1
        else:
            return None
    return i



 
def find_rise_start(preds, peaks, emo_label):
    data = preds[emo_label]
    peak_rises=[]
    treshold=0.5
    for peak in peaks:
        if peak.emotion == emo_label:
            i = peak.start
            if i>1:
                if i==2:
                    diff = data[i]-data[i-1]
                else:
                    diff = data[i]-data[i-2]
                #print(i, ratio, not_below_treshold(i, data, treshold, 'rise'), data[i]-data[i-1], (data[i]-data[i-1])/(data[i+1]-data[i]))
                while diff>0.1 or not_below_treshold(i, data, treshold, 'rise') or (data[i]-data[i-1]>0 and (data[i]-data[i-1])/(data[i+1]-data[i])>0.15):
                    i -=1
                    if i<2:
                        if data[i]>treshold:
                            i = None
                        break
                    if i==2:
                        diff = data[i]-data[i-1]
                    else:
                        diff = data[i]-data[i-2]
                    #print(i, ratio, not_below_treshold(i, data, treshold, 'rise'), data[i]-data[i-1], (data[i]-data[i-1])/(data[i+1]-data[i]))
                if i is not None:
                    if i>2:
                        if (data[i]-data[i-1])-(data[i-1]+data[i-2])>0.1 and (data[i]-data[i-1])>0:
                            i -= 1
            else:
                i = None
                
            peak_rises.append(i)

            
    return peak_rises


def find_fall_end(preds, peaks, emo_label):
    data = preds[emo_label]
    peak_falls=[]
    treshold=0.5
    
    for peak in peaks:
        if peak.emotion == emo_label:
            i = peak.end
            if i<data.shape[0]:
                if i==data.shape[0]-1:
                    ratio = data[i]-data[i+1]
                else:
                    ratio = data[i]-data[i+2]
                
                while ratio>0.1 or not_below_treshold(i, data, treshold, 'fall') or (data[i]-data[i+1]>0 and (data[i]-data[i+1])/(data[i-1]-data[i])>0.15):
                    i +=1
                    if i>data.shape[0]-1:
                        if data[i]>treshold:
                            i = None
                        break
                    if i==data.shape[0]-1:
                        ratio = data[i]-data[i+1]
                    else:
                        ratio = data[i]-data[i+2]
                if i is not None:
                    if i<data.shape[0]-1:
                        if (data[i]-data[i+1])-(data[i+1]+data[i+2])>0.1 and (data[i]-data[i+1])>0:
                            i += 1
            else:
                i = None
            peak_falls.append(i)
    return peak_falls



def not_below_treshold(i, data, treshold, side):
    if side=='fall':
        if i == data.shape[0]-1:
            return data[i]>treshold and data[i+1]>treshold
        else:
            return data[i]>treshold and data[i+1]>treshold and data[i+2]>treshold
    else:
        if i == 2:
            return data[i]>treshold and data[i-1]>treshold
        else:
            return data[i]>treshold and data[i-1]>treshold and data[i-2]>treshold
        
        
    
def no_later_decrease(data, ind, side):
    nb_pts_used=6
    treshold=0.6
    counter=0
    if side == 'left':
        if ind > nb_pts_used:
            for i in range(1, nb_pts_used):
                if data[ind-i]<treshold:
                    counter+=1
        else:
            return False
    if side == 'right':
        if data.shape[0]-ind+1 > nb_pts_used:
            for i in range(1, nb_pts_used):
                if data[ind+i]<treshold:
                    counter+=1
        else:
            return False
    #if counter>=nb_pts_used:
    if counter!=0:
        return False
    return True



        
def find_best_peak(peaks, emotion, max_peak_size):
    scores = []
    for peak in peaks:
        if peak.emotion == emotion:
            
            L_score = min(1, peak.T/max_peak_size)**2
            #L_score = peak.treshold+min(1, peak.T/max_cut_size)*(1-peak.treshold)
            A_score = max((peak.avg-peak.treshold+0.1), 0)*max((peak.max-peak.treshold+0.1), 0)/(1-peak.treshold)**2
            score = (L_score*A_score)**(1/4)
          
            
        else:
            score = 0
        #print(L_score)
        scores.append(score)
        
    best_peak = peaks[np.argmax(scores)]
    best_peak.score = max(scores)
    return best_peak, scores



def merge_neighbor_peaks(peaks, emotion):     
    if len(peaks)>1:
        i = 0
        while i < len(peaks)-1:
            if peaks[i+1].emotion == emotion and peaks[i].emotion == emotion:
                if peaks[i+1].start - peaks[i].end <= 1:
                    peaks[i].end = peaks[i+1].end
                    del peaks[i+1]
                    i -= 1
            i += 1
      
    return peaks
                
    
def remove_overlap(peaks, max_cut_size):
    if len(peaks)>1:
        peak_to_rm=[]
        for i in range(len(peaks)-1):
            if peaks[i].fall_end > peaks[i+1].rise_start:
                conflicted_peaks=[peaks[i], peaks[i+1]]
                _, scores = find_best_peak(conflicted_peaks, max_cut_size)
                peak_to_rm.append(i+np.argmin([scores]))
        for label in peak_to_rm:
            del peaks[label]
    return peaks


def longest_consec_subseq(X, Y, m, count, maxi): 
    if m == 0: 
        return maxi
    elif X[m-1] == Y: 
        count+=1
    else:
        maxi = max(maxi, count)
        count=0
    return longest_consec_subseq(X, Y, m-1, count, maxi)







#def wavelet_filtering(signal, nb_coeffs, lvl):
 #   coeffs = pywt.wavedec(signal, 'db1', level=lvl)
  #  return pywt.waverec(coeffs[:nb_coeffs], 'db1')
