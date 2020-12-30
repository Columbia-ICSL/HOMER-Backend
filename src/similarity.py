#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 11:53:45 2019

@author: hugomeyer
"""


import numpy as np
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import os




class Similarity_cut(object):
    def __init__(self, index, time, label, score=-1):
        self.index=index
        self.score=score
        self.label=label
        self.time = time
    
    def info(self):
        return {'index': self.index, 'time': self.time, 'score': self.score, 'label': self.label}



    
class Similarity(object):
    def __init__(self, start, end, data, fps):
        if start > len(data) or end > len(data):
            raise ValueError("The defined subscene is out of the scene boundaries.")

        self.start=start
        self.end=end
        self.values = data
        self.indices=np.arange(start, end+1, 1)
        self.tops = []
        self.pits = []
        self.fps = fps
        self.features = []
        self.feature_labels = {
                                    'valley_pit': 1, 
                                    'plateau_start': 2, 
                                    'plateau_end': 3, 
                                    'hill_start': 4, 
                                    'hill_end': 5, 
                                    'hill_top': 6
                              }
            
            

        
    def processing(self):
        #if sum(self.values) != 0:
        self.remove_discontinuities(mu=0.12, eps=0.25, nb_pts=5)
        self.butter_lowpass_filter(2.5)
        self.find_tops_pits(nb_pts=6)
         #   return 1
        #else:
        #    return 0
        
        
        
    def remove_discontinuities(self, mu=0.12, eps=0.25, nb_pts=5):

        signal = np.asarray(self.values).copy()
        
            
        scores = []
        for i in range(len(signal)-nb_pts+1):
            packet = signal[i:i+nb_pts]
            for j in range(nb_pts-1, 2, -1):
                if abs(packet[0]-packet[j]) < mu and min(packet[0], packet[j])-min(packet[:j+1]) > eps:
                    score = np.argmin(packet)+i+1
                    if score not in scores:
                        #scores.append(score)
                        signal[i:i+j+1] = np.linspace(packet[0], packet[j], j+1).tolist()

        self.values = signal.tolist()
        
        
    


    def butter_lowpass_filter(self, freq):

            data = np.asarray(self.values).copy()
            if len(data)>10:
                order = 6
                ratio_pts_used_for_side=0.1
                fs = 30
                shift={1: 27, 2: 12, 2.5: 8, 3: 7, 4: 5, 5: 4, 6: 3}
        
                #Avoid side effects
                ext_size = max(round(ratio_pts_used_for_side*data.size), 1)#, shift[freq]+1)
                sig_shift = min(shift[freq], ext_size-1)
                x1 = np.flipud(data[1:ext_size+1])
                x2 = np.flipud(data[-1-ext_size:-1])
                data = np.insert(data, 0, x1)
                data = np.append(data, x2)
        
                #filtering
                b, a = self.butter_lowpass(freq, fs, order=order)
                y = lfilter(b, a, data)
                inf, sup = ext_size+sig_shift, -ext_size+sig_shift
                self.values = y[inf : sup]

    
    
    
    def butter_lowpass(self, cutoff, fs, order=5):
        """
        PARAMETERS: 
            - cutoff: cutoff frequency determining the boudary of the low pass band o the filter
            - fs: sampling frequency
            - order: order of the filter
            
        DESCRIPTION:
            Compute the filtering coefficients given the chosen parameters
            
        RETURN:
            - b, a: filter coefficients  
        """
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a
            
            
    
    def extract_features(self):
        if self.tops.any() or self.pits.any():
                
            pits_tops, labels = self.detect_pits_tops()
            
            
            #features = [Similarity_cut(int(el[0]), int(el[0])/self.fps, labels=[], score=self.compute_score_top(el))
            #        for el, label in zip(pits_tops, labels) if label=='top']
            
            
            features = self.find_labels(pits_tops, labels)
          
            self.features = np.asarray(features)[np.argsort([cut.index for cut in features])].tolist()
            #self.pit_cuts = self.score_pits(pits_tops, labels)
            
            
            featured_pits_labels = ['hill_start', 'hill_end', 'valley_pit']
            indices_of_featured_pits = [feat.index for feat in self.features if any(elem in feat.label for elem in featured_pits_labels)]
            indices_to_rm = [i for i in range(self.pits.shape[0]) if self.pits[i, 0] in indices_of_featured_pits]

            self.pits = np.delete(self.pits, indices_to_rm, axis=0)

  
   
    
    
        
        
    def find_labels(self, pits_tops, labels):
        features = []
        double_tops = [i for i, (el1, el2) in enumerate(zip(labels[:-1], labels[1:])) if [el1, el2] == ['top', 'top']]
        double_pits = [i for i, (el1, el2) in enumerate(zip(labels[:-1], labels[1:])) if [el1, el2] == ['pit', 'pit']]

        for i in double_tops:
            front_top = pits_tops[i]
            back_top = pits_tops[i+1]
            if i-1 >= 0:
                front_pit = pits_tops[i-1]
            else:
                local_min_left, ind_left = self.find_next_min_local(front_top, 'left', nb_pts=6)
                front_pit = [ind_left, local_min_left]
            if i+2 < len(pits_tops):
                back_pit = pits_tops[i+2]
            else:
                local_min_right, ind_right = self.find_next_min_local(back_top, 'right', nb_pts=6)
                back_pit = [ind_right, local_min_right]
            
            if self.plateau_shaped(front_top, back_top, front_pit, back_pit, spread_factor=1.5):#The higher the spread_factor, the spreader we allow the hill to be
                ## Back top
                front_score = self.compute_score_top(front_top)
                back_score = self.compute_score_top(back_top)
                features.append(Similarity_cut(int(front_top[0]), int(front_top[0])/self.fps, 
                                                   label='plateau_start', score=front_score))
                features.append(Similarity_cut(int(back_top[0]), int(back_top[0])/self.fps, 
                                                   label='plateau_end', score=back_score))
                '''
                if i-1 >= 0:
                    features.append(Similarity_cut(int(pits_tops[i-1][0]), int(pits_tops[i-1][0])/self.fps, 
                                                   label='hill_start', score=front_score))
                if i+2 < len(pits_tops):
                    features.append(Similarity_cut(int(pits_tops[i+2][0]), int(pits_tops[i+2][0])/self.fps, 
                                                   label='hill_end', score=back_score))
                '''
        for i in double_pits:
            if i-1 >= 0:
                features.append(Similarity_cut(int(pits_tops[i-1][0]), int(pits_tops[i-1][0])/self.fps, 
                                                   label='valley_start', score=self.compute_score_top(pits_tops[i-1])))
            if i+2 < len(pits_tops):
                features.append(Similarity_cut(int(pits_tops[i+2][0]), int(pits_tops[i+2][0])/self.fps, 
                                                   label='valley_end', score=self.compute_score_top(pits_tops[i+2])))
        
        already_visited = [i for i, el in enumerate(pits_tops) if el[0] in [feat.index for feat in features]]
        single_tops = [i for i, label in enumerate(labels) if label == 'top' and i not in already_visited]
        
        for i in single_tops:
            single_top = pits_tops[i]
            if i > 0:
                # Replace top by its previous pit if the slope is low 
                right_min_local, _ = self.find_next_min_local(single_top, 'right')
                ratio = (single_top[1]-right_min_local)/(single_top[1]-pits_tops[i-1][1])
                time_ratio = (single_top[1]-pits_tops[i-1][1])*100/(single_top[0]-pits_tops[i-1][0])
                if ratio < 0.25 and ratio > 0 and labels[i-1] != 'top' and time_ratio<2:

                    features.append(Similarity_cut(int(pits_tops[i-1][0]), int(pits_tops[i-1][0])/self.fps, 
                                                   label='valley_pit', score=self.compute_score_top(single_top)))
                    
           # if not pit_replaced_top:
            features.append(Similarity_cut(int(single_top[0]), int(single_top[0])/self.fps, 
                                                   label='hill_top', score=self.compute_score_top(single_top)))
        return features
        
        
    def find_tops_pits(self, nb_pts):
        tops = []
        pits = []

        data =self.values
        nb_pts = min(nb_pts, int(len(data)/2))
        going_down = data[0] > data[nb_pts]
        old_moving_avg = data[0]
        for i in range(len(data)-nb_pts+1):
            new_moving_avg = sum(data[i:i+nb_pts])/len(data[i:i+nb_pts])
            if going_down and new_moving_avg>old_moving_avg:
                pits.append(i+np.argmin(data[i:i+nb_pts])+1)
                going_down = False
            if not going_down and new_moving_avg<old_moving_avg:
                tops.append(i+np.argmax(data[i:i+nb_pts])+1)
                going_down = True
            old_moving_avg = new_moving_avg
    
        #Ensure than tops and pits at the edges of the clips are taken into account
        if max(data[-nb_pts:]) != data[-nb_pts] and max(data[-nb_pts:]) != data[-1]:
            index = len(data)-nb_pts+np.argmax(data[-nb_pts:])+1
            if index not in tops:
                tops.append(index)
            
        if max(data[:nb_pts]) != data[0] and max(data[:nb_pts]) != data[nb_pts-1]:
            index = np.argmax(data[:nb_pts])+1
            if index not in tops:
                tops.append(index)
            
        if min(data[-nb_pts:]) != data[-nb_pts] and min(data[-nb_pts:]) != data[-1]:
            index = len(data)-nb_pts+np.argmin(data[-nb_pts:])+1
            if index not in pits:
                pits.append(index)
            
        if min(data[:nb_pts]) != data[0] and min(data[:nb_pts]) != data[nb_pts-1]:
            index = np.argmin(data[:nb_pts])+1
            if index not in pits:
                pits.append(index)
                
        tops.sort()
        pits.sort()
     
        
        
        if tops and pits:
            tops, pits = self.keep_meaningful_tops_pits(tops, pits)
        elif tops:
            tops = [top for top in tops 
                      if max(data[top] - self.find_next_min_local([top, data[top]], 'right')[0], 
                             data[top] - self.find_next_min_local([top, data[top]], 'left')[0]) 
                      > 0.1]
        else:
            pits = [pit for pit in pits 
                      if max(data[pit] - self.find_next_max_local([pit, data[pit]], 'right')[0], 
                             data[pit] - self.find_next_max_local([pit, data[pit]], 'left')[0]) 
                      > 0.1]
        
            
        pits=np.asarray([(pit_ind, data[pit_ind-1]) for pit_ind in pits])
        tops=np.asarray([(top_ind, data[top_ind-1]) for top_ind in tops])

        self.tops, self.pits = tops, pits
 
        

   
    def keep_meaningful_tops_pits(self, tops, pits):

        data =self.values
        seq = np.sort(tops+pits)
        remove=[]
        start=0
        treshold=0.1
        if pits[0]<tops[0]:
            start=1
            if abs(data[pits[0]-1]-data[tops[0]-1]) < treshold:
                remove.append(pits[0])
        for i in range(start, len(seq)-1, 2):
            diff=abs(data[seq[i]-1]-data[seq[i+1]-1])
            if diff < 0.15:
                #forw = i+1
                #while forw < len(seq) and 
                if i == 0:
                    remove.append(seq[i])
                else:
                    back=i-1
                    while back >= 0 and (seq[back] in remove or (seq[back] in tops and data[seq[back]]<data[seq[i]])):
                        back-=1
                    if (abs(data[seq[i]]-data[seq[back]]) < 0.15 or back<0): #and abs(data[seq[i+1]]-data[seq[back]]) < 0.8:
                        remove.append(seq[i])
                if i<len(seq)-2 and abs(data[seq[i+1]]-data[seq[i+2]])<0.15:
                    remove.append(seq[i+1])
        if pits[-1] < tops[-1] and tops[-1]-pits[-1]<15:
            if abs(data[pits[-1]-1]-data[tops[-1]-1]) < treshold:
                remove.append(tops[-1])
                
        remove = self.respawn_first_top_of_deleted_serie(seq, tops, remove, 6)
        
                
        tops = [top for top in tops if top not in remove]
        pits = [pit for pit in pits if pit not in remove]
        seq = [el for el in seq if el not in remove]
        
        seq, pits = self.rm_local_pits(seq, pits, 0.1, 0.1)
        tops = self.rm_more_than_2_consec_tops(seq, tops)
                
        return tops, pits
                
        
        
    def rm_local_pits(self, pits_tops, pits, tresh1, tresh2):

        data =self.values
        remove=[]
        for i in range(len(pits)-2):
            triangle_basis_eps = abs(data[pits[i]-1]-data[pits[i+2]-1])
            triangle_top_eps = data[pits[i+1]-1]-(data[pits[i]-1]+data[pits[i+2]-1])/2
            if triangle_basis_eps<tresh1 and triangle_top_eps>tresh2:
                remove.append(pits[i+1])
        
        pits = [pit for pit in pits if pit not in remove]
        return [el for el in pits_tops if el not in remove], pits
        
        
    
    
    def respawn_first_top_of_deleted_serie(self, tops_pits, tops, remove, nb_min_el_in_serie):
        if len(remove) < 2:
            return remove
        
        indices = np.asarray([i for i, el in enumerate(tops_pits) if el in remove])
        indices_shifted = np.append(indices[1:], indices[-1]+1)
        diff = indices_shifted-indices-1
        counter = 0
        respawned_indices=[]
        first_index=0
        for i, el in enumerate(diff):
            if el:
                if counter >= nb_min_el_in_serie-1 and indices[i]+1 < len(tops_pits):
                    if tops_pits[indices[i]+1] in tops:
                        tops_pits[indices[i]]
                        respawned_indices.append(first_index)
                counter=0
            else:
                if counter == 0:
                    first_index=i
                counter+=1
        if counter >= nb_min_el_in_serie-1 and indices[i]+1 < len(tops_pits):
            if tops_pits[indices[i]+1] in tops:
                respawned_indices.append(first_index)
        respawned_indices = [ind if remove[ind] in tops else ind+1 for ind in respawned_indices]
        return np.delete(remove, respawned_indices).tolist()
    
    
    
    
    
    def rm_more_than_2_consec_tops(self, tops_pits, tops):
        rm_consec = [tops_pits[i] for i in range(1, len(tops_pits)-1) 
                     if tops_pits[i-1] in tops and tops_pits[i] in tops and tops_pits[i+1] in tops]

        return[top for top in tops if top not in rm_consec]
            

        
        
    
    
    def detect_pits_tops(self):

        if not self.tops.any():
            pits_tops=self.pits
        elif not self.pits.any():
            pits_tops=self.tops
        else:
            pits_tops = np.concatenate((self.tops, self.pits))
        
        indices = np.argsort(pits_tops[:, 0])
        pits_tops = pits_tops[indices]
        labels = np.asarray(['top']*self.tops.shape[0] + ['pit']*self.pits.shape[0])[indices]
        
        
        return pits_tops, labels
    
    
    
    
    def score_pits(self, pits_tops, labels):
        pits = [pits_tops[i] for i in range(len(pits_tops)) if labels[i]=='pit']
        return [Similarity_cut(int(pit[0]), pit[0]/self.fps, [], score=self.compute_score_pit(pit)) for pit in pits]
        
        

        
    def compute_score_top(self, top):
        left_min_local, _ = self.find_next_min_local(top, 'left', nb_pts=6)
        right_min_local, _ = self.find_next_min_local(top, 'right', nb_pts=6)

        return ((abs(left_min_local-right_min_local)**2*(top[1]-min(left_min_local, right_min_local)))**(1./3))*2
        
        
    
    def compute_score_pit(self, pit):
        left_max_local, _ = self.find_next_max_local(pit, 'left')
        right_max_local, _ = self.find_next_max_local(pit, 'right')

        return ((abs(left_max_local-right_max_local)**2*(max(left_max_local, right_max_local)-pit[1]))**(1./3))*2
    
        
      
    def find_next_min_local(self, top, side, nb_pts=10):
        nb_pts = min(nb_pts, int(len(self.values)/2))
        i = int(top[0])-self.start
        old_moving_avg = self.values[i]
        if side == 'right':
            while i+nb_pts < len(self.values)+1:
                new_moving_avg = sum(self.values[i:i+nb_pts])/len(self.values[i:i+nb_pts])
                if new_moving_avg>old_moving_avg:
                    min_index=i+np.argmin(self.values[i:i+nb_pts])+1+self.start
                    min_value=self.values[min_index-self.start]
                    return min_value, min_index#[min_index, min_value]

                i+=1
                old_moving_avg = new_moving_avg
            
            return self.values[-1], len(self.values)
        else:
            while i-nb_pts+1 >= 0:
                new_moving_avg = sum(self.values[i-nb_pts+1:i+1])/len(self.values[i-nb_pts+1:i+1])
                if new_moving_avg>old_moving_avg:
                    min_index=i-np.argmin(self.values[i-nb_pts+1:i+1][::-1])+self.start
                    min_value=self.values[min_index-self.start]
                    return min_value, min_index#[min_index, min_value]

                i-=1
                old_moving_avg = new_moving_avg
                
            return self.values[0], self.start#[self.start, ]
        


    def find_next_max_local(self, pit, side, nb_pts=10):
        nb_pts = min(nb_pts, int(len(self.values)/2))
        i = int(pit[0])-self.start
        old_moving_avg = self.values[i]
        if side == 'right':
            while i+nb_pts < len(self.values)+1:
                new_moving_avg = sum(self.values[i:i+nb_pts])/len(self.values[i:i+nb_pts])
                if new_moving_avg<old_moving_avg:
                    max_index=i+np.argmax(self.values[i:i+nb_pts])+1+self.start
                    max_value=self.values[max_index-self.start]
                    return max_value, max_index#[min_index, min_value]

                i+=1
                old_moving_avg = new_moving_avg
            
            return self.values[-1], len(self.values)
        else:
            while i-nb_pts+1 >= 0:
                new_moving_avg = sum(self.values[i-nb_pts+1:i+1])/len(self.values[i-nb_pts+1:i+1])
                if new_moving_avg<old_moving_avg:
                    max_index=i-np.argmax(self.values[i-nb_pts+1:i+1][::-1])+self.start
                    max_value=self.values[max_index-self.start]
                    return max_value, max_index#[min_index, min_value]

                i-=1
                old_moving_avg = new_moving_avg
            return self.values[0], self.start#[self.start, ]


    
    def plateau_shaped(self, front_top, back_top, front_pit, back_pit, spread_factor=1.5):
        plateau_start = int(front_top[0])
        plateau_end = int(back_top[0])
        
        min_betw_tops = self.values[plateau_start:plateau_end+1].min()
        #local_min_left, ind_left = self.find_next_min_local(front_top, 'left', nb_pts=6)
        #local_min_right, ind_right = self.find_next_min_local(back_top, 'right', nb_pts=6)

        local_mins=[front_pit[1], back_pit[1]]
        indices = [front_pit[0], back_pit[0]]
        tops = [front_top, back_top]
        
        local_min = local_mins[np.argmax(local_mins)]
        top = tops[np.argmax(local_mins)]
        local_index = indices[np.argmax(local_mins)]
        
        time_diff = abs(local_index-top[0])/self.fps
        diff_value = top[1]-local_min
        if (min_betw_tops - local_min) < 0.05 or time_diff/(10*diff_value)>spread_factor:
            return False
        elif abs(back_top[1]-front_top[1])<0.1:
            return True
        else:
            return False
        
        
        
        
    
    def lineplot_analog(self, clip_ID, yticks=0.1, duration=None, fps=None, title=None,
                        tops_pits=True, save=False, path='../'):
        #plt.figure(num=None, , dpi=80, facecolor='w', edgecolor='k')
        if title is None:
            title = ''
        if duration is None:
            x_label='Frames'
        else:
            x_label = 'Time'
        y_label='Similarity'

        
        factor = 1
        if duration is not None:
            factor = 1/self.fps
        elif fps is not None:
            factor = fps/self.fps
            
        x = np.asarray(self.indices)*factor#+self.offset
        y = np.asarray(self.values)
    
        if save:
            dpi = 80
        else:
            dpi = 80
    
        fig, ax = plt.subplots(figsize=(10, 4), dpi=dpi)
        
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
        
        X_ticks_maj = np.arange(max(int(x[0])-factor, 0), int(x[-1])+2, step)
        y_ticks = np.arange(int(y.min()), round(y.max(), 1), yticks)
        X_ticks_min = np.arange(x[0]-factor, x[-1]+1, step*0.2)
        X_ticks_min[0] = int(x[0])
        
        if duration is not None:
            X_ticks_maj[0] = 0
        else:
            X_ticks_maj[0] = 1
        
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
            
        plt.plot(x, y)
        
        if tops_pits:
            if self.features:
                for feat in self.features:
                    if any(elem in feat.label for elem in ['hill_start', 'hill_end']):
                        color='k'
                    elif any(elem in feat.label for elem in ['plateau_start', 'plateau_end']):
                        color='b'
                    elif any(elem in feat.label for elem in ['valley_start', 'valley_end']):
                        color='m'
                    else:
                        color='g'
                        
                    plt.scatter(feat.time, self.values[feat.index-1], color=color, s=180, alpha=0.2)
                    plt.scatter(feat.time, self.values[feat.index-1], color=color, s=40, alpha=1)
                    score = '{:.2f}' .format(feat.score)
                    ax.annotate(score, (feat.time, self.values[feat.index-1]))
            
            if self.pits.any():

                pits_ind = self.pits[:, 0].astype(int)-1

                indices = self.indices[pits_ind]*factor
                plt.scatter(indices, self.pits[:, 1], color='r', s=180, alpha=0.2)
                plt.scatter(indices, self.pits[:, 1], color='r', s=40, alpha=1)
           
            
        string = ''

            
        if save == True:
            plt.savefig(os.path.join(path, 'Clip_{}_similarity2' .format(clip_ID)+string+'.png'))
        else:
            plt.show()
    
    
        