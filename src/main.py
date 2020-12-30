#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 13:11:42 2019

@author: hugomeyer
"""


#%% Import libraries and initialize the clip

from clip import Clip
import os
import sys

#clip_ids = pd.read_excel(targets_file).dropna().clip_nb.tolist()
#for clip_nb in clip_ids[23:50]:
if len(sys.argv) == 1:
    #clip_nb = 10
    back_video_path='/Volumes/HUGUETTE/Datasets/Experiment3/Evaluation/Back/Clip_{}.mp4' .format(clip_nb)
    #back_video_path='/Users/hugomeyer/Desktop/PDM/Smartphone-app-multimedia-smart-selection/Final/Video/back/back_camera.mp4'
    front_video_path='/Volumes/HUGUETTE/Datasets/Experiment3/Evaluation/Front/Clip_{}.mp4' .format(clip_nb)
    #front_video_path='/Users/hugomeyer/Desktop/PDM/Smartphone-app-multimedia-smart-selection/Final/Video/front/front_camera.mp4'
    hl_min_time=-1
    hl_max_time=-1
    video_name = back_video_path.split('/')[-1].split('.')[0]
    hl_export="/Volumes/HUGUETTE/Datasets/Experiment3/Evaluation/Highlight"
    many_hls=False
elif len(sys.argv) == 7 or len(sys.argv) == 6:
    back_video_path = sys.argv[1]
    front_video_path = sys.argv[2]
    hl_export = sys.argv[3]
    hl_min_time = int(sys.argv[4])
    hl_max_time = int(sys.argv[5])
    many_hls = bool(sys.argv[6]) if len(sys.argv) == 7 else False
    video_name = back_video_path.split('/')[-1].split('.')[0]
    hl_export = os.path.join(hl_export, video_name) + '.mp4'
else:
    raise ValueError('Not enough input arguments. Should be respectively: input path, output path, min hl time, max hl time.')


targets_file = '../Data/Experiment3/labels_fps10.xlsx'

simil_fps=10
emo_fps = 4
emo_labels=['p7_hap', 'p7_surp']
audio_import_path = '.'.join(back_video_path.split('.')[:-1]) + '.wav'
export = False


if export and not os.path.isdir('/'.join(hl_export.split('/')[:-1])):
    os.makedirs('/'.join(hl_export.split('/')[:-1]))
    
clip = Clip(ID=video_name, back_video_path=back_video_path, 
            front_video_path=front_video_path, trimmed_video_export=hl_export) 

# %% Compute audio

succeeded = clip.compute_audio_signal(audio_import_path)
print("Audio: " + str(succeeded))
if succeeded:
    clip.sound.events_recognition()
    print(clip.sound.event_labels)
    clip.sound.segmentation_preprocess()
    clip.sound.events_segmentation()
    print(clip.sound.events)

#clip.sound.lineplot(clip_nb=clip_nb, markers='time', original_signal=True, save=export)

# %% Compute and process emotion
file_found = clip.compute_emotions(emo_labels, fps=emo_fps)
print("Emotion: " + str(file_found))
if file_found:
    clip.emotion.extract_features()
    print(clip.emotion.preds)
    clip.emotion_trigger = clip.emotion is not None and clip.emotion.best_emo_label is not None

# %% Compute similarity
clip.compute_img_similarity(simil_fps, importing_time_limit=30, ratio=0.1, bin_size=2)

# %% Process similarity
clip.similarity.processing()
clip.similarity.extract_features()
print(clip.similarity.features)

# %% Build event timeline
clip.compute_events_timeline(fps=10)
clip.events_timeline.compute_highlight(hl_min_size=hl_min_time, hl_max_size=hl_max_time, 
                                       many_hls=many_hls, hl_overlap_ratio=0.25, hl_duration_ratio=0.33, 
                                       max_hl_nb=3, rm_tresh_score=0.15)

# %% Export Highlight
clip.events_timeline.export_hl(back_video_path, hl_export)
