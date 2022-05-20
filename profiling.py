
from torchsummary import summary
import torchvision
import model.c2d as c2d
import os
import numpy as np
from torchvision.utils import save_image, make_grid
from torchvision.models import vgg19

import torch.nn as nn
import torch
import math
import cv2
from datetime import datetime
import time
import pandas as pd

def vid_dataloader(vid_folder, audio_file, vid_fps, duration,batch_size):
    df = pd.read_csv(audio_file,header=None)
    vid_files = []
    for vid_file in os.listdir(vid_folder):
        vid_time_stamp = int(time.mktime(datetime.strptime(vid_file[0:-4], '%Y-%m-%d_%H-%M-%S').timetuple()))
        vid_obj = cv2.VideoCapture(os.path.join(vid_folder,vid_file))    
        vid_files.append([vid_obj,vid_obj.get(cv2.CAP_PROP_FRAME_COUNT),vid_time_stamp])

    print('files read')

    while(1):
        indices = np.random.randint(0,len(vid_files),size=batch_size)
        x_batch = []
        y_batch = []
        for i in indices:
            vid_tuple = vid_files[i]
            last_second = int(vid_tuple[1]/vid_fps) - duration
            if last_second < 0:
                continue

            # print(int(vid_tuple[1]/vid_fps))
            # print(last_second)
            vid_start_second =  np.random.randint(0,last_second+1)
            audio_start_tstamp = vid_tuple[2] + vid_start_second + math.ceil(duration/2)
            video_snippet_start_index = vid_start_second*vid_fps
            video_snippet_end_index = video_snippet_start_index + (duration*vid_fps)
            vid_tuple[0].set(cv2.CAP_PROP_POS_FRAMES,video_snippet_start_index)

            y = df[df[0] == audio_start_tstamp].drop(columns=[1]).to_numpy()
            if y.shape[0] < 1:
                continue

            frames = []
            for j in range(video_snippet_start_index,video_snippet_end_index):
                
                retval,frame = vid_tuple[0].read()
                if not retval:                    
                    print(retval,video_snippet_start_index,j,vid_tuple[1],vid_start_second,last_second)
                frames.append(frame)
            x_batch.append(np.concatenate(frames,axis=2))
            y_batch.append(y[0])
        y_batch = np.array(y_batch).astype(np.float32)
        x_batch = np.array(x_batch).astype(np.float32)
        width = x_batch.shape[2]
        x_batch = np.concatenate([x_batch[...,:int(width/2),:],x_batch[...,int(width/2):,:]],axis=3)
        # yield [np.array(x_batch),np.array(y_batch)]
        yield [torch.transpose(torch.tensor(x_batch),1,3),torch.tensor(y_batch)]



duration = 5
vid_fps = 5
batch_size = 64
vid_folder = 'G:/.shortcut-targets-by-id/1VaRYG8M-m7nfxKooARU6qTiHpVPhNHuV/out'
audio_file = 'G:/Shared drives/UF-AI-Catalyst/UF AI Code/test_data/1649855468-1649880078.csv'

loader = vid_dataloader(vid_folder,audio_file,vid_fps,duration,batch_size)

for data in loader:
    break