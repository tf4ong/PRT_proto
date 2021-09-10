from tqdm import tqdm
import time
import os
from PIL import Image
import pandas as pd
import numpy as np
import cv2
import core.utils as utils



def remove_overinter(x):
    first_idx=x.first_valid_index()
    last_idx=x.last_valid_index()
    x2=x.loc[first_idx:last_idx]
    return x2
def consecutive_motion_detection(path,gap_fill=2,frame_thres=15):
    df1=pd.read_csv(path+'/yolo_dets.csv')
    temp=[1 if i =='Motion' else np.nan for i in df1.motion.values]
    temp=pd.Series(temp)
    temp=temp.interpolate(method ='linear',limit_direction ='both', limit = gap_fill)
    temp=np.split(temp, np.where(np.isnan(temp))[0])
    temp=[t.drop(t.index[0]) for t in temp if len(t)>frame_thres]
    inds=[i.index.tolist() for i in temp]
    ind=[i for sublists in inds for i in sublists ]
    values=['Motion' if i in ind else 'No_motion' for i in range(len(df1))]
    df1['motion']=values
    df1.to_csv(path+'/yolo_dets.csv')
    return df1


def pure_motion_detect(folder,blur_filter_k_size,motion_area_thresh,
                  intensity_thres,motion_interpolation,len_motion_thres,write_vid=False):
#https://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/
#Gaussian Mixture Model-based foreground and background segmentation:
    videopath = folder+'/raw.avi'
    vid = cv2.VideoCapture(videopath)
    pbar = tqdm(total=int(vid.get(cv2.CAP_PROP_FRAME_COUNT)),position=0,leave=True)
    with open(folder+'/motion.csv','w') as file:
        file.write('frame,motion,motion_roi\n')
    if write_vid:
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(folder+'/yolov4_det.avi', codec, fps, (width, height))
    avg=None
    frame_count=0
    while vid.isOpened:
        return_value, frame = vid.read()
        if return_value:  
            avg, status,cnt_bb= utils.motion_detection(frame,avg,k_size=(blur_filter_k_size,blur_filter_k_size), 
                                                       min_area=motion_area_thresh,intensity_thres=intensity_thres)
            with open(folder+'/motion.csv','a') as file:
                file.write(f'{str(frame_count)},{status},"{cnt_bb}"\n')
            frame_count+=1
            pbar.update(1)
        else:
            vid.release()
            break
    
blur_filter_k_size=11
motion_area_thresh=100
intensity_thres=25
motion_interpolation=2
len_motion_thres=15

cages=['/media/tony/data/data/ann_marie/hong2','/media/tony/data/data/ann_marie/last_cage']

for cage in cages:
    vid_paths= [cage+'/'+i for i in os.listdir(cage) if i[-4:]!='.csv' and i[-4:]!='.ini' and i[-4:]!='.txt']
    for vid in vid_paths:
        print(vid)
        pure_motion_detect(vid,blur_filter_k_size,motion_area_thresh,intensity_thres,motion_interpolation,len_motion_thres,write_vid=False)
        cols={'bboxes':eval}
        cols_2={'motion_roi':eval}
        df_motion=pd.read_csv(vid+'/motion.csv',converters=cols_2)
        df_dets=pd.read_csv(vid+'/yolo_dets.csv',converters=cols)
        df_merged=pd.merge(df_dets,df_motion)
        df_merged.to_csv(vid+'/yolo_dets.csv')
        consecutive_motion_detection(vid,gap_fill=2,frame_thres=15)




















