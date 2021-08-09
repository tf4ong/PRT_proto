import pandas as pd
import datetime as dt
import numpy as np 
import math
import cv2
from psyco_utils.track_utils import *
from collections import ChainMap

def get_track(tracks,tag):
    tag_track=[v for v in tracks if v[4]==tag]
    if tag_track != []:
        return tag_track[0]
    else:
        return []
def remove_pricdictions(x):
    first_idx=x.Tracks.first_valid_index()
    last_idx=x.Tracks.last_valid_index()
    x2=x.loc[first_idx:last_idx]
    return x2



def elist2nan(x):
    x.Tracks=x.Tracks.apply(lambda y:np.nan if y ==[] else y)
    return x


def location_compiler(tag,df,lim=5):
    tag_tracks=[get_track(track,tag) for track in df.RFID_tracks.values]
    tag_activity=[activity[tag] for activity in df.Activity]
    tag_xcoord=[bbox_to_centroid(v)[0] if v != [] else [] for v in tag_tracks]
    tag_ycoord=[bbox_to_centroid(v)[1] if v!= [] else [] for v in tag_tracks]
    columns=['Timestamp','Tracks','Centroid_X','Centroid_Y','Activity']
    df_tag=pd.DataFrame(list(zip(df.Time,tag_tracks,tag_xcoord,tag_ycoord,tag_activity))
                        ,columns=columns)
    df_tag.Centroid_X=df_tag.Centroid_X.apply(lambda y: np.nan if y==[] else y)
    df_tag.Centroid_Y=df_tag.Centroid_Y.apply(lambda y: np.nan if y==[] else y)
    df_tag['x1']=df_tag.Tracks.apply(lambda y: np.nan if y==[] else y[0])
    df_tag['y1']=df_tag.Tracks.apply(lambda y: np.nan if y==[] else y[1])
    df_tag['x2']=df_tag.Tracks.apply(lambda y: np.nan if y==[] else y[2])
    df_tag['y2']=df_tag.Tracks.apply(lambda y: np.nan if y==[] else y[3])
    df_tag.Centroid_X=round(df_tag.Centroid_X.interpolate(method ='linear',limit_direction ='both', limit = lim)) 
    df_tag.Centroid_Y=round(df_tag.Centroid_Y.interpolate(method ='linear',limit_direction ='both', limit = lim)) 
    df_tag.x1=round(df_tag.x1.interpolate(method ='linear',limit_direction ='both', limit = lim)) 
    df_tag.y1=round(df_tag.y1.interpolate(method ='linear',limit_direction ='both', limit = lim)) 
    df_tag.x2=round(df_tag.x2.interpolate(method ='linear',limit_direction ='both', limit = lim)) 
    df_tag.y2=round(df_tag.y2.interpolate(method ='linear',limit_direction ='both', limit = lim))
    df_tag['frame']=df.frame.values
    ds=np.split(df_tag,np.where(np.isnan(df_tag.Centroid_X))[0])
    ds=[t.drop(t.index[0]) for t in ds if len(t)>1]
    ds=[elist2nan(df) for df in ds]
    ds=[remove_pricdictions(df) for df in ds]
    return ds


#### maybe can increase accuracy afterwards


def extract_interpolation_dic(df):
    df_not_tracked=df.query('Activity == "Not_tracked"')
    data=zip (df_not_tracked.frame.values, df_not_tracked.x1.values,df_not_tracked.y1.values,
              df_not_tracked.x2.values,df_not_tracked.y2.values)
    dic_to_add={frame:[x1,y1,x2,y2] for frame, x1,y1,x2,y2 in data}
    return dic_to_add


def add_interpolation(bboxes,lost_tracks,RFID_tracks,tag,bbox,iou_thresh=0.25):
    r_track=bbox
    r_track.append(tag)
    if len(lost_tracks) ==0:
        tracks=RFID_tracks+[r_track]
        bboxes.append(r_track)
    else:
        ious=[iou(r_track,lost_track) for lost_track in lost_tracks]
        ious=[thres for thres in ious if thres>iou_thresh]
        if len(ious)>0:
            inde=ious.index(max(ious))
            r_track=lost_tracks[inde][:4]
            r_track.append(tag)
            tracks=RFID_tracks+[r_track]
        else:
            if len(lost_tracks)>1:
                tracks=RFID_tracks+[r_track]
                bboxes.append(r_track)
            else:
                ious2=[iou(r_track,lost_track) for lost_track in lost_tracks]
                ious2=[thres for thres in ious if thres>iou_thresh/2]
                if len(ious2)>0:
                    inde=ious2.index(max(ious))
                    r_track=lost_tracks[inde][:4]
                    r_track.append(tag)
                    tracks=RFID_tracks+[r_track]
                else:
                    tracks=RFID_tracks+[r_track]
                    bboxes.append(r_track)        
    return bboxes,tracks
        
        
def add_interpolation_df(frame,dic,bboxes,lost_tracks,RFID_tracks,tag,iou_thresh=0.1):
    #print(RFID_tracks)
    #print(type(RFID_tracks))
    if frame not in dic.keys():
        return bboxes, RFID_tracks
    else:
        bbox=dic[frame]
        bboxes, tracks=add_interpolation(bboxes,lost_tracks,RFID_tracks,tag,bbox,iou_thresh=0.1)
        return bboxes, tracks



