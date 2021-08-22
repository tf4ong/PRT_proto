import numpy as np
from configparser import ConfigParser
import math
import cv2
from numba import jit
from decord import VideoReader
from decord import cpu, gpu
from itertools import islice



"""
simple functions 
"""
def get_tags(path):
    with open(path+'/'+'logs.txt','r') as f:
        tags=f.readlines()
        tags=[int(i) for i in tags[1][6:].split(',')]
    return tags

def get_ids(x):
    ids=[]
    if x!=[]:
        ids=[i[4] for i in x]
    return ids


def get_unique_ids(x):
    newlist=[]
    for i in x:
        if len(i)!=0:
            for y in i:
                newlist.append(y)
    else:
        pass
    newlist=list(set(newlist))
    return newlist

@jit
def bbox_area(bbox):
    '''
    calculates the area of the bbself.df_track_temp
    Intake bb:x1,y1,x2,y2
    ''' 
    w=abs(bbox[0]-bbox[2])
    h=abs(bbox[1]-bbox[3])
    return w*h

def float_int(x):
    rounded=[]
    for i in x:
        tracks=i[:4]
        tracks.append(int(i[4]))
        rounded.append(tracks)
    return rounded

def bbox_to_centroid(bbox):
    '''
    returns the centroid of the bbox
    '''
    if bbox!=[]:
        cX=(bbox[0]+bbox[2])/2
        cY=(bbox[1]+bbox[3])/2
        return [cX,cY]
    else:
        return []


def Distance(centroid1,centroid2):
    ''' 
    calculates the centronoid distances between bbs
    intake centronoid
    '''
    dist = math.sqrt((centroid2[0] - centroid1[0])**2 + (centroid2[1] - centroid1[1])**2)
    return dist

@jit
def iou(bb_test,bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
      + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
    return(o)

def sublist_decompose(list_of_list):
    return [bpt for bpts in list_of_list for bpt in bpts]


def bboxContains(bbox,pt,slack=5):
    logic = bbox[0]-slack<= pt[0] <= bbox[2]+slack and bbox[1]-slack<= pt[1] <= bbox[3]+slack
    return logic

def RFID_ious(RFID, bbox,RFID_coords):
    reader_coords=RFID_coords[int(RFID)]
    iou_reader=iou(reader_coords,bbox)
    return iou_reader

def reconnect_id_update(reconnect_ids,id2remove):
    for i in id2remove:
        del reconnect_ids[i]
    return reconnect_ids

def distance_box_RFID(RFID,bbox,RFID_coords):
    '''
    Gets the centroid distance between RFID reader of interest and bbox
    '''
    bbox_1_centroid=bbox_to_centroid(RFID_coords[int(RFID)])
    bbox_2_centroid=bbox_to_centroid(bbox)
    return Distance(bbox_1_centroid,bbox_2_centroid)

'''
Gets the distance of the bb to RFIDs
'''
def distance_to_entrance(bbox2,RFID_coords,entrance_reader):
    bbox_1_centroid=bbox_to_centroid(RFID_coords[entrance_reader])
    bbox_2_centroid=bbox_to_centroid(bbox2)
    return Distance(bbox_1_centroid,bbox_2_centroid)

def list_split(it,size):
    it = iter(it)
    return list(iter(lambda: tuple(islice(it, size)), ()))

def apply_slack(listbb,slack):
    for bbi in range(len(listbb)):
        listbb[bbi]=[listbb[bbi][0]+slack,listbb[bbi][1]+slack,listbb[bbi][2]+slack,listbb[bbi][3]+slack,listbb[bbi][4]]
    return listbb





def bb_contain_mice_check(frame,bbox,diff_bg):
    xstart,ystart,xend,yend= int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
    cropped_bbox=frame[ystart:yend,xstart:xend]
    try:
        if np.absolute(np.mean(frame)-np.mean(cropped_bbox))> diff_bg:
            return True
        else:
            return False
    except Exception:
        return True


def detect_config_loader(path):
    config = ConfigParser()
    config.read(path)
    cfg = 'NMT_Detection'
    config_dic={}
    config_dic['weightpath']=str(config.get(cfg, 'weightpath')).split(',')
    config_dic['score']=float(config.get(cfg, 'score'))
    config_dic['iou']=float(config.get(cfg, 'iou'))
    config_dic['size']=int(config.get(cfg, 'size'))
    config_dic['motion_interpolation']=int(config.get(cfg, 'motion_interpolation'))
    config_dic['motion_area_thresh']=int(config.get(cfg, 'motion_area_thresh'))
    config_dic['len_motion_thres']=int(config.get(cfg, 'len_motion_thres'))
    config_dic['blur_filter_k_size']=int(config.get(cfg, 'blur_filter_k_size'))
    config_dic['intensity_thres']=int(config.get(cfg, 'intensity_thres'))
    return config_dic
    


def tracking_config_loader(path):
    config = ConfigParser()
    config.read(path)
    cfg = 'Tracking'
    config_dic={}
    config_dic['max_age']=int(config.get(cfg, 'max_age'))
    config_dic['min_hits']=int(config.get(cfg, 'max_age'))
    config_dic['iou_threshold']= float(config.get(cfg, 'iou_threshold'))
    config_dic['interaction_thres']= float(config.get(cfg, 'interaction_thres'))
    config_dic['iou_min_sbb_checker']= float(config.get(cfg, 'iou_min_sbb_checker'))
    config_dic['sbb_frame_thres']= int(config.get(cfg, 'sbb_frame_thres'))
    config_dic['leap_distance']= int(config.get(cfg, 'sbb_frame_thres'))
    config_dic['resolution']=list(map(int, config.get(cfg, 'resolution').split(',')))
    return config_dic
    
def analysis_config_loader(path):
    config = ConfigParser()
    config.read(path)
    cfg = 'NMT_RFID_Matching'
    config_dic={}
    config_dic['RFID_readers']=eval(str(config.get(cfg, 'RFID_readers')))
    config_dic['entrance_time_thres']=float(config.get(cfg, 'entrance_time_thresh'))
    config_dic['entrance_distance']=float(config.get(cfg, 'entrance_distance'))
    config_dic['correct_iou']= float(config.get(cfg, 'correct_iou'))
    config_dic['RFID_dist']= float(config.get(cfg, 'RFID_dist'))
    config_dic['entr_frames']=int(config.get(cfg, 'entr_frames'))
    config_dic['reader_thres']= float(config.get(cfg, 'reader_thres'))
    config_dic['trac_interpolation']=int(config.get(cfg, 'trac_interpolation'))
    entrance_readers= str(config.get(cfg, 'entrance_reader')).split(',')
    entrance_readers[0]=eval(entrance_readers[0])
    entrance_readers[1]=int(entrance_readers[1])
    config_dic['entrance_reader']=entrance_readers
    if config_dic['entrance_reader'][0]: 
        config_dic['entrance_reader']=config_dic['entrance_reader'][1]
    else: 
        config_dic['entrance_reader']=None
    return config_dic

def dlc_config_loader(path):
    config = ConfigParser()
    config.read(path)
    cfg = 'DLC'
    config_dic={}
    config_dic['dbpt']=str(config.get(cfg, 'dbpts')).split(',')
    config_dic['dbpt_distance_compute']=str(config.get(cfg, 'dbpt_distance_compute')).split(',')
    config_dic['dbpt_int']=str(config.get(cfg, 'dbpt_int')).split(',')
    config_dic['dbpt_box_slack']=int(config.get(cfg, 'dbpt_box_slack'))
    return config_dic