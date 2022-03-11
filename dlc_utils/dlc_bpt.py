from absl import app, flags, logging
from absl.flags import FLAGS
import deeplabcut as dlc 
import pickle, re
from deeplabcut.pose_estimation_tensorflow.lib.inferenceutils import (
    convertdetectiondict2listoflist,
)
import cv2
import os 
import pandas as pd 
#import modin.pandas as pd
import time 
from tqdm import tqdm
import shutil
flags.DEFINE_string('config_path', '/home/tony/multi_dlc_homecage-TF-2020-10-05/config.yaml',
                    'path to DLC config file')
flags.DEFINE_string('data','./data','path to video and RFID csv file for tracker cage')
flags.DEFINE_float('pcut_off', 0.85, 'DLC bpts detection cutoff threshold')


#pcut_off_apply=FLAGS.pcut_off
def read_fix(path):
    vid=cv2.VideoCapture(path+'/raw.avi')
    fps = 15
    codec = cv2.VideoWriter_fourcc(*'XVID')
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))   
    os.mkdir(path+'/frames')
    print('reading frames to write in new video')
    frame_count=1
    while vid.isOpened():
        ret,frame=vid.read()
        if ret:
            cv2.imwrite(path+f'/frames/frame_{frame_count}.jpg',frame)
            frame_count+=1
        else:
            break
    frame_list=os.listdir(f'{path}/frames')
    frame_list.sort(key=lambda x: int(x[:-4].split('_')[1]))
    print('writing in to new video')
    out = cv2.VideoWriter(path+'/raw.avi', codec, fps, (width, height))
    pbar=tqdm(total=len(frame_list))
    for i in frame_list:
        frame=cv2.imread(path+'/frames/'+i)
        out.write(frame)
        pbar.update(1)
    out.release()
    shutil.rmtree(path+'/frames')

def list_labels(list_coords,str):
    if list_coords ==[]:
        return []
    else:
        new_list=[]
        for i in list_coords:
            bpt=list(i[0:2])
            bpt.append(str)
            new_list.append(bpt)
        return new_list
# fix join name to include frame 

def video_splitter(video,frames=20000):
    print(video+'/raw.avi')
    vid=cv2.VideoCapture(video+'/raw.avi')
    fps=int(vid.get(cv2.CAP_PROP_FPS))
    codec=cv2.VideoWriter_fourcc(*'XVID')
    width=int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fc=int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    print(fc)
    if fc ==0:
        read_fix(video)
        vid=cv2.VideoCapture(video+'/raw.avi')
        fps=int(vid.get(cv2.CAP_PROP_FPS))
        codec=cv2.VideoWriter_fourcc(*'XVID')
        width=int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fc=int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    divides=fc/frames
    if divides>int(divides):
        divides=int(divides+1)
    else:
        divides=int(divides)
    vid.release()
    frame_count_start=0
    if not os.path.exists(video+'/subvideos'):
        print(video+'/subvideos')
        os.mkdir(video+'/subvideos')
    pbar = tqdm(total=divides)
    for i in range(divides):
        frame_count=0
        vid = cv2.VideoCapture(video+'/raw.avi')
        filename=video+'/subvideos/'+f'raw_{str(i)}.avi'
        out= cv2.VideoWriter(filename,codec,fps,(width, height))
        vid.set(cv2.CAP_PROP_POS_FRAMES,frame_count_start)
        while vid.isOpened():
            ret,img=vid .read()
            if ret:
                frame_count+=1
                #print(f"{str(frame_count)}")
                if frame_count<=frames:
                    out.write(img)
                else:
                    break
            else:
                break
        pbar.update(1)
        frame_count_start+=frames



def dlc_pic2csv(path,pcut=0.85):
    with open(path,'rb') as file:
        data=pickle.load(file)
    header=data.pop("metadata")
    frame_names = list(data)
    all_jointnames = header["all_joints_names"]
    numjoints = len(all_jointnames)
    bpts = range(numjoints)
    frames = [int(re.findall(r"\d+", name)[0]) for name in frame_names]
    df1=pd.DataFrame(columns=all_jointnames)
    list_parts_sum=[]
    for a in frames:
        dets=convertdetectiondict2listoflist(data[frame_names[a]], bpts)
        list_parts=[[] for i in all_jointnames]
        for i in bpts:
            for z in dets[i]:
                if z!=[]:
                    if z[2]>pcut:
                        list_parts[i].append(z)
                else:
                    list_parts[i].append([])
        #df1.loc[a]=list_parts
        list_parts_sum.append(list_parts)
    df1=pd.DataFrame(columns=all_jointnames,data=list_parts_sum)
    for i in df1.columns:
        df1[i]=df1[i].apply(lambda x: list_labels(x,i))
    df1.to_csv(os.path.split(path)[0]+os.path.split(path)[1][-7]+'.csv')
    return df1
def main(_argv):
   #vids=[i[-4:] for i in os.listdir(FLAGS.data)]
    #print(vids)
    vid_path=[FLAGS.data+vid_folders for vid_folders in os.listdir(FLAGS.data) if vid_folders[-4:]!='.csv' and vid_folders[-4:]!='.txt']
    print(vid_path)
    for videos in vid_path:
        if os.path.exists(videos+'/dlc_bpts.csv'):
            print('already done')
            pass
        else:
            print(videos)
            video_splitter(videos)
            #saves the file as pickle file
            subvid_path=[videos+f'/subvideos/{i}' for i in os.listdir(videos+'/subvideos/')]
            dlc.analyze_videos(FLAGS.config_path,subvid_path, videotype='.avi')
            pic_path=[videos+f'/subvideos/{i}' for i in os.listdir(videos+'/subvideos/') if i[-11:]=='full.pickle']
            print(pic_path)
            for pic in pic_path:
                df=dlc_pic2csv(pic)
                path_save=os.path.split(pic)[0]+'/'+os.path.split(pic)[1][:-7]+'.csv'
                df.to_csv(path_save)
            csv_path=[videos+f'/subvideos/{i}' for i in os.listdir(videos+'/subvideos/') if i[-4:]=='.csv']
            re_in=[os.path.basename(i) for i in csv_path]
            re_in=[i.split('DLC',1)[0][4:] for i in re_in]
            re_in=[int(i) for i in re_in]
            csv_path_ordered=[x for _,x in sorted(zip(re_in,csv_path))]
            df_list=[pd.read_csv(i) for i in csv_path_ordered]
            df_bpts=pd.concat(df_list)
            df_bpts['frame']=[i for i in range(len(df_bpts))]
            df_bpts=df_bpts.drop(columns=['Unnamed: 0'])
            df_bpts.to_csv(videos+'/dlc_bpts.csv')
            shutil.rmtree(videos+'/subvideos')
    '''
    with open(pic_path,'rb') as file:
        data=pickle.load(file)
    header=data.pop("metadata")
    all_jointnames = header["all_joints_names"]
    numjoints = len(all_jointnames)
    bpts = range(numjoints)
    frame_names = list(data)
    frames = [int(re.findall(r"\d+", name)[0]) for name in frame_names]
    t1=time.time()
    #df1=pd.DataFrame(columns=all_jointnames)
    list_parts_sum=[]
    for a in frames:
        dets=convertdetectiondict2listoflist(data[frame_names[a]], bpts)
        list_parts=[[] for i in all_jointnames]
        for i in bpts:
            for z in dets[i]:
                if z!=[]:
                    if z[2]>FLAGS.pcut_off:
                        list_parts[i].append(z)
                else:
                    list_parts[i].append([])
        #df1.loc[a]=list_parts
        list_parts_sum.append(list_parts)
    df1=pd.DataFrame(columns=all_jointnames,data=list_parts_sum)
    for i in df1.columns:
        df1[i]=df1[i].apply(lambda x: list_labels(x,i))
    df1.to_csv(FLAGS.data+'/dlc_bpts.csv')
    t2=time.time()
    print(t2-t1)
    '''
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
    

    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
