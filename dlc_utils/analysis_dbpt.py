import os
import pandas as pd
import matplotlib.pylab as plt
import match_temp2 as mm
from track_utils import *
import itertools
import tqdm
from get_mouse_status import *
import multiprocessing 
import numpy as np
from datetime import datetime,timedelta
def function_combine(i):
    dics={'RFID_tracks':eval,'motion_roi':eval}
    df_rfid=pd.read_csv(i+'/RFID_tracks.csv',converters=dics)
    df_motion= pd.read_csv(i+'/motions.csv')
    df_tracks_out=pd.merge(df_rfid,df_motion,on='frame')
    with open(i+'/'+'logs.txt','r') as f:
        tags=f.readlines()
        tags=[int(i) for i in tags[1][6:].split(',')]
    columns=['frame']+ c
    dics={y: eval for y in columns}
    df_bpts=pd.read_csv(i+'/'+'dlc_bpts.csv',converters=dics)
    df_dbpt_columns=[f'df_bpts["{i}"]' for i in c]
    df_bpts['bpts']=eval('+'.join(df_dbpt_columns))
    df_bpts['frame']=range(len(df_bpts))
    df_bpts=df_bpts.drop(columns=c)
    columns=['bboxes']
    df_tracks_out=pd.merge(df_tracks_out,df_bpts, on='frame')
    dbpts=[mm.rfid2bpts(bpts,RFIDs,0,bpt2look=['head_center','mid_body','tail_base'])
           for bpts,RFIDs in zip(df_tracks_out['bpts'].values,df_tracks_out['RFID_tracks'].values)]
    df_tracks_out['dbpt2look']=[i[0] for i in dbpts]
    df_tracks_out['undetemined_bpt']=[i[1] for i in dbpts]
    list_bpts=list(map(sublist_decompose,df_tracks_out.dbpt2look.values.tolist()))
    for z in tags: 
        exec(f'list_bpt_{str(z)}=[]')
        for y in list_bpts:
            bpts=[v for v in y if v[3]==z]
            exec(f'list_bpt_{str(z)}.append(bpts)')
        df_tracks_out[f'{z}_bpts']=eval(f'list_bpt_{str(z)}')
    rows=df_tracks_out.apply(lambda x:mm.bpt_distance_compute(x,tags,['head_center']),axis=1)
    new_cols=[str(list(i)[0]) + '_'+str(list(i)[1]) for i in itertools.permutations(tags, 2)]
    for name,idx in zip(new_cols,range(len(new_cols))):
        df_tracks_out[name]=[dists[idx] for dists in rows]
    #dics={'RFID_tracks':eval,'motion_roi':eval}
    #df_tracks_out.to_csv(i+'/RFID_tracks_c.csv')
    #df_tracks_out=pd.read_csv(i+'/RFID_tracks_c.csv',converters=dics)
    #df_tracks_out['Activity']=[mm.get_tracked_activity(motion_status,motion_roi,RFID_tracks,tags) for motion_status,
    #                                motion_roi,RFID_tracks in zip(df_tracks_out['motion'].values,
    #                                                              df_tracks_out['motion_roi'].values,
    #                                                              df_tracks_out['RFID_tracks'].values)] 
    df_tracks_out.to_csv(i+'/RFID_tracks_c.csv')



path='/media/tony/TM_backup/temp/273107'
folders=[path+'/'+i for i in os.listdir(path) if i[-4:] !='.csv' and i[-4:]!='.txt']
c=['snout','right_ear','left_ear','head_center','neck','mid_body','lower_midbody','tail_base']


processes = []
for fold in folders: 
    process = multiprocessing.Process(target=function_combine,args=(fold,))
    processes.append(process)
    process.start()
for proc in processes:
    proc.join()
    
def get_activity(folder):
    with open(folder+'/'+'logs.txt','r') as f:
        tags=f.readlines()
        tags=[int(i) for i in tags[1][6:].split(',')]
    dics={'RFID_tracks':eval,'motion_roi':eval}
    df_tracks_out=pd.read_csv(folder+'/RFID_tracks_c.csv',converters=dics)
    df_tracks_out['Activity']=[mm.get_tracked_activity(motion_status,motion_roi,RFID_tracks,tags) for motion_status,
                                    motion_roi,RFID_tracks in zip(df_tracks_out['motion'].values,
                                                                  df_tracks_out['motion_roi'].values,
                                                                  df_tracks_out['RFID_tracks'].values)]
    df_tracks_out.to_csv(folder+'/RFID_tracks_c.csv')
    
processes = []
for fold in folders: 
    process = multiprocessing.Process(target=get_activity,args=(fold,))
    processes.append(process)
    process.start()
for proc in processes:
    proc.join()

'''
combines motion, dpbts, rfid_tracks
'''
"""
for i in folders: 
    dics={'RFID_tracks':eval}
    df_rfid=pd.read_csv(i+'/RFID_tracks.csv',converters=dics)
    df_motion= pd.read_csv(i+'/motions.csv')
    df_tracks_out=pd.merge(df_rfid,df_motion,on='frame')
    with open(i+'/'+'logs.txt','r') as f:
        tags=f.readlines()
        tags=[int(i) for i in tags[1][6:].split(',')]
    columns=['frame']+ c
    dics={y: eval for y in columns}
    df_bpts=pd.read_csv(i+'/'+'dlc_bpts.csv',converters=dics)
    df_dbpt_columns=[f'df_bpts["{i}"]' for i in c]
    df_bpts['bpts']=eval('+'.join(df_dbpt_columns))
    df_bpts['frame']=range(len(df_bpts))
    df_bpts=df_bpts.drop(columns=c)
    columns=['bboxes']
    df_tracks_out=pd.merge(df_tracks_out,df_bpts, on='frame')
    dbpts=[mm.rfid2bpts(bpts,RFIDs,0,bpt2look=['head_center','mid_body','tail_base'])
           for bpts,RFIDs in zip(df_tracks_out['bpts'].values,df_tracks_out['RFID_tracks'].values)]
    df_tracks_out['dbpt2look']=[i[0] for i in dbpts]
    df_tracks_out['undetemined_bpt']=[i[1] for i in dbpts]
    list_bpts=list(map(sublist_decompose,df_tracks_out.dbpt2look.values.tolist()))
    for z in tags: 
        exec(f'list_bpt_{str(z)}=[]')
        for y in list_bpts:
            bpts=[v for v in y if v[3]==z]
            exec(f'list_bpt_{str(z)}.append(bpts)')
        df_tracks_out[f'{z}_bpts']=eval(f'list_bpt_{str(z)}')
    rows=df_tracks_out.apply(lambda x:mm.bpt_distance_compute(x,tags,['head_center']),axis=1)
    new_cols=[str(list(i)[0]) + '_'+str(list(i)[1]) for i in itertools.permutations(tags, 2)]
    for name,idx in zip(new_cols,range(len(new_cols))):
        df_tracks_out[name]=[dists[idx] for dists in rows]
    df_tracks_out.to_csv(i+'/RFID_tracks_c.csv')
    
'''
finds active mice
'''
    
for i in folders:
    with open(i+'/'+'logs.txt','r') as f:
        tags=f.readlines()
        tags=[int(i) for i in tags[1][6:].split(',')]
    dics={'RFID_tracks':eval,'motion_roi':eval}
    df_tracks_out=pd.read_csv(i+'/RFID_tracks_c.csv',converters=dics)
    df_tracks_out['Activity']=[mm.get_tracked_activity(motion_status,motion_roi,RFID_tracks,tags) for motion_status,
                                    motion_roi,RFID_tracks in zip(df_tracks_out['motion'].values,
                                                                  df_tracks_out['motion_roi'].values,
                                                                  df_tracks_out['RFID_tracks'].values)] 
    df_tracks_out.to_csv(i+'/RFID_tracks_c.csv')
    

"""





"""
mous_status active read
"""
def status_int(folder):
     with open(folder+'/'+'logs.txt','r') as f:
        tags=f.readlines()
        tags=[int(y) for y in tags[1][6:].split(',')]
     pathin=folder+'/RFID_tracks_c.csv'
     #print(folder)
     df1=read_RFID_data(pathin,tags,3)
     #print(df1.columns)
     cdrop1=[f'{z}_bpts' for z in tags]
     df1=df1.drop(columns=['motion', 'motion_roi',
        'bpts', 'dbpt2look', 'undetemined_bpt']+cdrop1)
     file_name= os.path.dirname(pathin)+'/mouse_status.csv'
     df1.to_csv(file_name)





#pbar=tqdm.tqdm(total=len(folders),position=0, leave=True)

    
processes = []
for fi in folders: 
    process = multiprocessing.Process(target=status_int,args=(fi,))
    processes.append(process)
    process.start()
for proc in processes:
    proc.join() 
    
    
"""
     with open(i+'/'+'logs.txt','r') as f:
        tags=f.readlines()
        tags=[int(i) for i in tags[1][6:].split(',')]
     pathin=folder+'/RFID_tracks_c.csv'
     df1=read_RFID_data(pathin,tags,3)
     cdrop1=[f'{i}_bpts' for i in tags]
     df1=df1.drop(columns=['Unnamed: 0.1', 'Unnamed: 0_x','Unnamed: 0_y', 'motion', 'motion_roi',
       'Unnamed: 0.1.1', 'bpts', 'dbpt2look', 'undetemined_bpt','Unnamed: 0.1.1.1']+cdrop1)
     file_name= os.path.dirname(pathin)+'/mouse_status.csv'
     df1.to_csv(file_name)
     pbar.update(1)
"""
def gma(tag,dic):
    if dic[tag] == 'Active':
        return 1
    else:
        return 0
def gmna(tag,dic):
    if dic[tag] == 'Active':
        return 0
    else:
        return 1

#df1[df1['2016050945'].map(len)==1].index

corners=[[53, 48, 138, 127],[43, 195, 155, 293],[348, 36, 458, 123]]
corner_cent=[bbox_to_centroid(i) for i in corners]
corner_thres=50


def adjust_read(f2):
    with open(f2+'/'+'logs.txt','r') as f:
        tags=f.readlines()
        tags=[int(z) for z in tags[1][6:].split(',')]
    df1=read_mouse_interactions(f2,'',tags,corner_thres,corner_cent)
    for t in tags:
        incorner_values=df1[f'{t}corner_dur'].values
        outcorner_values=df1[f'{t}ocorner_dur'].values
        activity_adjusta=np.asarray([gma(t,dic) for dic in df1['Activity'].values])
        activity_adjustna=np.asarray([gmna(t,dic) for dic in df1['Activity'].values])
        values=activity_adjusta.tolist()[:-1]*df1['Duration'].values[1:]
        df1[f'{t}_active']=np.insert(values,len(values),np.nan)
        df1[f'{t}corner_dur']=incorner_values*activity_adjustna
        df1[f'{t}ocorner_dur']=outcorner_values*activity_adjustna   
    file_name= f2+'/mouse_status.csv'
    df1.to_csv(file_name)

processes = []
for fold in folders: 
    process = multiprocessing.Process(target=adjust_read,args=(fold,))
    processes.append(process)
    process.start()
for proc in processes:
    proc.join()   
    
#df1=pd.read_csv(i+'/mouse_status.csv')



    

from multiprocessing import Pool

def get_allms(fold):
    with open(fold+'/'+'logs.txt','r') as f:
        tags=f.readlines()
        tags=[int(y) for y in tags[1][6:].split(',')]
    c_tag=[str(list(i)[0]) + '_'+str(list(i)[1]) for i in itertools.permutations(tags, 2)]
    c_a=[]
    for t in tags: 
        c_a.append(f'{t}_active')
        c_a.append(f'{t}corner_dur')
        c_a.append(f'{t}ocorner_dur')
        c_a.append(f'{t}_active')
        c_a.append(f'{t}in_cage_dur')
    columns=['Time','Tracked','Activity']+c_tag+[str(z) for z in tags]
    dics={y:eval for y in columns}
    df1=pd.read_csv(fold+'/mouse_status.csv',converters=dics)
    df1=df1[columns+['frame']+['Duration']+c_a]
    return df1



#filenames=[y+'/mous_status.csv' for y in folders]
f=Pool(processes=12)
results = f.map(get_allms, folders)
df_final=pd.concat(results)
df_final=df_final.sort_values(by=['Time'])





df_final = df_final.loc[:,~df_final.columns.duplicated()]

tags=[2016050811,2016050855,2016080244]

for tag in tags:
    name=path+'/'+str(tag)+'dist'
    os.mkdir(name)
    idxes=df_final[(df_final[str(tag)].map(len)!=0) & (df_final[f'{str(tag)}_active'] !=0)].index
    head_dist=[]
    body_dist=[]
    tail_dist=[]
    for ind in idxes:
        c2look=[str(tag)+'_'+str(t) for t in df_final.iloc[ind][str(tag)]] 
        for c in c2look:
            if c[-2:]=='UK':
                pass
            else:
                dist_dic=df_final.iloc[ind][c]
                if len(dist_dic) !=0:
                    for bpt,dist in dist_dic.items():
                        if bpt[12:] == 'head_center':
                            head_dist.append(dist[4])
                        elif bpt[12:] == 'mid_body':
                            body_dist.append(dist[4])
                        elif bpt[12:] == 'tail_base':
                            tail_dist.append(dist[4])
                        else:
                            pass
    df_head = pd.DataFrame(data={"head_dist": head_dist})
    df_mb = pd.DataFrame(data={"mb_dist": body_dist})
    df_tail = pd.DataFrame(data={"tail_dist": tail_dist})
    df_head.to_csv(name+'/a_head.csv')
    df_mb.to_csv(name+'/a_mb.csv')
    df_tail .to_csv(name+'/a_tail.csv')
    
for tag in tags:
    name=path+'/'+str(tag)+'dist'
    #os.mkdir(name)
    idxes=df_final[(df_final[str(tag)].map(len)!=0) & (df_final[f'{str(tag)}_active'] ==0)].index
    head_dist=[]
    body_dist=[]
    tail_dist=[]
    for ind in idxes:
        c2look=[str(tag)+'_'+str(t) for t in df_final.iloc[ind][str(tag)]] 
        for c in c2look:
            if c[-2:]=='UK':
                pass
            else:
                dist_dic=df_final.iloc[ind][c]
                if len(dist_dic) !=0:
                    for bpt,dist in dist_dic.items():
                        if bpt[12:] == 'head_center':
                            head_dist.append(dist[4])
                        elif bpt[12:] == 'mid_body':
                            body_dist.append(dist[4])
                        elif bpt[12:] == 'tail_base':
                            tail_dist.append(dist[4])
                        else:
                            pass
    df_head = pd.DataFrame(data={"head_dist": head_dist})
    df_mb = pd.DataFrame(data={"mb_dist": body_dist})
    df_tail = pd.DataFrame(data={"tail_dist": tail_dist})
    df_head.to_csv(name+'/ia_head.csv')
    df_mb.to_csv(name+'/ia_mb.csv')
    df_tail .to_csv(name+'/ia_tail.csv')   

df_final.to_csv(path+'/status.csv')

df_list_sum=[]
for tag in tags:
    a=df_final[str(tag)+'corner_dur'].to_list()
    e=df_final[str(tag)+'ocorner_dur'].to_list()
    b=df_final[str(tag)+'in_cage_dur'].to_list()
    c=df_final[str(tag)+'_active'].to_list()
    d=df_final['Time']
    dic={'Time':d,'corner':a,'out_corner':e,'cage_dur':b,'active':c}
    df=pd.DataFrame.from_dict(dic)
    df['Time'] =[datetime.utcfromtimestamp(t) for t in df['Time']] 
    df=df.set_index('Time')
    df=df.groupby(pd.Grouper(freq='1H')).sum()
    df['Mouse']=tag
    df_list_sum.append(df)
df_sum=pd.concat(df_list_sum)
df_sum.to_csv(path+'/status_sum.csv')

