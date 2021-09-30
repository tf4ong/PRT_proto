from psyco import PSYCO
import numpy as np
import os
import pandas as pd
import time 



path_dic='/media/tony/data/data/ann_marie/'
cages=[path_dic+folder for folder in os.listdir(path_dic) if folder[-4:]!='.txt' and folder[-4:] !='ions' and folder[-4:]!='.csv' and folder[-4:]!='.ini']
coverage=[]
detections=[]
n_animals=[]
durations=[]
timespent=[]
#paths=[paths[3]]
cages=['/media/tony/data/data/ann_marie/273107','/media/tony/data/data/ann_marie/273115',
       '/media/tony/data/data/ann_marie/273115','/media/tony/data/data/ann_marie/hong1','/media/tony/data/data/ann_marie/mx1']
#['/media/tony/data/data/ann_marie/mx1','/media/tony/data/data/ann_marie/fx2','/media/tony/data/data/ann_marie/hong1',
#       '/media/tony/data/data/ann_marie/273107','/media/tony/data/data/ann_marie/273115','/media/tony/data/data/ann_marie/fx3']
for cage in cages:
    vid_paths= [cage+'/'+i for i in os.listdir(cage) if i[-4:]!='.csv' and i[-4:]!='.ini' and i[-4:]!='.txt']
    #coverage=[]
    #detections=[]
    #durations=[]
    config_path=cage+'/config.ini'
    for vid in vid_paths:
        print('')
        print(f'Processing {vid}')
        test2=PSYCO(vid,config_path)
        #df2=pd.read_csv(f'{vid}/RFID_data_all.csv',index_col=False)
        #df2.Time=pd.to_datetime(df2['Time'],format="%Y-%m-%d_%H:%M:%S.%f")
        #df2['Time']=df2['Time'].astype(np.int64)/10**9
        #duration=df2.iloc[-1]['Time']-df2.iloc[0]['Time']
        #durations.append(duration)
        #t1=time.time()
        #test2.load_RFID()
        #test2.load_dets()
        #_,_,cov=test2.RFID_match()
        #t2=time.time()
        test2.find_activte_mice()
        test2.load_dlc_bpts()
        dets=sum([len(i) for i in test2.df_tracks_out.sort_tracks.values])
        #print(dets)
        #print(cov)
        #test2.generate_validation_video()
        #timespent.append(t2-t1)
        #coverage.append(cov)
        #detections.append(dets)
        test2.compile_travel_trjectories(dlc=True)
    #files=[os.path.split(i)[1] for i in vid_paths]
    #df_coverage=pd.DataFrame(columns=['file','duration','Analysis_time','n_detections','coverage'],data=list(zip(files,durations,timespent,detections,coverage)))
    #df_coverage.to_csv(cage+'/coverage_new.csv')
print('finished!')


