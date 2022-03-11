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


from psyco import PSYCO
import numpy as np
import os
import pandas as pd
import time 


paths='/media/tony/data/data/of_cage_no/'
config_path='/media/tony/data/data/si_itc/config.ini'



#path1='/media/tony/cage5_2018_10/si_mice/'
#p_si=[path1+v for v in os.listdir(path1) if v[-4:]!='.ini']

#p_s=['/media/tony/cage5_2018_10/stroke/AC1','/media/tony/cage5_2018_10/stroke/AC2',
#    '/media/tony/cage5_2018_10/stroke/AC4']

#vids=[paths+folder for folder in os.listdir(paths) if folder[-4:]!='.txt' and folder[-4:] !='ions' and folder[-4:]!='.csv' and folder[-4:]!='.ini']
#vids=['/media/tony/data/data/of_cage_no/cage11non','/media/tony/data/data/of_cage_no/cage_mates',
#      '/media/tony/data/data/of_cage_no/cage_none_cage','/media/tony/data/data/of_cage_no/cage_none_cage2',
#      '/media/tony/data/data/of_cage_no/one_one_cm']
vids=['/media/tony/data/data/si_itc/2021-09-23_16-39-01_EN3']

#vids=[vids[1]]
errors=[]
errors2=[]
for path in vids:
    with open(path+'/'+'logs.txt','r') as f:
        tags=f.readlines()
        tags=[int(i) for i in tags[1][6:].split(',')]
    df2=pd.read_csv(f'{path}/RFID_data_all.csv',index_col=False)
    df2.Time=pd.to_datetime(df2['Time'],format="%Y-%m-%d_%H:%M:%S.%f")
    df2['Time']=df2['Time'].astype(np.int64)/10**9
    duration=df2.iloc[-1]['Time']-df2.iloc[0]['Time']
    test2=PSYCO(path,config_path)
    #test2.detect_mice()
    test2.load_RFID()
    test2.load_dets()
    _,_,cov=test2.RFID_match()
    test2.find_activte_mice()
    #test2.generate_validation_video()
    #cov=test2.trajectory_interpolation()
    #dets=sum([len(i) for i in test2.df_tracks_out.sort_tracks.values])
    #test2.compile_travel_trjectories()
    test2.generate_validation_video()
