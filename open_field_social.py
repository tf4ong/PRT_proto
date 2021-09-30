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
