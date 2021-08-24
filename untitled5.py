from psyco import PSYCO
import numpy as np
import os
import pandas as pd
import time 

path_dic='/media/tony/data/data/Pre_stroke/'
paths=[path_dic+folder for folder in os.listdir(path_dic) if folder[-4:]!='.txt' and folder[-4:] !='ions' and folder[-4:]!='.csv' and folder[-4:]!='.ini']
config_path='/home/tony/alt_tracker/tests/config.ini'
paths=['/media/tony/data/data/Pre_stroke/D1']
for path in paths:
    test2=PSYCO(path,config_path)
    #test2.detect_mice()
    test2.load_RFID()
    test2.load_dets()
    _,_,cov=test2.RFID_match()
    test2.find_activte_mice()
    test2.compile_travel_trjectories()
    
    
    
    
    
import traja
import matplotlib,pylab as plt
import seaborn as sns

df1=pd.read_csv('/media/tony/data/data/Pre_stroke/p3/trajec_analysis.csv')
df2=pd.read_csv('/media/tony/data/data/Pre_stroke/P3/trajec_analysis.csv')



for p in paths:
    subpath=p+'/trajectories/'
    tra_path=[subpath+i for i in os.listdir(subpath)][0]
    trajecs=[pd.read_csv(tra_path+'/'+i,index_col=False) for i in os.listdir(tra_path)]
    if len(trajecs) != 1:
        print('error')
        print(subpath)
    trajecs=trajecs[0]
    trajecs=trajecs.rename(columns={'Centroid_X':'x','Centroid_Y':'y'})
    trajecs=trajecs.drop(columns=['Unnamed: 0'])
    trajecs.traja.calc_turn_angle()
    df_speed=trajecs.traja.get_derivatives()
    df_speed['frame']=[i+1 for i in range(len(df_speed))] 
    trajecs=pd.merge(trajecs,df_speed,on='frame')
    trajecs.to_csv(p+'/trajec_analysis.csv',index=False)
    
    
    

    
df1=pd.read_csv('/media/tony/data/data/Pre_stroke/D1/trajectories/1/track_0.csv')
df1=pd.read_csv('/media/tony/data/data/Pre_stroke/D1/trajectories/1/track_1.csv')





