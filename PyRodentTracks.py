import pandas as pd 
from prt_utils import mouse_matching as mm
import os
import numpy as np
import prt_utils.trajectory_analysis_utils as ta
import prt_utils.configloader as config_loader
import sys
from prt_utils.track_utils import *
import itertools
from prt_utils.generate_vid import generate_RFID_video,create_validation_Video
from prt_utils.detect_utils import yolov4_detect
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
"""
Open arena (i.e. no entrance/exit arena) speed not optomized
may need adiitionl methods.
"""
class PRT:
    def __init__(self,path,config_path):
        self.path=path 
        with open(path+'/'+'logs.txt','r') as f:
            tags=f.readlines()
            self.tags=[int(i) for i in tags[1][6:].split(',')]
        self.config_path=config_path
        self.config_dic_analysis=config_loader.analysis_config_loader(config_path)
        self.config_dic_detect=config_loader.detect_config_loader(config_path)
        self.config_dic_tracking=config_loader.tracking_config_loader(config_path)
        self.config_dic_dlc=config_loader.dlc_config_loader(config_path)
        if os.path.exists(self.path+'/RFID_tracks.csv'):
            print('Already analyzed. Found RFID_tracks.csv file in path')
            self.load_data()
            #reload=input('Load analyzed data? Y/N')
            #if reload.lower() =='y':
            #    self.load_data()
            #else:
            #    pass
        return
    def load_data(self):
        convert_dict={'sort_tracks':eval,'RFID_tracks':eval,'ious_interaction':eval,
                      'Interactions':eval,'motion_roi':eval,'RFID_matched':eval,
                      'Activity':eval}
        try:
            self.df_tracks_out=pd.read_csv(self.path+'/RFID_tracks.csv',converters=convert_dict)
            print(f'Loaded RFID_tracks.csv at path {self.path}')
        except Exception as e:
            print(e)
            print('confimr that mice detections are in folder')
        return
    def detect_mice(self,write_vid=False):
        yolov4_detect(self.path,self.config_dic_detect,write_vid)
        return
    def load_RFID(self):
        try:
            self.df_RFID=mm.RFID_readout(self.path,self.config_dic_analysis,len(self.tags))
        except Exception as e:
            print(e)
            print('Confirm correct folder path')
            sys.exit(0)
        if len(self.tags) !=1:
            n_RFID_readings=len(self.df_RFID[self.df_RFID['RFID_readings'].notnull()])
        else:
            n_RFID_readings=0
        duration=self.df_RFID.iloc[-1]['Time']-self.df_RFID.iloc[0]['Time']
        print(f'{n_RFID_readings} Tags were read in {duration} seconds')
    def load_dets(self):
        self.df_tracks=mm.read_yolotracks(self.path,self.config_dic_analysis,self.config_dic_tracking,
                                          self.df_RFID,len(self.tags))
        if self.config_dic_analysis['entrance_reader'] is None:
            self.df_tracks=mm.reconnect_tracks_ofa(self.df_tracks,len(self.tags))
            pass
        else:
            reconnect_ids=mm.get_reconnects(self.df_tracks)
            self.df_tracks,id2remove=mm.spontanuous_bb_remover(reconnect_ids,self.df_tracks,
                                                               self.config_dic_analysis,self.config_dic_tracking)       
            reconnect_ids=mm.reconnect_id_update(reconnect_ids,id2remove)
            trac_dict_leap=mm.reconnect_leap(reconnect_ids,self.df_tracks,self.config_dic_tracking['leap_distance'])
            self.df_tracks=mm.replace_track_leap_df(trac_dict_leap,self.df_tracks,self.config_dic_analysis)
        self.df_tracks=mm.Combine_RFIDreads(self.df_tracks,self.df_RFID)
        return self.df_tracks
    def RFID_match(self,report_coverage=True,save_csv=True):
        if len(self.tags)!=1:
            self.df_tracks_out,self.validation_frames=mm.RFID_matching(self.df_tracks,self.tags,self.config_dic_analysis,
                                                                       self.path)
            self.df_tracks_out=mm.match_left_over_tag(self.df_tracks_out,self.tags,self.config_dic_analysis)
            self.df_tracks_out=mm.tag_left_recover_simp(self.df_tracks_out,self.tags)
            self.df_tracks_out=mm.interaction2dic(self.df_tracks_out,self.tags,self.config_dic_analysis['itc_slack'])
            self.df_tracks_out=self.df_tracks_out[['frame','Time','sort_tracks','RFID_tracks','ious_interaction','Interactions',
                                                   'motion','motion_roi','RFID_matched','RFID_readings','Correction',
                                                   'Matching_details']]
            if save_csv:
                self.df_tracks_out.to_csv(self.path+'/RFID_tracks.csv')
                #print(f'csv file saved at {self.path+"/RFID_tracks.csv"}')
            if report_coverage:
                coverage=mm.coverage(self.df_tracks_out)
            return self.df_tracks_out,self.validation_frames,coverage
        else:
            self.validation_frames=[]
            self.df_tracks['lost_tracks']=self.df_tracks['sort_tracks'].values
            self.df_tracks_out=mm.tag_left_recover_simp(self.df_tracks,self.tags)
            self.df_tracks_out.to_csv(self.path+'/RFID_tracks.csv')
            if report_coverage:
                coverage=mm.coverage(self.df_tracks_out)
            self.df_tracks_out=mm.interaction2dic(self.df_tracks_out,self.tags,self.config_dic_analysis['itc_slack'])
            self.df_tracks_out=self.df_tracks_out[['frame','Time','sort_tracks','RFID_tracks','ious_interaction','Interactions',
                                                   'motion','motion_roi','RFID_matched']]
            self.df_tracks_out['Correction']=[[] for i in range(len(self.df_tracks_out))]
            self.df_tracks_out['Matching_details']=[[] for i in range(len(self.df_tracks_out))]
            self.df_tracks_out['RFID_readings']=[[] for i in range(len(self.df_tracks_out))]
            if save_csv:
                self.df_tracks_out.to_csv(self.path+'/RFID_tracks.csv')
                print(f'csv file saved at {self.path+"/RFID_tracks.csv"}')
            return self.df_tracks_out,self.validation_frames,coverage
    def find_activte_mice(self,save_csv=True):
        self.df_tracks_out['Activity']=[mm.get_tracked_activity(motion_status,motion_roi,RFID_tracks,self.tags) for motion_status,
                                        motion_roi,RFID_tracks in zip(self.df_tracks_out['motion'].values,
                                                                      self.df_tracks_out['motion_roi'].values,
                                                                      self.df_tracks_out['RFID_tracks'].values)]
        if save_csv:
            self.df_tracks_out.to_csv(self.path+'/RFID_tracks.csv')
            #print(f'csv file saved at {self.path+"/RFID_tracks.csv"}')
        return self.df_tracks_out
    def load_dlc_bpts(self):
        print('Loading Deeplabcut body parts to PRT')
        columns=['frame']+ self.config_dic_dlc['dbpt']
        dics={y: eval for y in columns}
        df_bpts=pd.read_csv(self.path+'/'+'dlc_bpts.csv',converters=dics)
        df_dbpt_columns=[f'df_bpts["{i}"]' for i in self.config_dic_dlc['dbpt']]
        df_bpts['bpts']=eval('+'.join(df_dbpt_columns))
        df_bpts['frame']=range(len(df_bpts))
        df_bpts=df_bpts.drop(columns=self.config_dic_dlc['dbpt'])
        columns=['bboxes']
        #self.df_tracks_out.to_csv('test.csv')
        self.df_tracks_out=self.df_tracks_out[['frame', 'Time','sort_tracks', 'RFID_tracks', 
                                               'ious_interaction', 'Interactions','motion', 
                                               'motion_roi', 'RFID_matched', 'Activity']]
        self.df_tracks_out=pd.merge( self.df_tracks_out,df_bpts, on='frame')
        dbpts=[mm.rfid2bpts(bpts,RFIDs,self.config_dic_dlc['dbpt_box_slack'],bpt2look=self.config_dic_dlc['dbpt_distance_compute']) 
               for bpts,RFIDs in zip(self.df_tracks_out['bpts'].values,self.df_tracks_out['RFID_tracks'].values)]
        self.df_tracks_out['dbpt2look']=[i[0] for i in dbpts]
        self.df_tracks_out['undetemined_bpt']=[i[1] for i in dbpts]
        list_bpts=list(map(sublist_decompose,self.df_tracks_out.dbpt2look.values.tolist()))
        for i in self.tags: 
            exec(f'list_bpt_{str(i)}=[]')
            for y in list_bpts:
                bpts=[v for v in y if v[3]==i]
                exec(f'list_bpt_{str(i)}.append(bpts)')
            self.df_tracks_out[f'{i}_bpts']=eval(f'list_bpt_{str(i)}')
        rows=self.df_tracks_out.apply(lambda x:mm.bpt_distance_compute(x,self.tags,self.config_dic_dlc['dbpt_int']),axis=1)
        new_cols=[str(list(i)[0]) + '_'+str(list(i)[1]) for i in itertools.combinations(self.tags, 2)]
        for name,idx in zip(new_cols,range(len(new_cols))):
            self.df_tracks_out[name]=[dists[idx] for dists in rows]
        print('Finished Loading Deeplabcut body parts to PRT')
        self.df_tracks_out.to_csv(self.path+'/RFID_tracks.csv')
        return 

    def compile_travel_trjectories(self,dlc=False):
        msg='Generating Individual Rodent Trajectory'
        pbar=tqdm(total=len(self.tags),position=0,leave=True,desc=msg)
        if not os.path.exists(self.path+'/trajectories'):
            os.mkdir(self.path+'/trajectories')
        for tag in self.tags:
            list_df=ta.location_compiler(tag,self.df_tracks_out,dlc,lim=5)
            if not os.path.exists(self.path+'/trajectories'+f'/{tag}'):
                os.mkdir(self.path+'/trajectories'+f'/{tag}')
            count=0
            if list_df != []:
                if dlc: 
                    list_df=list(map(ta.dbpts2xy,[[self.config_dic_dlc['dbpt'],df] for df in list_df]))
                for tracks in list_df:
                    tracks.to_csv(self.path+'/trajectories'+f'/{tag}'+f'/track_{count}.csv')
                    count+=1
                df_t=self.df_tracks_out[['Time','frame']]
                df_tag=pd.concat(list_df)
                df_tag=df_tag.sort_values(by=['frame'])
                df_tag_c=pd.merge(df_t,df_tag,on='frame',how='outer')
                df_itc=mm.itc_duration(self.df_tracks_out,tag,self.tags)
                df_tag_c=pd.merge(df_tag_c,df_itc,on='frame',how='outer')
                df_tag_c.iloc[np.where(df_tag_c['x'].isnull())[0],4:]=np.nan
                #df_tag_c.loc[df_tag_c.isnull().any(axis=1), :] = np.nan
                df_tag_c.to_csv(self.path+'/'+f'{tag}.csv')
            pbar.update(1)
        return 
    
    def generate_labeled_video(self,dlc_bpts=False,plot_motion=False,out_folder=None):
        generate_RFID_video(self.path,self.df_RFID,self.tags,self.df_tracks_out,\
                               self.validation_frames,self.config_dic_analysis,self.config_dic_dlc,plot_motion,out_folder=out_folder)
        
    def generate_validation_video(self,out_folder='None'):
        create_validation_Video(self.path,self.df_tracks_out,self.tags,self.config_dic_analysis,output=None)
    
    
    