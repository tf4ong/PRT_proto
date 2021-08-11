import pandas as pd 
from psyco_utils import mouse_matching as mm
import os
import psyco_utils.trajectory_analysis_utils as ta
import sys
from psyco_utils.track_utils import *
import itertools
from generate_vid import generate_RFID_video,create_validation_Video
from psyco_utils.detect_utils import yolov4_detect
from collections import ChainMap


"""
Open arena (i.e. no entrance/exit arena) speed not optomized
may need adiitionl methods.
"""
class PSYCO:
    def __init__(self,path,config_path):
        self.path=path 
        
        with open(path+'/'+'logs.txt','r') as f:
            tags=f.readlines()
            self.tags=[int(i) for i in tags[1][6:].split(',')]
        self.config_path=config_path
        self.config_dict_analysis=analysis_config_loader(config_path)
        self.config_dic_detect=detect_config_loader(config_path)
        if self.config_dict_analysis['entrance_reader'][0]: 
            self.entrance_RFID=self.config_dict_analysis['entrance_reader'][1]
        else: 
            self.entrance_RFID=None
        self.RFID_coords=self.config_dict_analysis['RFID_readers']
    def detect_mice(self,write_vid=False):
        yolov4_detect(self.path,self.config_dic_detect['size'],
                      self.config_dic_detect['weightpath'][0],
                      self.config_dic_detect['iou'],
                      self.config_dic_detect['score'],
                      self.config_dic_detect['blur_filter_k_size'],
                      self.config_dic_detect['motion_area_thresh'],
                      self.config_dic_detect['intensity_thres'],
                      self.config_dic_detect['motion_interpolation'],
                      self.config_dic_detect['len_motion_thres'],
                      write_vid)
        return
    def load_RFID(self):
        try:
            self.df_RFID=mm.RFID_readout(self.path,self.entrance_RFID,
                                         self.config_dict_analysis['entrance_time_thres'])
        except Exception as e:
            print(e)
            print('Confirm correct folder path')
            sys.exit(0)
        n_RFID_readings=len(self.df_RFID[self.df_RFID['RFID_readings'].notnull()])
        duration=self.df_RFID.iloc[-1]['Time']-self.df_RFID.iloc[0]['Time']
        print(f'{n_RFID_readings} Tags were read in {duration} seconds')
    def load_dets(self):
        self.df_tracks=mm.read_yolotracks(self.path,self.RFID_coords,self.entrance_RFID,
                                          self.config_dict_analysis['entrance_distance'],self.df_RFID,
                                          self.config_dict_analysis['interaction_thres'],self.config_dict_analysis['entrance_time_thres'],
                                          len(self.tags),self.config_dict_analysis['resolution'])
        if self.entrance_RFID is None:
            self.df_tracks=mm.reconnect_tracks_ofa(self.df_tracks,len(self.tags))
            pass
        else:
            reconnect_ids=mm.get_reconnects(self.df_tracks)
            self.df_tracks,id2remove=mm.spontanuous_bb_remover(self.config_dict_analysis['iou_min_sbb_checker'],
                                                                       reconnect_ids,self.df_tracks,self.RFID_coords,
                                                                       self.config_dict_analysis['sbb_frame_thres'],self.entrance_RFID,
                                                                       self.config_dict_analysis['entrance_distance'])       
            reconnect_ids=mm.reconnect_id_update(reconnect_ids,id2remove)
            trac_dict_leap=mm.reconnect_leap(reconnect_ids,self.df_tracks,self.config_dict_analysis['leap_dist'])
            self.df_tracks=mm.replace_track_leap_df(trac_dict_leap,self.df_tracks,self.RFID_coords,
                                                    self.entrance_RFID,self.config_dict_analysis['entrance_distance'])
        self.df_tracks=mm.Combine_RFIDreads(self.df_tracks,self.df_RFID)
        return self.df_tracks
    def RFID_match(self,report_coverage=True,save_csv=True):
        self.df_tracks_out,self.validation_frames=mm.RFID_matching(self.df_tracks,self.tags,self.entrance_RFID,
                                                                   self.RFID_coords,self.config_dict_analysis['entr_frames'],
                                                                   self.config_dict_analysis['correct_iou'],
                                                                   self.config_dict_analysis['reader_thres'],
                                                                   self.config_dict_analysis['RFID_dist'],
                                                                   self.config_dict_analysis['entrance_distance'],
                                                                   self.path)
        self.df_tracks_out=mm.match_left_over_tag(self.df_tracks_out,self.tags,self.entrance_RFID,
                                                   self.RFID_coords,self.config_dict_analysis['entrance_distance'],
                                                   self.config_dict_analysis['correct_iou'])
        self.df_tracks_out=mm.tag_left_recover_simp(self.df_tracks_out,self.tags)
        self.df_tracks_out.to_csv(self.path+'/RFID_tracks.csv')
        if report_coverage:
            coverage=mm.coverage(self.df_tracks_out)
        self.df_tracks_out=mm.interaction2dic(self.df_tracks_out,self.tags,0)
        self.df_tracks_out=self.df_tracks_out[['frame','Time','sort_tracks','RFID_tracks','ious_interaction','Interactions',
                                               'motion','motion_roi','RFID_readings','Correction','RFID_matched','Matching_details']]
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
            print(f'csv file saved at {self.path+"/RFID_tracks.csv"}')
        return self.df_tracks_out
    def load_dlc_bpts(self):
        print('Loading Deeplabcut body parts to PSYCO')
        columns=['frame']+ self.config_dict_analysis['dbpt']
        dics={y: eval for y in columns}
        df_bpts=pd.read_csv(self.path+'/'+'dlc_bpts.csv',converters=dics)
        df_dbpt_columns=[f'df_bpts["{i}"]' for i in self.config_dict_analysis['dbpt']]
        df_bpts['bpts']=eval('+'.join(df_dbpt_columns))
        df_bpts['frame']=range(len(df_bpts))
        df_bpts=df_bpts.drop(columns=self.config_dict_analysis['dbpt'])
        columns=['bboxes']
        self.df_tracks_out=pd.merge( self.df_tracks_out,df_bpts, on='frame')
        dbpts=[mm.rfid2bpts(bpts,RFIDs,self.config_dict_analysis['dbpt_box_slack'],bpt2look=self.config_dict_analysis['dbpt_distance_compute']) 
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
        rows=self.df_tracks_out.apply(lambda x:mm.bpt_distance_compute(x,self.tags,self.config_dict_analysis['dbpt_int']),axis=1)
        new_cols=[str(list(i)[0]) + '_'+str(list(i)[1]) for i in itertools.combinations(self.tags, 2)]
        for name,idx in zip(new_cols,range(len(new_cols))):
            self.df_tracks_out[name]=[dists[idx] for dists in rows]
        print('Finished Loading Deeplabcut body parts to PSYCO')
        self.df_tracks_out.to_csv(self.path+'/RFID_tracks.csv')
        return 

    def compile_travel_trjectories(self):
        print('Generating Individual Rodent Trajectory')
        if not os.path.exists(self.path+'/trajectories'):
            os.mkdir(self.path+'/trajectories')
        for tag in self.tags:
            list_df=ta.location_compiler(tag,self.df_tracks_out,lim=5)
            if not os.path.exists(self.path+'/trajectories'+f'/{tag}'):
                os.mkdir(self.path+'/trajectories'+f'/{tag}')
            count=0
            if list_df != []:
                for tracks in list_df:
                    tracks.to_csv(self.path+'/trajectories'+f'/{tag}'+f'/track_{count}.csv')
                    count+=1
        return 

    def generate_labeled_video(self,dlc_bpts=False,plot_motion=False,out_folder=None):
        generate_RFID_video(self.path,self.df_RFID,self.tags,self.df_tracks_out,\
                               self.validation_frames,self.RFID_coords,self.entrance_RFID,dlc_bpts,plot_motion,out_folder=out_folder)
        
    def generate_validation_video(self,out_folder='None'):
        create_validation_Video(self.path,self.config_path,output=None)
        
        
        
        
        
