from multiprocessing.pool import ThreadPool
from collections import deque
import cv2


def create_validation_Video(path,config_dic,tags,df1,validation_frames,resolution):
    source=path+'/raw.avi'
    cap = cv2.VideoCapture(source)
    thread_num = cv2.getNumberOfCPUs()
    pool = ThreadPool(processes=thread_num)
    pending_task = deque()
    while True:
        while len(pending_task) > 0 and pending_task[0].ready():
            res = pending_task.popleft().get()
        if len(pending_task) < thread_num:
            frame_got, frame = cap.read()











        
def get_write_frames_details(df1,validation_frames):
    yolo_bboxes=df1.sort_tracks.values
    rfid_tracks=df1.RFID_tracks.values
    RFID_readings={i:df1.iloc[i]['RFID_readings'] for i in validation_frames}
    corrections={i:v for i,v in enumerate(df1.Correction) if v!= []}
    correction_frames=[frame for frame in corrections.keys()]
    matched={i:v for i,v in enumerate(df1.RFID_matched.values) if v!= []}
    matched_frames=[frame for frame in matched.keys()]
    match_details={i:v for i,v in enumerate(df1.Matching_details.values) if v!= []}
    md_frames=[frame for frame in match_details.keys()]
    return yolo_bboxes,rfid_tracks,RFID_readings,corrections,correction_frames,matched,matched_frames,match_details,md_frames

def validation_frame_gen(config_dic,tags,frame,frame_count,
                         yolo_bboxes,rfid_tracks,RFID_readings,
                         corrections,correction_frames,matched,
                         matched_frames,match_details,md_frames,
                         validation_frames):
    max_mice=len(tags)
    entrance_reader=config_dic['entrance_reader']
    RFID_coords=config_dic['RFID_readers']
    colors=['green','blue','purple','black','orange','red','yellow','aqura','magenta','lbrown']
    tag_codes={tag: num+1 for num, tag in zip(range(len(tags)),tags)}
    tag_codes={i:[tag_codes[i],z] for i,z in zip(tag_codes.keys(),colors)}
    color_dic={'green':(0,255,0),'blue':(255,0,0),'purple':(128,0,128),'black':(0,0,0),
               'orange':(155,140,0),'red':(0,0,255),'yellow':(0,255,255),'aqura':(212,255,127),
               'magenta':(255,0,255),'lbrown':(181,228,255)}
    width=frame.shape[1]
    height=frame.shape[0]
    if entrance_reader != None:
        entrance_reader=entrance_reader
    else:
        entrance_reader=None
    if len([i for i in validation_frames if i>=frame_count])>0:
        RFID_frame=[i for i in validation_frames if i>=frame_count][0]
    else:
        RFID_frame=0
    if len([i for i in correction_frames if i>=frame_count])>0:
        correction_frame=[i for i in correction_frames if i>=frame_count][0]
    else:
        correction_frame=0
    if len([i for i in matched_frames if i>=frame_count])>0:
        matched_frame=[i for i in matched_frames if i>=frame_count][0]
    else:
         matched_frame=0
    if len([i for i in md_frames if i>=frame_count])>0:
        md_frame=[i for i in md_frames if i>=frame_count][0]
    else:
        md_frame=0
    rfid_tracker=rfid_tracks[frame_count]
    yolo_dets=yolo_bboxes[frame_count]
    #corrections_display=corrections[frame_count]
    #matched_display=matched[frame_count]
    #bpt_plot=bpts[frame_count]
    img_yolo=frame.copy()
    img_rfid=frame.copy()
    for objects in rfid_tracker:
        xmin, ymin, xmax, ymax, index = int(objects[0]), int(objects[1]),\
            int(objects[2]), int(objects[3]), int(objects[4])
        m_display=tag_codes[index][0]
        m_color=color_dic[tag_codes[index][1]]
        cv2.rectangle(img_rfid, (xmin, ymin), (xmax, ymax), m_color, 3)   
        cv2.putText(img_rfid, str(m_display), (xmin, ymin-20), 0, 5e-3 * 200, m_color, 3)
    for objects in yolo_dets:
        xmin, ymin, xmax, ymax, index = int(objects[0]), int(objects[1]),\
            int(objects[2]), int(objects[3]), int(objects[4])
        cv2.rectangle(img_yolo, (xmin, ymin), (xmax, ymax), (0,255,0), 3)    
        cv2.putText(img_yolo, str(index), (xmin, ymin+10), 0, 5e-3 * 200, (0,0,255), 3)
    for i,v in RFID_coords.items():
        if i!=entrance_reader:
            xmin, ymin, xmax, ymax=v[0],v[1],v[2],v[3]
            cv2.rectangle(img_rfid, (xmin, ymin), (xmax, ymax), (0,0,0), 2)
            cent_point=bbox_to_centroid(v)
            cv2.putText(img_rfid,f"{str(i)}",(cent_point[0],cent_point[1]),0, 5e-3 * 200,(0,0,0),2)
    if 3*width +100<2800:
        width_b=2300
    else:
        width_b=3*width+300
    blankimg=255*np.ones(shape=[height+400,width_b,3],dtype=np.uint8)
    blankimg[200:height+200,100:width+100]=img_yolo
    blankimg[200:height+200,width+150:2*width+150]=img_rfid
    cv2.putText(blankimg,f"Frame: {str(frame_count)}",(int(0.5*width),50),0, 5e-3 * 250,(0,0,255),2)
    cv2.putText(blankimg,f"Maximum of Mice in Video: {max_mice}",(int(0.5*width)+275,50),0, 5e-3 * 250,(0,0,255),2)
    cv2.putText(blankimg,f"Current Mice in Video: {len(yolo_dets)}",(int(0.5*width)+950,50),0, 5e-3 * 250,(0,0,255),2)
    cv2.putText(blankimg,'SORT ID Tracking',(100,150),0, 5e-3 * 250,(0,0,255),5)
    cv2.putText(blankimg,'RFID Tracking',(150+int(width),150),0, 5e-3 * 250,(0,0,255),5)
    if RFID_frame != [] and RFID_frame !=0:
        spacer=0
        for i in RFID_readings[RFID_frame]:
            if i[1] in tags:
                if i[0] == entrance_reader:
                    reader_display='Entrancer'
                else:
                    reader_display=i[0]
                tag_display=tag_codes[i[1]][0]
                cv2.putText(blankimg,f'Frame {RFID_frame}: reader {str(reader_display)}    tag read {tag_display}', \
                            (int(1.4*width),int(height+300+spacer)),0,5e-3 * 210,(0,0,255),3)#1.4*height+spacer
                spacer+=50        
    cv2.putText(blankimg,'RFID Tag Codes',(2*width+175,150),0,5e-3 * 210,(0,0,255),5)
    spacer=0
    for i,v in tag_codes.items():
        cv2.putText(blankimg,f"{str(i)} = {str(v[0])}",(2*width+175,230+spacer),0,5e-3 * 180,color_dic[v[1]],3)
        spacer+=35
    spacer+=10
    cv2.putText(blankimg,'RFID-SID Matching Log',(2*width+175,250+spacer),0,5e-3 * 210,(0,0,0),4)
    spacer+=35
    if correction_frame != []  and correction_frame !=0:
        for i in corrections[correction_frame]:
            for item,value in i[3].items():
                if value != None:
                    cv2.putText(blankimg,f'Correction on SID {item} from frame {value} to frame {i[2]} ', \
                                (2*width+175,250+spacer),0,5e-3 * 200,(0,0,255),2)#1.4*height+spacer
                spacer+=50
    if matched_frame != [] and matched_frame != 0:
        for i in matched[matched_frame]:
            if type(i[1])!= str:
                if type(i[2])!=str:
                    tag_display=tag_codes[i[1]][0]
                    cv2.putText(blankimg,f'Frame: {matched_frame} {tag_display} matched to sid: {i[0]} ', \
                                (2*width+175,250+spacer),0,5e-3 * 200,(0,0,255),2)#1.4*height+spacer
                    spacer +=50
                else:
                    tag_display=tag_codes[i[1]][0]
                    cv2.putText(blankimg,f'Frame {matched_frame}:Last tag {tag_display} matched to sid: {i[0]} ', \
                                (2*width+175,250+spacer),0,5e-3 * 180,(0,0,255),2)#1.4*height+spacer
                    spacer +=50
            else:
                tag_display=tag_codes[i[2]][0]
                cv2.putText(blankimg,f'Frame {matched_frame}: {tag_display} {i[1]} matched to sid: {i[0]} ', \
                            (2*width+175,250+spacer),0,5e-3 * 180,(0,0,255),2)#1.4*height+spacer
                spacer+=50        
    if  md_frame != [] and md_frame !=0:
        for i in match_details[md_frame]:
                tag_display=tag_codes[int(i[1])][0]
                cv2.putText(blankimg,f'frame {md_frame}: {i[0]} {tag_display}', \
                            (2*width+175,250+spacer),0,5e-3 * 200,(0,0,255),2)
                spacer+=30
    return blankimg





def create_validation_Video(path,config_dic,tags,df1,validation_frames,resolution):
    source=path+'/raw.avi'
    fvs = FileVideoStream(source).start()
    vid=cv2.VideoCapture(path+'/raw.avi')
    vid_length=int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    vid.release()
    codec = cv2.VideoWriter_fourcc(*'XVID')
    output= cv2.VideoWriter(path+'/val.avi', codec, fps, (resolution[0], resolution[1]),True)
    yolo_bboxes,rfid_tracks,RFID_readings,corrections,\
        correction_frames,matched,matched_frames,\
            match_details,md_frames=get_write_frames_details(df1,validation_frames)
    frame_count=0
    pbar=tqdm(total=vid_length,leave=True)
    while fvs.more():
        frame = fvs.read()
        frame=validation_frame_gen(config_dic,tags,frame,frame_count,
                         yolo_bboxes,rfid_tracks,RFID_readings,
                         corrections,correction_frames,matched,
                         matched_frames,match_details,md_frames,
                         validation_frames)
        frame=cv2.resize(frame,(resolution[0],resolution[1]))
        output.write(frame)
        frame_count+=1
        pbar.update(1)
    fvs.stop()
    output.release()
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        