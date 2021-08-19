import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import pandas as pd
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from configparser import ConfigParser
import imutils
from threading import Thread
from datalogger import datalogger
from datetime import datetime, timedelta
from RFID_reader import RFID_reader
import os


flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-best2',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', '0', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.35, 'iou threshold')
flags.DEFINE_float('score', 0.6, 'score threshold')
flags.DEFINE_boolean('dont_show', True, 'dont show video output')
flags.DEFINE_integer('motion_interpolation',2,'Frames of consecutive no_motion to ignore')
flags.DEFINE_integer('len_motion',15,' Minimum length of consecutive motion frames to keep')
flags.DEFINE_integer('min_area',100,'Minimum area of contour changes to consider for motion detection')
flags.DEFINE_integer('blur_filter_k_size',11,'area to blur (input x input)')
flags.DEFINE_integer('inten_thres',25,'Intensity difference in backgroud subtraction to consider for motion detections')
flags.DEFINE_boolean('show_motion',True,'Show Changes in contours detected')

def remove_overinter(x):
    first_idx=x.first_valid_index()
    last_idx=x.last_valid_index()
    x2=x.loc[first_idx:last_idx]
    return x2
def consecutive_motion_detection(path,gap_fill=2,frame_thres=15):
    df1=pd.read_csv(path+'/yolo_dets.csv')
    temp=[1 if i =='Motion' else np.nan for i in df1.motion.values]
    temp=pd.Series(temp)
    temp=temp.interpolate(method ='linear',limit_direction ='both', limit = gap_fill)
    temp=np.split(temp, np.where(np.isnan(temp))[0])
    temp=[t.drop(t.index[0]) for t in temp if len(t)>frame_thres]
    inds=[i.index.tolist() for i in temp]
    ind=[i for sublists in inds for i in sublists ]
    values=['Motion' if i in ind else 'No_motion' for i in range(len(df1))]
    df1['motion']=values
    df1.to_csv(path+'/yolo_dets.csv')
    return df1

from sort import Sort
def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']
    config = ConfigParser()
    config.read('config.ini')
    cfg = 'tracker_cage_record'
    data_root = config.get(cfg, 'data_root')
    rfid=config.get(cfg,'rfid')
    nreaders=int(config.get(cfg,'nreaders'))
    tm = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    data_root = data_root + tm
    os.makedirs(data_root)
    # begin video capture
    vid = cv2.VideoCapture(0)
    resolution = list(map(int, config.get(cfg, 'resolution').split(', ')))
    print(resolution)
    vid.set(3,resolution[0])
    vid.set(4,resolution[1])
    #vid.set(cv2.CAP_PROP_FPS,30)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(width)
    print(height)
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
    out_original = cv2.VideoWriter(data_root+'/raw.avi', codec, fps, (width, height))
    out_label= cv2.VideoWriter(data_root+'/lab.avi', codec, fps, (width, height))
    with open(data_root+'/frames.csv','w') as file:
        file.write('frame,Time,bboxes,sort_tracks,motion,motion_roi,fps\n')
    mot_tracker=Sort()
    mot_tracker.reset_count()
    frame_count=0
    if rfid =='True':
        readers=["reader{}=RFID_reader('/dev/ttyUSB{}', '{}',data_root+'/text{}.csv')".format(i,i,i,i) for i in range(nreaders)]
    for i in readers:
        exec(i)
    if rfid =='True':
        reader_process=["t_rfid{}=Thread(target=reader{}.scan,daemon=True)".format(i,i) for i in range(nreaders)]
    for i in reader_process:
        exec(i)
    if rfid =='True':
        reader_startup=["t_rfid{}.start()".format(i) for i in range(nreaders)]
    for i in reader_startup:
        exec(i)
    avg=None
    while True:
        return_value, frame = vid.read()
        
        if return_value:
            out_original.write(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break 
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        image = utils.draw_bbox(frame, pred_bbox)
        ds_boxes_array=np.asarray(image[1])
        trackers = mot_tracker.update(ds_boxes_array)
        avg, status,cnt_bb= utils.motion_detection(frame,avg,k_size=\
                                                   (FLAGS.blur_filter_k_size,FLAGS.blur_filter_k_size)\
                                                       ,min_area=FLAGS.min_area,intensity_thres=FLAGS.inten_thres)
        fps = 1.0 / (time.time() - start_time)
        #print("FPS: %.2f" % fps)
        #result = np.asarray(image[0])
        with open(data_root+'/frames.csv','a') as file:
            file.write(f'{str(frame_count)},{time.time()},"{image[1]}","{trackers.tolist()},{status},"{cnt_bb}","{fps}"\n')
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        #frame = imutils.resize(result, width=resolution[0])
        for objects in trackers:
            xmin, ymin, xmax, ymax, index = int(objects[0]), int(objects[1]), int(objects[2]), int(objects[3]), int(objects[4])
            #sort_tracker=[xmin, ymin, xmax, ymax, index]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255,0,0), 3)
            cv2.putText(frame, str(index), (xmin, ymin), 0, 5e-3 * 200, (0,255,0), 3)
        if FLAGS.show_motion:
            if status == 'Motion':
                for c in cnt_bb:
                            xmin, ymin, xmax, ymax = int(c[0]), int(c[1]),int(c[2]), int(c[3])
                            sub_img = frame
                            white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
                            frame = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)              
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if FLAGS.dont_show:
            cv2.imshow("result", frame)
        frame_count+=1
        out_label.write(frame)
        if cv2.waitKey(30) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        #print("FPS: %.2f" % fps)
        consecutive_motion_detection(os.path.dirname(i),gap_fill=FLAGS.motion_interpolation,frame_thres=FLAGS.len_motion)
        pass
