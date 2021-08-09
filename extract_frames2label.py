from core.frame_extraction import *
from absl.flags import FLAGS
from absl import flags, app
from skimage.util import img_as_ubyte
from skimage import io
import numpy as np
import os

#from Deeplabcut 


flags.DEFINE_integer('numframes2pick', 50, 'number of frames to extract for labeling')
flags.DEFINE_string('video_folder','./data','path to videos')





def main(_argv):
    output_path=FLAGS.video_folder+'labeled_data'
    print(output_path)
    vids=[FLAGS.video_folder+i+'/raw.avi' for i in os.listdir(FLAGS.video_folder)]
    os.mkdir(output_path)
    count=0
    for i in vids:
        cap = VideoReader(i)
        #nframes = len(cap)
        frames2pick = KmeansbasedFrameselectioncv2(cap, FLAGS.numframes2pick, 0, 1,
                                                   None,None, step=1,
                                                   resizewidth=30,
                                                   color=False)
        #indexlength = int(np.ceil(np.log10(nframes)))
        for index in frames2pick:
            cap.set_to_frame(index)
            frame = cap.read_frame()
            if frame is not None:
                image = img_as_ubyte(frame)
                img_name = (str(output_path)+ "/img"+ str(count)+ ".png")
                io.imsave(img_name, image)
                count+=1
        cap.close()
        
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
