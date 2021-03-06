import os
import tensorflow as tf
import numpy as np
import cv2
from test_image_reader import TestImageReader
from net import Encoder_Decoder
from initialization import parse_args
from ops import make_test_data_list, get_picture
tf.reset_default_graph()
args = parse_args()

def test():
    if not os.path.exists(args.test_out_dir): 
        os.makedirs(args.test_out_dir)
 
    datalists = make_test_data_list(args.test_data_path) 
    test_image = tf.placeholder(tf.float32,shape=[1, 256, 256, 3], name = 'test_image')
    print('1')
    fake = Encoder_Decoder(image=test_image, reuse=False, name='Encoder_Decoder')
 
    restore_var = [v for v in tf.global_variables() if 'Encoder_Decoder' in v.name]
 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True 
    sess = tf.Session(config=config)
    
    saver = tf.train.Saver(var_list=restore_var,max_to_keep=1) 
    checkpoint = tf.train.latest_checkpoint(args.snapshot_dir) 
    saver.restore(sess, checkpoint) 
 
    total_step = len(datalists)
    for step in range(total_step):
        test_image_name, test_img = TestImageReader(datalists, step, args.image_size) 
        batch_image = np.expand_dims(np.array(test_img).astype(np.float32), axis = 0) 
        feed_dict = { test_image : batch_image} 
        fake_value = sess.run(fake, feed_dict=feed_dict) 
        write_image= get_picture(test_img, fake_value) 
        write_image_name = args.test_out_dir + "/"+ test_image_name + ".png" 
        cv2.imwrite(write_image_name, write_image) 
        print('step {:d}'.format(step))
