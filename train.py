import random
import os
import tensorflow as tf
import numpy as np
import cv2
from train_image_reader import TrainImageReader
from net import Encoder_Decoder, discriminator
from ops import make_train_data_list, save, gan_loss, l1_loss, get_write_picture
from initialization import parse_args
tf.reset_default_graph()

def train():
    args = parse_args()
    if not os.path.exists(args.snapshot_dir): 
        os.makedirs(args.snapshot_dir)
    if not os.path.exists(args.out_dir): 
        os.makedirs(args.out_dir)

    x_datalists, y_datalists = make_train_data_list(args.x_train_data_path, args.y_train_data_path) 
    tf.set_random_seed(args.random_seed) 
    x_img = tf.placeholder(tf.float32,shape=[1, args.image_size, args.image_size,3],name='x_img') 
    y_img = tf.placeholder(tf.float32,shape=[1, args.image_size, args.image_size,3],name='y_img') 

    fake_y = Encoder_Decoder(image=x_img, reuse=False, name='Encoder_Decoder') 
    dy_fake = discriminator(image=fake_y, reuse=False, name='discriminator_fake') 
    dx_real = discriminator(image=x_img, reuse=False, name='discriminator_real1') 
    dy_real = discriminator(image=y_img, reuse=False, name='discriminator_real2')

    encoder_loss = gan_loss(dy_fake, tf.ones_like(dy_fake)) + args.lamda*l1_loss(y_img, fake_y)
    dis_loss = gan_loss(dy_fake, tf.zeros_like(dy_fake)) + gan_loss(dx_real, tf.ones_like(dx_real)) + gan_loss(dy_real, tf.ones_like(dx_real))
    
    dis_loss = args.coefficient * dis_loss + (1 - args.coefficient) * args.svalue

    encoder_loss_sum = tf.summary.scalar("encoder_loss", encoder_loss)
    discriminator_sum = tf.summary.scalar("dis_loss", dis_loss)

    summary_writer = tf.summary.FileWriter(args.snapshot_dir, graph=tf.get_default_graph()) 

    g_vars = [v for v in tf.trainable_variables() if 'Encoder_Decoder' in v.name]
    d_vars = [v for v in tf.trainable_variables() if 'discriminator' in v.name]

    lr = tf.placeholder(tf.float32, None, name='learning_rate')
    d_optim = tf.train.AdamOptimizer(lr, beta1=args.beta1) 
    e_optim = tf.train.AdamOptimizer(lr, beta1=args.beta1) 
    d_grads_and_vars = d_optim.compute_gradients(dis_loss, var_list=d_vars) 
    d_train = d_optim.apply_gradients(d_grads_and_vars) 
    e_grads_and_vars = e_optim.compute_gradients(encoder_loss, var_list=g_vars) 
    e_train = e_optim.apply_gradients(e_grads_and_vars) 

    train_op = tf.group(d_train, e_train)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=100) 
        counter = 0
        number = 0
        ckpt = tf.train.get_checkpoint_state(args.snapshot_dir)  
        if ckpt and ckpt.model_checkpoint_path:        
            saver.restore(sess,ckpt.model_checkpoint_path)       
            print("Model restored...")    
        else:
            print('No Model')

        for epoch in range(args.epoch): 
           
            if number < 50000:
                if epoch < 50:
                    lrate = args.base_lr if epoch < args.epoch_step else args.base_lr*(args.epoch-epoch)/(args.epoch-args.epoch_step) 
                    for step in range(len(x_datalists)): 
                        counter += 1
                        x_image_resize, y_image_resize = TrainImageReader(x_datalists, y_datalists, step, args.image_size) 
                        batch_x_image = np.expand_dims(np.array(x_image_resize).astype(np.float32), axis = 0) 
                        batch_y_image = np.expand_dims(np.array(y_image_resize).astype(np.float32), axis = 0) 
                        feed_dict = { lr : lrate, x_img : batch_x_image, y_img : batch_y_image} 
                        encoder_loss_value, dis_loss_value, _ = sess.run([encoder_loss, dis_loss, train_op], feed_dict=feed_dict) 
                        if counter % args.save_pred_every == 0: 
                            save(saver, sess, args.snapshot_dir, counter)
                        if counter % args.summary_pred_every == 0: 
                            encoder_loss_sum_value, discriminator_sum_value = sess.run([encoder_loss_sum, discriminator_sum], feed_dict=feed_dict)
                            summary_writer.add_summary(encoder_loss_sum_value, counter)
                            summary_writer.add_summary(discriminator_sum_value, counter)
                        if counter % args.write_pred_every == 0:  
                            fake_y_value= sess.run(fake_y, feed_dict=feed_dict)
                            write_image = get_write_picture(x_image_resize, y_image_resize, fake_y_value)
                            write_image_name = args.out_dir + "/out"+ str(counter) + ".png" 
                            cv2.imwrite(write_image_name, write_image) 
                        if counter % args.human_number == 0:
                            for continued in range(args.duration):
                                args.svalue = random.uniform(0, 0.1)
                                args.coefficient = 0.3
                            number += 1
                        print('epoch {:d} step {:d} \t encoder_loss = {:.3f}, dis_loss = {:.3f}'.format(epoch, step, encoder_loss_value, dis_loss_value))

                elif epoch > 49 and epoch < 80 :
                    lrate = args.base_lr if epoch < args.epoch_step else args.base_lr*(args.epoch-epoch)/(args.epoch-args.epoch_step) 
                    for step in range(len(x_datalists)):
                        counter += 1
                        x_image_resize, y_image_resize = TrainImageReader(x_datalists, y_datalists, step, args.image_size) 
                        batch_x_image = np.expand_dims(np.array(x_image_resize).astype(np.float32), axis = 0) 
                        batch_y_image = np.expand_dims(np.array(y_image_resize).astype(np.float32), axis = 0) 
                        feed_dict = { lr : lrate, x_img : batch_x_image, y_img : batch_y_image} 
                        encoder_loss_value, dis_loss_value, _ = sess.run([encoder_loss, dis_loss, train_op], feed_dict=feed_dict) 
                        if counter % args.save_pred_every == 0: 
                            save(saver, sess, args.snapshot_dir, counter)
                        if counter % args.summary_pred_every == 0: 
                            encoder_loss_sum_value, discriminator_sum_value = sess.run([encoder_loss_sum, discriminator_sum], feed_dict=feed_dict)
                            summary_writer.add_summary(encoder_loss_sum_value, counter)
                            summary_writer.add_summary(discriminator_sum_value, counter)
                        if counter % args.write_pred_every == 0:  
                            fake_y_value= sess.run(fake_y, feed_dict=feed_dict)
                            write_image = get_write_picture(x_image_resize, y_image_resize, fake_y_value)
                            write_image_name = args.out_dir + "/out"+ str(counter) + ".png" 
                            cv2.imwrite(write_image_name, write_image)
                        if counter % args.human_number == 0:
                            for continued in range(args.duration):
                                args.svalue = random.uniform(0.2, 0.5)
                                args.coefficient = 0.3
                            number += 1
                        print('epoch {:d} step {:d} \t encoder_loss = {:.3f}, dis_loss = {:.3f}'.format(epoch, step, encoder_loss_value, dis_loss_value))

                else:
                    lrate = args.base_lr if epoch < args.epoch_step else args.base_lr*(args.epoch-epoch)/(args.epoch-args.epoch_step) 
                    for step in range(len(x_datalists)): 
                        counter += 1
                        x_image_resize, y_image_resize = TrainImageReader(x_datalists, y_datalists, step, args.image_size) 
                        batch_x_image = np.expand_dims(np.array(x_image_resize).astype(np.float32), axis = 0) 
                        batch_y_image = np.expand_dims(np.array(y_image_resize).astype(np.float32), axis = 0) 
                        feed_dict = { lr : lrate, x_img : batch_x_image, y_img : batch_y_image} 
                        encoder_loss_value, dis_loss_value, _ = sess.run([encoder_loss, dis_loss, train_op], feed_dict=feed_dict) 
                        if counter % args.save_pred_every == 0: 
                            save(saver, sess, args.snapshot_dir, counter)
                        if counter % args.summary_pred_every == 0: 
                            encoder_loss_sum_value, discriminator_sum_value = sess.run([encoder_loss_sum, discriminator_sum], feed_dict=feed_dict)
                            summary_writer.add_summary(encoder_loss_sum_value, counter)
                            summary_writer.add_summary(discriminator_sum_value, counter)
                        if counter % args.write_pred_every == 0:  
                            fake_y_value= sess.run(fake_y, feed_dict=feed_dict)
                            write_image = get_write_picture(x_image_resize, y_image_resize, fake_y_value)
                            write_image_name = args.out_dir + "/out"+ str(counter) + ".png" 
                            cv2.imwrite(write_image_name, write_image)
                        if counter % args.human_number == 0:
                            for continued in range(args.duration):
                                args.svalue = random.uniform(0.6, 0.9)
                                args.coefficient = 0.3
                            number += 1
                        print('epoch {:d} step {:d} \t encoder_loss = {:.3f}, dis_loss = {:.3f}'.format(epoch, step, encoder_loss_value, dis_loss_value))

            else:
                lrate = args.base_lr if epoch < args.epoch_step else args.base_lr*(args.epoch-epoch)/(args.epoch-args.epoch_step) 
                for step in range(len(x_datalists)): 
                    counter += 1
                    x_image_resize, y_image_resize = TrainImageReader(x_datalists, y_datalists, step, args.image_size) 
                    batch_x_image = np.expand_dims(np.array(x_image_resize).astype(np.float32), axis = 0) 
                    batch_y_image = np.expand_dims(np.array(y_image_resize).astype(np.float32), axis = 0) 
                    feed_dict = { lr : lrate, x_img : batch_x_image, y_img : batch_y_image} 
                    encoder_loss_value, dis_loss_value, _ = sess.run([encoder_loss, dis_loss, train_op], feed_dict=feed_dict) 
                    if counter % args.save_pred_every == 0: 
                        save(saver, sess, args.snapshot_dir, counter)
                    if counter % args.summary_pred_every == 0: 
                        encoder_loss_sum_value, discriminator_sum_value = sess.run([encoder_loss_sum, discriminator_sum], feed_dict=feed_dict)
                        summary_writer.add_summary(encoder_loss_sum_value, counter)
                        summary_writer.add_summary(discriminator_sum_value, counter)
                    if counter % args.write_pred_every == 0:  
                        fake_y_value= sess.run(fake_y, feed_dict=feed_dict)
                        write_image = get_write_picture(x_image_resize, y_image_resize, fake_y_value)
                        write_image_name = args.out_dir + "/out"+ str(counter) + ".png" 
                        cv2.imwrite(write_image_name, write_image)
                    print('epoch {:d} step {:d} \t encoder_loss = {:.3f}, dis_loss = {:.3f}'.format(epoch, step, encoder_loss_value, dis_loss_value))
