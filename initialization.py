import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='')
    
    parser.add_argument("--name", default='test', help="[train, test, reduction] chooise one in this") 
    parser.add_argument("--snapshot_dir", default='half/cheakpoint', help="path of snapshots") 
    parser.add_argument("--out_dir", default='things/output', help="path of train outputs") 
    parser.add_argument("--image_size", type=int, default=256, help="load image size") 
    parser.add_argument("--random_seed", type=int, default=1234, help="random seed") 
    parser.add_argument('--base_lr', type=float, default=0.0002, help='initial learning rate for adam') 
    parser.add_argument('--epoch', dest='epoch', type=int, default=67, help='# of epoch') 
    parser.add_argument('--epoch_step', dest='epoch_step', type=int, default=100, help='# of epoch to decay lr') 
    parser.add_argument("--lamda", type=float, default=10.0, help="L1 lamda") 
    parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam') 
    parser.add_argument("--summary_pred_every", type=int, default=1000, help="times to summary.") 
    parser.add_argument("--write_pred_every", type=int, default=100, help="times to write image.") 
    parser.add_argument("--save_pred_every", type=int, default=20000, help="times to save ckpt.") 
    parser.add_argument("--human_number", type=int, default=650, help="human number for train.") 
    parser.add_argument("--x_train_data_path", default='half/trainA/', help="path of x training datas.") 
    parser.add_argument("--y_train_data_path", default='half/trainB/', help="path of y training datas.") 
    parser.add_argument("--svalue", type=int, default=0, help="person give train score")   
    parser.add_argument("--coefficient", type=int, default=1, help="gen_loss coefficient") 
    parser.add_argument("--duration", type=int, default=250, help="duration") 
    
    parser.add_argument("--test_data_path", default='half/image', help="path of x test dataset") 
    parser.add_argument("--test_out_dir", default='half/matting_output2/',help="Output Folder") 
    args = parser.parse_args()
    return args