import glob
import cv2
import numpy as np
from initialization import parse_args
import os

args = parse_args()
def mask():
    if not os.path.exists(args.mask_dir): 
        os.makedirs(args.mask_dir) 
    for m in glob.glob(args.cut_dir_x + '*.png'):   
        name = m.split('/')[1].split('\\')[1].split('.')
        img2 = cv2.imread(args.cut_dir_x + name[0] + '.png')
        wigth = img2.shape[0]
        height = img2.shape[1]
        img = np.zeros((wigth ,height,3),np.uint8)      
        for i in range(wigth):
            for j in range(height):
                if img2[i,j][2] != 0: 
                    img[i,j] = [255,255,255]
                else:
                    img[i,j] = [0,0,0]   
        cv2.imwrite(args.mask_dir + name[0] + '.jpg', img)

def background():
    if not os.path.exists(args.background_dir): 
        os.makedirs(args.background_dir)     
    for m in glob.glob(args.x_test_data_path + '*.jpg'):
        for n in glob.glob(args.mask_dir + '*.jpg'):
    
            name1 = m.split('/')[2].split('\\')[1].split('.')
            name2 = n.split('/')[1].split('\\')[1].split('.')
    
            if name1[0] == name2[0]:
    
                img1 = cv2.imread(args.x_test_data_path + name1[0] + '.jpg') 
                img2 = cv2.imread(args.mask_dir + name2[0] + '.jpg')  
                
                num = 256
                img1 = cv2.resize(img1,(num, num))
                img2 = cv2.resize(img2,(num, num))
                
                img = np.zeros((num,num,3),np.uint8)  
                for i in range(num):
                    for j in range(num):
                        if img2[i,j][2] == 0:
                            img = cv2.add(img1, img2)
                        else:
                            img[i,j] = img2[i,j] 
                
                cv2.imwrite(args.background_dir + name1[0] + '.jpg', img)
