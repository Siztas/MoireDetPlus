# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 22:00
# @Author  : zhoujun

from __future__ import print_function
import sys
sys.path.append(r"/public/home/lijinjin/下载/MoireDet++_train/MoireDet-main/MoireDet")


import os
from lib.utils import load_json
from lib.models import get_model
from lib.data_loader import get_dataloader
import math
import torch
import cv2
import numpy as np
from collections import Counter
import json
from tqdm import tqdm
import time


config = load_json('./predict_configs/predict_final_imgs_420.yaml')

base_dir = config['data_loader']['args']['dataset']['base_dir']
output_dir = config['out_path']
is_train = False

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main(config):


    model = get_model(config)
    if config['checkpoint'].startswith('hdfs'):
        checkpoint_name = os.path.basename(config['checkpoint'])
        # os.popen("sh down_hdfs.sh {}".format(config['checkpoint']))


        checkpoint = torch.load(checkpoint_name)
        # os.system('rm {}'.format(checkpoint_name))
    else:
        checkpoint = torch.load(config['checkpoint'])




    # model = get_model(config)
    # checkpoint = torch.load(config['checkpoint'])


    output_dir = config['out_path']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model.eval()



    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        for key in list(checkpoint["state_dict"].keys()):
            new_key = key.replace('module.', '')
            checkpoint["state_dict"][new_key] = checkpoint["state_dict"].pop(key)
        model.load_state_dict(checkpoint["state_dict"])

    device = torch.device("cuda:0")

    model = model.to(device)

    train_loader = get_dataloader('',config['data_loader']['args'], is_transform= False)
    img_index = -1
    with torch.no_grad():
        start = time.time()
        for i, (imgs,imgs_bak,moires, index_list) in tqdm(enumerate(train_loader)):
            # if i >= 100:
            #     break
            imgs = imgs.to(device)
            pred_moires = model(imgs)
            try:
                attentions = pred_moires[0][1]
            except:
                attentions = pred_moires[0][0]
            pred_moires = pred_moires[0][0]



            for pred_moire,moire,index,img_bak,attention in zip(pred_moires,moires,index_list,imgs_bak,attentions):
                img_index += 1
                moire = moire.permute(1,2,0)
                img_bak = img_bak.numpy()

                attention = attention.permute(1,2,0).cpu().numpy()
                attention = np.uint8(attention*255)
                attention = cv2.cvtColor(attention, cv2.COLOR_GRAY2BGR)

                pred_moire = pred_moire.permute(1,2,0).cpu().numpy()
                pred_moire = 255 - np.clip(pred_moire,0,255).astype(np.uint8)
                pred_moire = cv2.cvtColor(pred_moire,cv2.COLOR_GRAY2BGR)
                moire = cv2.cvtColor(np.uint8(moire),cv2.COLOR_GRAY2BGR)

                #combined_moire = np.concatenate([255 - moire,pred_moire,img_bak],axis=0)

                combined_moire = np.concatenate([pred_moire,img_bak],axis=0)
                
                img_path = train_loader.dataset.img_file[index]
                img_path = img_path.replace(base_dir,output_dir)

                if not os.path.exists(os.path.dirname(img_path)):
                    os.makedirs(os.path.dirname(img_path))

                img_name = '{}.jpg'.format(str(img_index).zfill(5))
                cv2.imwrite(img_path,pred_moire)

                # img_name = '{}_pred_moire.jpg'.format(str(img_index).zfill(5))
                # cv2.imwrite(os.path.join(output_dir,img_name),pred_moire)
                #
                # img_name = '{}_ori_moire.jpg'.format(str(img_index).zfill(5))
                # cv2.imwrite(os.path.join(output_dir, img_name), img_bak)
        end = time.time()
        print(end-start)

if __name__ == '__main__':
    main(config)
