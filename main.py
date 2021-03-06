# Demo - train the DeepFuse network & use it to generate an image

from __future__ import print_function

import time

from deepfuse_2.train_recons import train_recons
from deepfuse_2.generate import generate
from deepfuse_2.utils import list_images
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#IS_TRAINING = True
IS_TRAINING = False

BATCH_SIZE = 5
EPOCHES = 1

SSIM_WEIGHTS = [1, 10, 100, 1000]
MODEL_SAVE_PATH = 'D:/Deepfuse_youngpoong/result/deepfuse_model_bs2_epoch4_all.ckpt'

# model_pre_path is just a pre-train model and not necessary. It is set as None when you want to train your own model.
# model_pre_path  = 'your own pre-train model'
model_pre_path  = None

def main():

    if IS_TRAINING:
        original_imgs_path = list_images('D:/Deepfuse_youngpoong/dataset_coco/')
        
# for ssim_weight in zip(SSIM_WEIGHTS):
        ssim_weight = 100    
        print('\nBegin to train the network ...\n')
        train_recons(original_imgs_path, MODEL_SAVE_PATH, model_pre_path, ssim_weight, EPOCHES, BATCH_SIZE, debug=True)

        print('\nSuccessfully! Done training...\n')
    else:

        output_save_path = 'outputs'
        print('\nBegin to generate pictures ...\n')
        
        #fusion_type = 'l1'
        fusion_type = 'addition'

        EO_DIR = 'D:/dataset_eoir/EO/'
        IR_DIR = 'D:/dataset_eoir/IR/'

        image_num = [12]

        for index in image_num :
            eo_path = EO_DIR +'EO_'+str(index)+ '.png'
            ir_path = IR_DIR +'IR_'+str(index)+ '.png'
            generate(eo_path, ir_path, MODEL_SAVE_PATH, model_pre_path,  1, index, fusion_type, output_path=output_save_path)
            

if __name__ == '__main__':
    main()

