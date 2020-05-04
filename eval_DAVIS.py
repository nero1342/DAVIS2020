from __future__ import division
import torch
from torch.autograd import Variable
from torch.utils import data

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models

# general libs
import yaml
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import time
import tqdm
import os
import argparse
import copy


### My libs
import yaml
import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch.utils import data
from torchnet import meter

from torch.utils.data import DataLoader 
import argparse
import pprint

from workers.trainer import Trainer
from utils.random_seed import set_seed, set_determinism
from utils.getter import get_instance, get_data
from datasets.DAVIS import DAVIS
from models.stm import STM
from utils.helper import db_eval_iou, db_eval_boundary

torch.set_grad_enabled(False) # Volatile

def get_arguments():
    parser = argparse.ArgumentParser(description="SST")
    parser.add_argument("-s", type=str, help="set", required=True)
    parser.add_argument("-y", type=int, help="year", required=True)
    parser.add_argument("-viz", help="Save visualization", action="store_true")
    parser.add_argument("-D", type=str, help="path to data",default='/local/DATA')
    parser.add_argument('--config')
    parser.add_argument('--gpus', default=None)
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()

args = get_arguments()

YEAR = args.y
SET = args.s
VIZ = args.viz
DATA_ROOT = args.D
config_path = args.config
config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
config['gpus'] = args.gpus
config['debug'] = args.debug

set_determinism()

# Get device
dev_id = 'cuda:{}'.format(config['gpus']) \
    if torch.cuda.is_available() and config.get('gpus', None) is not None \
    else 'cpu'
device = torch.device(dev_id)

# Get pretrained model
pretrained_path = config["pretrained"]

pretrained = None
if (pretrained_path != None):
    pretrained = torch.load(pretrained_path, map_location=dev_id)
    for item in ["model"]:
        config[item] = pretrained["config"][item]
# 1: Load datasets
test_dataset = get_instance(config['dataset']['test'])
test_dataloader = DataLoader(test_dataset,
                        **config['dataset']['test']['loader']
)
# 2: Define network
set_seed(config['seed'])
model = get_instance(config['model']).to(device)
model = nn.DataParallel(model)
if torch.cuda.is_available():
    model.cuda()
model.eval() # turn-off BN
# Train from pretrained if it is not None
if pretrained is not None:
    model.load_state_dict(pretrained['model_state_dict'])
#pth_path = 'STM_weights.pth'
#print('Loading weights:', pth_path)
#model.load_state_dict(torch.load(pth_path))

# 3: Define loss
set_seed(config['seed'])
criterion = get_instance(config['loss']).to(device)

# # 4: Define Optimizer
# set_seed(config['seed'])
# optimizer = get_instance(config['optimizer'],
#                             params=model.parameters())

# if pretrained is not None:
#     optimizer.load_state_dict(pretrained['optimizer_state_dict'])

# 6: Define metrics
set_seed(config['seed'])
metric = {mcfg['name']: get_instance(mcfg)
            for mcfg in config['metric']}


# Model and version
MODEL = 'STM'
print(MODEL, ': Testing on DAVIS')

if torch.cuda.is_available():
    print('using Cuda devices, num:', torch.cuda.device_count())

if VIZ:
    print('--- Produce mask overaid video outputs. Evaluation will run slow.')
    print('--- Require FFMPEG for encoding, Check folder ./viz')


palette = Image.open(DATA_ROOT + '/Annotations/480p/blackswan/00000.png').getpalette()

def Run_video(Fs, Ms, num_frames, num_objects, Mem_every=None, Mem_number=None):
    # initialize storage tensors
    if Mem_every:
        to_memorize = [int(i) for i in np.arange(0, num_frames, step=Mem_every)]
    elif Mem_number:
        to_memorize = [int(round(i)) for i in np.linspace(0, num_frames, num=Mem_number+2)[:-1]]
    else:
        raise NotImplementedError

    Es = torch.zeros_like(Ms)
    Es[:,:,0] = Ms[:,:,0]
    
    for t in tqdm.tqdm(range(1, num_frames)):
        # memorize
        with torch.no_grad():
            prev_key, prev_value = model(Fs[:,:,t-1], Es[:,:,t-1], torch.tensor([num_objects])) 

        if t-1 == 0: # 
            this_keys, this_values = prev_key, prev_value # only prev memory
        else:
            this_keys = torch.cat([keys, prev_key], dim=3)
            this_values = torch.cat([values, prev_value], dim=3)
        
        # segment
        with torch.no_grad():
            logit = model(Fs[:,:,t], this_keys, this_values, torch.tensor([num_objects]))
        Es[:,:,t] = F.softmax(logit, dim=1)
        
        # update
        if t-1 in to_memorize:
            keys, values = this_keys, this_values
        
    pred = np.argmax(Es[0].cpu().numpy(), axis=0).astype(np.uint8)
    return pred, Es


code_name = '{}_DAVIS_{}{}'.format(MODEL,YEAR,SET)
print('Start Testing:', code_name)

import pandas as pd 
summary = pd.DataFrame() 
name_list = []
j_list = []
f_list = []
for seq, V in enumerate(test_dataloader):
    Fs, Ms, num_objects, info = V
    seq_name = info['name'][0]
    num_frames = info['num_frames'][0].item()
    print('[{}]: num_frames: {}, num_objects: {}'.format(seq_name, num_frames, num_objects[0][0]))
    
    pred, Es = Run_video(Fs, Ms, num_frames, num_objects, Mem_every=5, Mem_number=None)
        
    # Save results for quantitative eval ######################
    test_path = os.path.join('./test', code_name, seq_name)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    for f in range(num_frames):
        img_E = Image.fromarray(pred[f])
        img_E.putpalette(palette)
        img_E.save(os.path.join(test_path, '{:05d}.png'.format(f)))

    if VIZ:
        from helpers import overlay_davis
        # visualize results #######################
        viz_path = os.path.join('./viz/', code_name, seq_name)
        if not os.path.exists(viz_path):
            os.makedirs(viz_path)

        j_score = np.zeros((num_frames, num_objects)) 
        f_score = np.zeros((num_frames, num_objects))
        for f in tqdm.tqdm(range(num_frames)):
            gt = np.array(Image.open(os.path.join(Testset.mask480_dir, seq_name, str(f).zfill(5) + '.png')).convert("P"))
            pF = (Fs[0,:,f].permute(1,2,0).numpy() * 255.).astype(np.uint8)
            pE = pred[f]
            canvas = overlay_davis(pF, pE, palette)
            canvas = Image.fromarray(canvas)
            canvas.save(os.path.join(viz_path, 'f{:05d}.jpg'.format(f)))
            #print(f, pE.shape, gt.shape)
            for i in range(num_objects):
                target = (gt == (i+1)) 
                pre = (pE == (i + 1))
                j_score[f,i] = db_eval_iou(target, pre)
                f_score[f,i] = db_eval_boundary(target, pre)

        for i, x in enumerate(np.average(j_score, axis = 0)):
            name_list.append(seq_name + '_' + str(i + 1))
            j_list.append(x)
        for x in np.average(f_score, axis = 0):
            f_list.append(x)
        print('IoU:{}\nF:{}'.format(np.average(j_score, axis = 0), np.average(f_score, axis = 0)))
        vid_path = os.path.join('./viz/', code_name, '{}.mp4'.format(seq_name))
        frame_path = os.path.join('./viz/', code_name, seq_name, 'f%05d.jpg')
        os.system('ffmpeg -framerate 10 -i {} {} -vcodec libx264 -crf 10  -pix_fmt yuv420p  -nostats -loglevel 0 -y'.format(frame_path, vid_path))
summary['seq_name'] = name_list
summary['iou'] = j_list
summary['f'] = f_list
summary.to_csv('summary.csv', index = False)


