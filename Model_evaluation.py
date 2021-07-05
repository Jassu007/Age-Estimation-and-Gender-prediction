# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 22:57:55 2020

@author: jeswanth.gutti
"""

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim import lr_scheduler

import numpy as np
import torchvision
from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt

import time
import os
import copy
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler

age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
Gender = ['F', 'M']
def get_age_model(model_path):
    model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 8) 
    model.load_state_dict(torch.load(model_path))
    
    
def get_Gender_model(model_path):
    model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_path))

def get_age(model_path, test_tensor):
    model = get_age_model(model_path)
    model.eval()
    score = model(test_tensor)
    
    return age_list[score.argmax().item()]

def get_Gender(model_path, test_tensor):
    model = get_Gender_model(model_path)
    model.eval()
    score = model(test_tensor)
    
    return Gender[score.argmax().item()]
    
test_Image_path = ''
model_path=''

input_image = Image.open((test_Image_path)).convert('RGB').resize((256, 256))
test_tensor = transforms.ToTensor()(input_image).unsqueeze_(0)
Pred_age = get_age(model_path, test_tensor)
Pred_gender = get_Gender(model_path, test_tensor)

    




