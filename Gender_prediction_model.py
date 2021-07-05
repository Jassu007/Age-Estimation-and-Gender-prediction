# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 22:43:35 2020

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

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class GenderEstimationModel:
    def __init__(self):
        self.data_dir='Gender_folder/'
        self.image_datasets= None
        self.dataloaders = None
        self.data_tranforms = None
        self.valid_size=0.2
        
    def set_data_transforms(self):
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.RandomRotation(degrees=(-30, 30)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }
        
    def set_image_datasets(self):
        self.image_datasets={x: datasets.ImageFolder(self.data_dir,self.data_transforms[x])
                             for x in ['train', 'val']}
    
    def set_data_loaders(self):
        num_train = len(self.image_datasets['val']) #in image datasets both train and valid are same but in this function we willsplit it
        indices = list(range(num_train))
        split = int(np.floor(self.valid_size * num_train))
        np.random.shuffle(indices)
        train_idx, test_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(test_idx)
        dataloaders = {}
        dataloaders['train'] = torch.utils.data.DataLoader(self.image_datasets['train'], sampler=train_sampler,batch_size=30, num_workers=4)
        dataloaders['val'] = torch.utils.data.DataLoader(self.image_datasets['val'], sampler=test_sampler,batch_size=30, num_workers=4)
        self.dataloaders = dataloaders
        
    def start_data_processing(self):
        self.set_data_transforms()
        self.set_image_datasets()
        self.set_data_loaders()
        
        
        
    def train_model(self, model, criterion, optimizer, scheduler, num_epochs=50):
        since = time.time()
        writer = SummaryWriter()
    
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        epoch_losses_ = []
        epoch_accuracies_ =  [] 
    
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs-1))
            print('-'*10)
        
            for phase in ['train', 'val']:
                
                if phase == 'train':
                    model.train()
                else:
                    model.eval()
                
                running_loss = 0.0
                running_corrects =0
                total_images=1
            
                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                
                    optimizer.zero_grad()
                
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                    
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    running_loss += loss.item() *inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data).item()
                    total_images +=inputs.size(0)
                
                if phase == 'train':
                    scheduler.step()
                
                epoch_loss = running_loss/total_images
                epoch_acc = running_corrects/total_images
            
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                #writer.add_scalar('Loss/'+phase, epoch_loss, epoch)
                #writer.add_scalar('Accuracy/'+phase, epoch_acc, epoch)
                epoch_losses_.append(epoch_loss)
                epoch_accuracies_.append(epoch_acc)
                print('total images', total_images)
            
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
            print()
        
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model, epoch_losses_, epoch_accuracies_
    
if __name__=='__main__':
    gender = GenderEstimationModel()
    gender.start_data_processing()
    model_ft = models.resnet50(pretrained = True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    model_ft, epoch_losses_, epoch_accuracies_ = gender.train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=30)
    
                
    