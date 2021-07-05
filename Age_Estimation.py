# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 22:52:57 2020

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
import torch.nn.functional as F
import time
import os
import copy

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
plt.ion()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class AgeEstimationModel:
    def __init__(self):
        self.data_dir='Image_data/'
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
        temp_opt=optimizer
        since = time.time()
        writer = SummaryWriter()
    
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        epoch_losses_ = []
        epoch_accuracies_ =  [] 
    
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs-1))
            print('-'*10)
            
            train_not_require_layer =0
            if epoch == 0:
                new_layer_to_freeze = 9
            elif epoch == 1:
                new_layer_to_freeze = 6
            else:
                new_layer_to_freeze = 0
            if train_not_require_layer!=new_layer_to_freeze:
                for i, child in enumerate(model_ft.children()):
                    grad_require = i>=new_layer_to_freeze
                    for param in child.parameters():
                        param.requires_grad = grad_require
                optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01, momentum=0.9)
            else:
                optimizer = temp_opt
                
        
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
                        ##
                        #outputs = F.softmax(outputs, dim=1)
                        #outputs=torch.log(outputs)
                        ##
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
    
class FocalLoss(nn.CrossEntropyLoss):
    ''' Focal loss for classification tasks on imbalanced datasets '''
    def __init__(self, gamma=2, alpha=None, ignore_index=-100, reduction='none'):
        super().__init__(weight=alpha, ignore_index=ignore_index, reduction='none')
        self.reduction = reduction
        self.gamma = gamma
    def forward(self, input_, target):
        cross_entropy = super().forward(input_, target)
        # Temporarily mask out ignore index to '0' for valid gather-indices input.
        # This won't contribute final loss as the cross_entropy contribution
        # for these would be zero.
        target = target * (target != self.ignore_index).long()
        input_prob = torch.gather(F.softmax(input_, 1), 1, target.unsqueeze(1))
        loss = torch.pow(1 - input_prob, self.gamma) * cross_entropy
        return torch.mean(loss)
    
if __name__=='__main__':
    age_task = AgeEstimationModel()
    age_task.start_data_processing()
    model_ft = models.resnet50(pretrained = True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 8) 
    model_ft = model_ft.to(device)
    criterion = FocalLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)
    model_ft, epoch_losses_, epoch_accuracies_ = age_task.train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=30)
    
    
    