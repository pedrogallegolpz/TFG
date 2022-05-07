from torch.nn.modules.pooling import AdaptiveAvgPool2d
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.transforms import ToTensor, Lambda
import torchvision.models as models
import torchvision

import torch.optim as optim
from torch.optim import lr_scheduler

from sklearn.metrics import accuracy_score


import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import numpy as np

import os

import time
import os
import copy
import math

import cv2

import seaborn as sns

import abc  # For implementing abstract methods

import CAM.cam
from utils import train_model
import json
import math

# Reset CUDA cache
torch.cuda.empty_cache()


def get_data_transforms():
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224, scale=(0.96, 1.0), ratio=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize([224,224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize([224,224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    return data_transforms

def load_data(path, batch_size=8, split_size=0.2):# Data augmentation and normalization for training
    data_transforms = get_data_transforms()
    
    data_dir = path
    print(os.path.join(data_dir, "traub"))
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'test']}
    
    print(len(image_datasets["train"]))
    dataset_sizes = {
                        "train": int(len(image_datasets["train"])-np.floor(len(image_datasets["train"])*split_size)),
                        "val": int(np.floor(len(image_datasets["train"])*split_size)),
                        "test": len(image_datasets["test"])
                    }

    # Creamos el conjunto de validación
    indices = np.array(range(dataset_sizes['test']))
    split = int(np.floor(split_size * dataset_sizes['test']))

    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    
    # Creating PT data samplers and loaders:
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)
    
    dataloaders = {
                    "train": torch.utils.data.DataLoader(image_datasets["train"], 
                                                         batch_size=batch_size,
                                                         sampler=train_sampler,
                                                         num_workers=2),
                    "val": torch.utils.data.DataLoader(image_datasets["train"], 
                                                         batch_size=batch_size,
                                                         sampler=valid_sampler,
                                                         num_workers=2),
                    "test": torch.utils.data.DataLoader(image_datasets["test"], 
                                                         batch_size=batch_size,
                                                         shuffle=True,
                                                         num_workers=2)      
                    }
    
    
    # Test
    dataset_test = datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms['test'])
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, 
                                                  shuffle=True, num_workers=1)
    
    assert(image_datasets['train'].classes==image_datasets['train'].classes)
    
    class_names = image_datasets['train'].classes
    print("Clases: ", class_names) 
    print(f'Train image size: {dataset_sizes["train"]}')
    print(f'Validation image size: {dataset_sizes["val"]}')
    print(f'Test image size: {dataset_sizes["test"]}')
    
    return dataloaders, dataset_sizes, dataloader_test



dataloaders, dataset_sizes, dataloader_test = load_data(r'..\..\SICAPv1\299_patch_impar')

model_vgg = models.vgg16(pretrained=True)

model_cam = CAM.cam.CAM_model(model_vgg, D_out=2)
model_gradcam = CAM.cam.GradCAM_model(model_vgg, D_out=2)
model_gradcampp = CAM.cam.GradCAMpp_model(model_vgg, D_out=2)
model_smoothgradcampp = CAM.cam.SmoothGradCAMpp_model(model_vgg, D_out=2)

# To device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Usando: ", device)

try:
    model_cam = torch.load("./model_cam.pth").to(device)
    with open('dic_best_values_model_cam.json') as f_dic:
        dic_best_values_model_cam = json.load(f_dic)
    print("model_cam loaded")
except:
    model_cam = model_cam.to(device)
    dic_best_values_model_cam = {'loss': math.inf, 'acc':0.}
    print("model_cam not found")



try:
    model_gradcam = torch.load("./model_gradcam.pth").to(device)
    with open('dic_best_values_model_gradcam.json') as f_dic:
        dic_best_values_model_gradcam = json.load(f_dic)
    print("model_gradcam loaded")
except:
    model_gradcam = model_gradcam.to(device)
    dic_best_values_model_gradcam = {'loss': math.inf, 'acc':0.}
    print("model_gradcam not found")
    
try:
    model_gradcampp = torch.load("./model_gradcampp.pth").to(device)
    with open('dic_best_values_model_gradcampp.json') as f_dic:
        dic_best_values_model_gradcampp = json.load(f_dic)
    print("model_gradcampp loaded")
except:
    model_gradcampp = model_gradcampp.to(device)
    dic_best_values_model_gradcampp = {'loss': math.inf, 'acc':0.}
    print("model_gradcampp not found")

    
try:
    model_smoothgradcampp = torch.load("./model_smoothgradcampp.pth").to(device)
    with open('dic_best_values_model_smoothgradcampp.json') as f_dic:
        dic_best_values_model_smoothgradcampp = json.load(f_dic)
    print("model_smoothgradcampp loaded")
except:
    model_smoothgradcampp = model_smoothgradcampp.to(device)
    dic_best_values_model_smoothgradcampp = {'loss': math.inf, 'acc':0.}
    print("model_smoothgradcampp not found")

# Optimizer
optimizer_cam = optim.SGD(model_cam.parameters(), lr=0.001, momentum=0.9)
optimizer_gradcam = optim.SGD(model_gradcam.parameters(), lr=0.001, momentum=0.9)
optimizer_gradcampp = optim.SGD(model_gradcampp.parameters(), lr=0.001, momentum=0.9)
optimizer_smoothgradcampp = optim.SGD(model_smoothgradcampp.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler_cam = lr_scheduler.StepLR(optimizer_cam, step_size=7, gamma=0.1)
exp_lr_scheduler_gradcam = lr_scheduler.StepLR(optimizer_gradcam, step_size=7, gamma=0.1)
exp_lr_scheduler_gradcampp = lr_scheduler.StepLR(optimizer_gradcampp, step_size=7, gamma=0.1)
exp_lr_scheduler_smoothgradcampp = lr_scheduler.StepLR(optimizer_smoothgradcampp, step_size=7, gamma=0.1)

criterion = torch.nn.CrossEntropyLoss()


#####################
model_cam, best_val_loss, best_val_acc = train_model(model_cam,
                                                      dic_best_values_model_cam,
                                                      dataloaders, 
                                                      dataset_sizes,
                                                      criterion,
                                                      optimizer_cam,
                                                      exp_lr_scheduler_cam,
                                                      num_epochs = 3)

torch.save(model_cam, "./model_cam.pth")
dic_best_values_model_cam = {'loss': best_val_loss, 'acc': best_val_acc}
with open('./dic_best_values_model_cam.json','w') as f_dic:
    json.dump(dic_best_values_model_cam, f_dic)


#####################

model_gradcam, best_val_loss, best_val_acc = train_model(model_gradcam,
                                                      dic_best_values_model_gradcam,
                                                      criterion,
                                                      optimizer_gradcam,
                                                      exp_lr_scheduler_gradcam,
                                                      num_epochs = 3)

torch.save(model_gradcam, "./model_gradcam.pth")
dic_best_values_model_gradcam = {'loss': best_val_loss, 'acc': best_val_acc}
with open('./dic_best_values_model_gradcam.json','w') as f_dic:
    json.dump(dic_best_values_model_gradcam, f_dic)
    
#####################

torch.autograd.set_detect_anomaly(True)
model_gradcampp, best_val_loss, best_val_acc = train_model(model_gradcampp,
                                                      dic_best_values_model_gradcampp,
                                                      criterion,
                                                      optimizer_gradcampp,
                                                      exp_lr_scheduler_gradcampp,
                                                      num_epochs = 3)

torch.save(model_gradcampp, "./model_gradcampp.pth")
dic_best_values_model_gradcampp = {'loss': best_val_loss, 'acc': best_val_acc}
with open('./dic_best_values_model_gradcampp.json','w') as f_dic:
    json.dump(dic_best_values_model_gradcampp, f_dic)

#####################

torch.autograd.set_detect_anomaly(True)
model_smoothgradcampp, best_val_loss, best_val_acc = train_model(model_smoothgradcampp,
                                                      dic_best_values_model_smoothgradcampp,
                                                      criterion,
                                                      optimizer_smoothgradcampp,
                                                      exp_lr_scheduler_smoothgradcampp,
                                                      num_epochs = 3)

torch.save(model_smoothgradcampp, "./model_smoothgradcampp.pth")
dic_best_values_model_smoothgradcampp = {'loss': best_val_loss, 'acc': best_val_acc}
with open('./dic_best_values_model_smoothgradcampp.json','w') as f_dic:
    json.dump(dic_best_values_model_smoothgradcampp, f_dic)

