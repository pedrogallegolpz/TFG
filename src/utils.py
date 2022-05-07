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
from torchvision.transforms import Resize
# Reset CUDA cache
torch.cuda.empty_cache()

path_guardado_modelos = 'modelos/'
os.makedirs(path_guardado_modelos, exist_ok=True)

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
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'test']}
    #print(image_datasets["train"].imgs)

    dataset_sizes = {
                        "train": int(len(image_datasets["train"])-np.floor(len(image_datasets["train"])*split_size)),
                        "val": int(np.floor(len(image_datasets["train"])*split_size)),
                        "test": len(image_datasets["test"])
                    }

    # Creamos el conjunto de validación
    indices = np.array(range(dataset_sizes['train']+dataset_sizes['val']))

    np.random.shuffle(indices)
    train_indices, val_indices = indices[dataset_sizes['val']:], indices[:dataset_sizes['val']]
    
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
    
    assert(image_datasets['train'].classes==image_datasets['test'].classes)
    
    class_names = image_datasets['train'].classes
    print("Clases: ", class_names) 
    print(f'Train image size: {dataset_sizes["train"]}')
    print(f'Validation image size: {dataset_sizes["val"]}')
    print(f'Test image size: {dataset_sizes["test"]}')
    
    return dataloaders, dataset_sizes, dataloader_test


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('DEVICE: ', device)
dataloaders, dataset_sizes, dataloader_test = load_data(r'..\..\SICAPv1\299_patch_impar')

model_vgg = models.vgg16(pretrained=True)

models = {
            'cam': {
                        'model':CAM.cam.CAM_model(model_vgg, D_out=2),
                        'best_values': {'loss': math.inf, 'acc':0.}
                    },
            'gradcam':{
                        'model':CAM.cam.GradCAM_model(model_vgg, D_out=2),
                        'best_values': {'loss': math.inf, 'acc':0.}
                    },
            'gradcampp':{
                        'model':CAM.cam.GradCAMpp_model(model_vgg, D_out=2),
                        'best_values': {'loss': math.inf, 'acc':0.}
                    },
            'smoothgradcampp':{
                        'model':CAM.cam.SmoothGradCAMpp_model(model_vgg, D_out=2),
                        'best_values': {'loss': math.inf, 'acc':0.}
                    },
}


for name in models.keys():
    print('\n', name.upper())
    # Load model
    try:
        models[f'{name}']['model'] = torch.load(path_guardado_modelos+f"model_{name}.pth").to(device)
        with open(path_guardado_modelos+f'dic_best_values_model_{name}.json') as f_dic:
             models[f'{name}']['best_values'] = json.load(f_dic)
        print(f"model_{name} loaded")
    except:
        print(f"model_{name} not found")
    
    # Cogemos el optimizador y el criterio de aprendizaje
    optimizer = optim.SGD(models[f'{name}']['model'].parameters(), lr=0.001, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss() 

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Entrenamos
    model_gradcam, best_val_loss, best_val_acc = train_model(models[f'{name}']['model'],
                                                          models[f'{name}']['best_values'],
                                                          dataloaders, 
                                                          dataset_sizes,
                                                          criterion,
                                                          optimizer,
                                                          exp_lr_scheduler,
                                                          num_epochs = 0)

    # Guardamos el mejor modelo del entrenamiento
    torch.save(models[f'{name}']['model'], path_guardado_modelos+f"model_{name}.pth")
    
    models[f'{name}']['best_values'] = {'loss': best_val_loss, 'acc': best_val_acc.item()}
    with open(path_guardado_modelos+f'dic_best_values_model_{name}.json','w') as f_dic:
        json.dump(models[f'{name}']['best_values'], f_dic)
    
    print(f'model_{name} guardado')
