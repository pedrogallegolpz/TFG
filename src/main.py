# from torch.nn.modules.pooling import AdaptiveAvgPool2d
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
from utils import load_model, load_data, train_cam_models, test_cam_models,  informacion_umbral_mascaras
from CAM.utils_cam import plot_grid
import json
import math
from torchvision.transforms import Resize


# Reset CUDA cache
torch.cuda.empty_cache()

path_guardado_modelos = '..\\..\\modelos_entrenados\\'
path_dataset = r'..\..\SICAPv1\224_patch_par'
def run():
    # List of technics to train
    model_classes = ['cam', 'cam_pro']
    technics_pro = ['gradcam', 'gradcampp', 'smoothgradcampp']
    
    models_base = { 'VGG': {'model':models.vgg16(pretrained=True),
                            'input_size': 224},
                    'RESNET': {'model':models.resnet18(pretrained=True),
                               'input_size': 224},
                    'MOBILENET': {'model':models.mobilenet_v2(pretrained=True),
                                  'input_size': 224},
                    'EFFICIENTNET': {'model':torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 
                                                            'nvidia_efficientnet_b0', 
                                                            pretrained=True),
                                  'input_size': 224},
                   }
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('DEVICE: ', device)

    # Read datasets
    dataloaders, dataset_sizes = load_data(path_dataset)
    print('\n\n\n\n')

    
    
    # Entrenamos
    train_cam_models(path_modelos=path_guardado_modelos,
                     path_dataset=path_dataset,
                     epochs=100,
                     early_stopping=10,
                     device = device)
    print('\n\n\n\n')
    
    
    # Testeamos 
    test_cam_models(path_guardado_modelos, path_dataset)
    print('\n\n\n\n')
    

    informacion_umbral_mascaras(path_guardado_modelos, path_dataset, device='cuda')
    

    iter_dataloader = iter(dataloaders['val'])
    for i in range(10):
        # Cogemos la activación de las capas
        x, mask, act_classes=next(iter_dataloader)

        while act_classes[0]!=1:
            x, mask, act_classes=next(iter_dataloader)

        x=x.to(device)
        act_classes=act_classes.to(device)

        for net in models_base.keys():
            plot_grid(x,
                      mask,
                      act_classes,
                      net,
                      n_noise=10,
                      std=0.2,
                      cam=True,
                      gradcam=True,
                      gradcampp=True,
                      smoothgradcampp=True,
                      device=device,
                      path_modelos=path_guardado_modelos)
            
            print('='*50)


run()