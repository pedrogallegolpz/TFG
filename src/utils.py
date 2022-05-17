import torch
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, transforms
import time
import copy

import matplotlib.pyplot as plt

import os
import json
import math
import numpy as np


from torch.optim import lr_scheduler

import CAM.cam
from dataset import ImageFolder_and_MaskFolder

import cv2
from numba import cuda 

def get_data_transforms():
    # Just normalize for validation
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
    image_datasets = {x: ImageFolder_and_MaskFolder(os.path.join(data_dir, x),
                                                             data_transforms[x]
                                                    )
                      for x in ['train', 'test']}

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
                                                         batch_size=1,
                                                         shuffle=True,
                                                         num_workers=2)      
                    }

    assert(image_datasets['train'].classes==image_datasets['test'].classes)
    
    class_names = image_datasets['train'].classes
    print("Clases: ", class_names) 
    print(f'Train image size: {dataset_sizes["train"]}')
    print(f'Validation image size: {dataset_sizes["val"]}')
    print(f'Test image size: {dataset_sizes["test"]}')
    print('Dataset loaded.\n')
    
    return dataloaders, dataset_sizes


def load_model(path_modelos, name, device, original_model=None):
    name = name.lower()
    model = {}
    # Load model
    try:
        model['model'] = torch.load(path_modelos+f"model_{name}.pth").to(device)
        with open(path_modelos+f'dic_best_values_model_{name}.json') as f_dic:
             model['best_values'] = json.load(f_dic)
        print(f"model_{name} loaded")
    except:
        if name=='cam':
            model['model'] = CAM.cam.CAM_model(original_model, D_out=2)
        else:
            model['model'] = CAM.cam.CAM(original_model, D_out=2)
       
        model['best_values'] = {'loss': math.inf, 'acc':0.}
        print(f"model_{name} not found") 
        
        
    model['model'] = model['model'].to(device)
    
    return model


############################################################################
# ENTRENAMIENTO
############################################################################
def train_model(model, dic_best_values, dataloaders, dataset_sizes,criterion, optimizer, scheduler, num_epochs=2, device='cuda'):
    since = time.time()

    # Saving actual weights as best weights
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = dic_best_values['loss']
    best_acc = dic_best_values['acc']

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i, (inputs, _, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                if i % 200 == 199:
                    print('[%d, %d] loss: %.3f' %(epoch + 1, i, running_loss / (i * inputs.size(0))))

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # statistics
                    running_loss += loss.item()
                    running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'train':
                scheduler.step()
            elif phase == 'val' and epoch_loss < best_loss:
                # deep copy the model
                print('New best model found!')
                print(f'New record loss: {epoch_loss}, previous record loss: {best_loss}')
                best_loss = epoch_loss
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:.4f} Best val loss: {:.4f}'.format(best_acc, best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, best_loss, best_acc


def train_cam_models(path_modelos, cam=True, cam_pro=True, epochs=1, learning_rate=0.001, momentum=0.9):
    """
    Parameters
    ----------
    path_modelos : str
        path where the models will be stored.
    cam : bool, optional
        If True, this model will be trained. The default is True.
    cam_pro : bool, optional
        If True, this model will be trained. The default is True.
    epochs : int, optional
        num of epochs for training. The default is 1.
    learning_rate : float, optional
        hyperparameter. The default is 0.001.
    momentum : float, optional
        hyperparameter. The default is 0.9.


    DESCRIPTION
    ------------
        Run the train phase for CAM models. It's enough to call this function
        as:
            - utils.train_cam_models('.')
            - utils.train_cam_models('models/')

    """
    if path_modelos[-1] != '/':
        path_modelos += '/'
        
    # List of technics to train
    technics = {'cam': cam, 'cam_pro': cam_pro}

    os.makedirs(path_modelos, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('DEVICE: ', device)
    
    # Read datasets
    dataloaders, dataset_sizes = load_data(r'..\..\SICAPv1\299_patch_impar')
    
    model_vgg = models.vgg16(pretrained=True)
    
    models_dic = {}
    for name in technics.keys():
        if not technics[f'{name}']:
            continue
        
        print('\n', name.upper())
        
        # Load model
        models_dic[f'{name}'] = load_model(path_modelos, name, device, original_model=model_vgg)
        
        # Cogemos el optimizador y el criterio de aprendizaje
        optimizer = optim.SGD(models_dic[f'{name}']['model'].parameters(), lr=learning_rate, momentum=momentum)
        criterion = torch.nn.CrossEntropyLoss() 
    
        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
        # Entrenamos
        model_gradcam, best_val_loss, best_val_acc = train_model(models_dic[f'{name}']['model'],
                                                              models_dic[f'{name}']['best_values'],
                                                              dataloaders, 
                                                              dataset_sizes,
                                                              criterion,
                                                              optimizer,
                                                              exp_lr_scheduler,
                                                              num_epochs = epochs)
    
        # Guardamos el mejor modelo del entrenamiento
        torch.save(models_dic[f'{name}']['model'], path_modelos+f"model_{name}.pth")
        
        
        

        
        models_dic[f'{name}']['best_values'] = {'loss': best_val_loss, 'acc': best_val_acc.item()}
        with open(path_modelos+f'dic_best_values_model_{name}.json','w') as f_dic:
            json.dump(models_dic[f'{name}']['best_values'], f_dic)
        
        print(f'model_{name} guardado')
        
        
############################################################################
# TEST
############################################################################
def test_model(model, dataloader_test, test_size, criterion, device='cuda'):
    since = time.time()

    model.eval()   # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    for i, (inputs, _, labels) in enumerate(dataloader_test):
        inputs = inputs.to(device)
        labels = labels.to(device)

        if i % 200 == 199:
            print('[%d] loss: %.3f' %( i, running_loss / (i * inputs.size(0))))

        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # statistics
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data)

    final_loss = running_loss / test_size
    final_acc = running_corrects.double() / test_size
    print('TEST\n','Loss: {:.4f} Acc: {:.4f}'.format(final_loss, final_acc))

    
    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return final_loss, final_acc


def test_cam_models(path_modelos, cam=True, cam_pro=True):
    # List of technics to train
    technics = {'cam': cam, 'cam_pro': cam_pro}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('DEVICE: ', device)

    # Read datasets
    dataloaders, dataset_sizes = load_data(r'..\..\SICAPv1\299_patch_impar')

    models_dic = {}
    for name in technics.keys():
        if not technics[f'{name}']:
            continue
        
        # Load model
        models_dic[f'{name}'] = load_model(path_modelos, name, device)

        criterion = torch.nn.CrossEntropyLoss() 

        test_model(models_dic[f'{name}']['model'] , dataloaders['test'], dataset_sizes['test'], criterion, device)
        
        
        
############################################################################
# UMBRAL PARA LA MÁSCARA
############################################################################
def prepare_mask(mask):
    try:
        mask.cpu()
    except:
        pass
    
    if len(mask.shape)!=2:
        if mask.shape[0]==3:
            mask = mask.permute(1,2,0)
        
        mask = mask.numpy()
        if mask.shape[2]==3:
            mask = cv2.cvtColor(mask,cv2.COLOR_RGB2GRAY)
        
        # Reescalamos
        mask = cv2.resize(mask, dsize=(299, 299), interpolation=cv2.INTER_CUBIC)
    
    mask[mask>0.]=1.
    mask[mask<0.]=0.
    
    return mask

def iou(mask1, mask2):
    mask1 = prepare_mask(mask1)
    mask2 = prepare_mask(mask2)
    
    iou = np.logical_and(mask1, mask2).sum() / np.logical_or(mask1, mask2).sum()
    
    return iou
    
    
    
def generate_mask_from_heatmap(heatmap, umbral):
    # Normalizamos
    heatmap /= heatmap.max()
    
    # Pasamos a np array
    heatmap = heatmap.cpu().detach().numpy()
    
    # Reescalamos
    heatmap_aumentado = cv2.resize(heatmap, dsize=(299, 299), interpolation=cv2.INTER_CUBIC)
    
    # Creamos la máscara
    mask = np.zeros_like(heatmap_aumentado)
    mask[heatmap_aumentado>umbral] = 1.
    
    return mask
    
    
 
    

def curvas_umbral_mascara(model, technique, dataloader_val, device='cuda'):
    model.eval()   # Set model to evaluate mode

    # Iterate over data.
    iou_tabla = []
    for i, (inputs, masks, labels) in enumerate(dataloader_val):
        for inp, mask, label in zip(inputs, masks,labels):
            if label.item()!=1:
                continue
        
            inp = inp.to(device)[None,:]
            label=label.to(device)
    
    
            heatmaps = model.saliency_map(inp,
                                         technique,
                                         n_noise=20,
                                         std=0.2,
                                         device=device)

            
            iou_fila = []
            for umbral in np.linspace(0,1,21):
                mask_generated = generate_mask_from_heatmap(heatmaps[label.item()], umbral)
                iou_fila.append(iou(mask, mask_generated))
            
            iou_tabla.append(iou_fila)
            
            
            if i % 40 == 39:
                print(f'[{i}]')
     
                
    iou_tabla = np.array(iou_tabla)
    valores_iou_medios = iou_tabla.mean(axis=0)

    print(f"[{technique}] Mejor valor: ", valores_iou_medios.max(), " con el umbral: ", np.argmax(valores_iou_medios)*0.05)
    
    # Área bajo la curva
    area = 0
    for i in range(len(valores_iou_medios)-1):
        minimo = min(valores_iou_medios[i],valores_iou_medios[i+1])
        maximo = max(valores_iou_medios[i],valores_iou_medios[i+1])
        area += minimo*0.05 + (maximo-minimo)*0.0025            # Es como sumar el cuadrado y luego el triángulo que queda arriba
        
    print(f"[{technique}] Área bajo la curva: ", area)
    
    # Plot de la curva
    plt.plot(np.linspace(0,1,21), valores_iou_medios, 'o')
    plt.title(f"[{technique}]")
    plt.show()
    
    
    return valores_iou_medios, area
    
                
                
      
def informacion_umbral_mascaras(path_modelos, cam=True, cam_pro=True, device=None):
    # List of technics to train
    model_classes = {'cam': cam, 'cam_pro': cam_pro}
    technics = ['cam','gradcam', 'gradcampp', 'smoothgradcampp']

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('DEVICE: ', device)

    # Read datasets
    dataloaders, dataset_sizes = load_data(r'..\..\SICAPv1\299_patch_impar')

    models_dic = {}
    iou_tecnicas = np.array([])
    areas_bajo_iou = np.array([])
    for name in model_classes.keys():
        if not model_classes[f'{name}']:
            continue
        
        # Load model
        models_dic[f'{name}'] = load_model(path_modelos, name, device)
        
        if 'cam' != name:
            for technique in technics[1:]:
                iou_medios, area = curvas_umbral_mascara(models_dic[f'{name}']['model'], 
                                                    technique, 
                                                    dataloaders['val'], 
                                                    device=device)
        else:
            iou_medios, area = curvas_umbral_mascara(models_dic[f'{name}']['model'], 
                                                  None, 
                                                  dataloaders['val'], 
                                                  device=device)     
            
            
        np.append(iou_tecnicas, iou_medios)
        np.append(areas_bajo_iou, area)
        
    print("La mejor técnica ha sido: ", technics[np.argmax(areas_bajo_iou)], " con un área de: ", areas_bajo_iou.max())
    
    
        
        
        
