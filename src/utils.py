import torch
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, transforms
import time
import copy

import os
import json
import math
import numpy as np


from torch.optim import lr_scheduler

import CAM.cam

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
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
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
                    running_loss += loss.item() * inputs.size(0)
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
    assert(name=='cam' or name=='gradcam' or name=='gradcampp' or name=='smoothgradcampp')
    
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
        elif name =='gradcam':
            model['model'] = CAM.cam.GradCAM_model(original_model, D_out=2)
        elif name =='gradcampp':
            model['model'] = CAM.cam.GradCAMpp_model(original_model, D_out=2)
        elif name =='smoothgradcampp':
            model['model'] = CAM.cam.SmoothGradCAMpp_model(original_model, D_out=2)
        else:
            print("\nError creating a new model. Maybe original_model==None\n")            
            raise
          
        model['best_values'] = {'loss': math.inf, 'acc':0.}
        print(f"model_{name} not found") 
        
    return model



def train_cam_models(path_modelos, cam=True, gradcam=True, gradcampp=True, smoothgradcampp=True, epochs=1, learning_rate=0.001, momentum=0.9):
    """
    Parameters
    ----------
    path_modelos : str
        path where the models will be stored.
    cam : bool, optional
        If True, this model will be trained. The default is True.
    gradcam : bool, optional
        If True, this model will be trained. The default is True.
    gradcampp : bool, optional
        If True, this model will be trained. The default is True.
    smoothgradcampp : bool, optional
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
    technics = {'cam': cam, 'gradcam': gradcam, 'gradcampp':gradcampp, 'smoothgradcampp':smoothgradcampp}

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
    for i, (inputs, labels) in enumerate(dataloader_test):
        inputs = inputs.to(device)
        labels = labels.to(device)

        if i % 200 == 199:
            print('[%d] loss: %.3f' %( i, running_loss / (i * inputs.size(0))))

        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    final_loss = running_loss / test_size
    final_acc = running_corrects.double() / test_size
    print('TEST\n','Loss: {:.4f} Acc: {:.4f}'.format(final_loss, final_acc))

    
    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return final_loss, final_acc


def test_cam_models(path_modelos, cam=True, gradcam=True, gradcampp=True, smoothgradcampp=True):

    # List of technics to train
    technics = {'cam': cam, 'gradcam': gradcam, 'gradcampp':gradcampp, 'smoothgradcampp':smoothgradcampp}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('DEVICE: ', device)

    # Read datasets
    dataloaders, dataset_sizes = load_data(r'..\..\SICAPv1\299_patch_impar')

    models_dic = {}
    for name in technics.keys():
        # Load model
        models_dic[f'{name}'] = load_model(path_modelos, name, device)

        criterion = torch.nn.CrossEntropyLoss() 

        test_model(models_dic[f'{name}']['model'] , dataloaders['test'], dataset_sizes['test'], criterion, device)