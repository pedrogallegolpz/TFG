from sklearn.metrics import accuracy_score
import torch
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F

from torchvision import transforms
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
import math

from torch.utils.tensorboard import SummaryWriter


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
FUNCIONES PyTORCH
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# helper functions
def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
FUNCIONES PROPIAS
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def areas_iou(model, name, dataloader, writer=None, epoch=0, alpha=0.05, device='cuda'):
    '''
    Genera una figura de matplotlib del intersection over union del modelo
    con todos los modelos diseñados para él.
    '''
    technics = ['cam','gradcam', 'gradcampp', 'smoothgradcampp']

    iou_tecnicas = list()
    areas_bajo_iou = list()
    

    if 'cam' != name.split('-')[1]:
        ncols = 3
        technics = ['gradcam', 'gradcampp', 'smoothgradcampp']
    else:
        ncols = 1
        technics = ['cam']
    
    axe_idx=0
    fig, axes = plt.subplots(nrows=1,
                            ncols=ncols,
                            figsize=(5*ncols,5)) 
    
    plt.setp(axes, xticks=np.linspace(0,1,11), yticks=np.linspace(0,1,11))
    
    for technique in technics:
        iou_medios, _, mejor_umbral = curvas_umbral_mascara(model, 
                                                            technique, 
                                                            dataloader, 
                                                            alpha=alpha,
                                                            device=device)
        iou_tecnicas.append(iou_medios)



        print('Técnica: ', technique)
        
        
        # Área bajo la curva
        area = 0
        for i in range(len(iou_medios)-1):
            minimo = min(iou_medios[i],iou_medios[i+1])
            maximo = max(iou_medios[i],iou_medios[i+1])
            area += minimo*alpha + (maximo-minimo)*(alpha*0.5)   # Es como sumar el cuadrado y luego el triángulo que queda arriba     

            
        
        writer.add_scalar(f'IoU_evol-{technique}',
                          area,
                          epoch)


    return fig


def get_data_transforms(dsize):
    # Just normalize for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(5),
            transforms.Resize([dsize,dsize]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize([dsize,dsize]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize([dsize,dsize]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    return data_transforms


def load_data(path, dsize=224, batch_size=8, split_size=0.2):# Data augmentation and normalization for training
    data_transforms = get_data_transforms(dsize = dsize)
    
    data_dir = path
    image_datasets = {x: ImageFolder_and_MaskFolder(os.path.join(data_dir, x),
                                                             data_transforms[x]
                                                    )
                      for x in ['train', 'val', 'test']}

    dataset_sizes = {
                        "train": len(image_datasets["train"]),
                        "val": len(image_datasets["val"]),
                        "test": len(image_datasets["test"])
                    }

    dataloaders = {
                    "train": torch.utils.data.DataLoader(image_datasets["train"], 
                                                         batch_size=batch_size,
                                                         shuffle=True,
                                                         num_workers=2),
                    "val": torch.utils.data.DataLoader(image_datasets["val"], 
                                                         batch_size=1,
                                                         shuffle=False,
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


def load_model(path_modelos, name, device, original_model=None, print_terminal=True):
    name = name.lower()
    model = {}
    # Load model
    try:
        model['model'] = torch.load(path_modelos+f"model_{name}.pth").to(device)
        with open(path_modelos+f'dic_best_values_model_{name}.json') as f_dic:
             model['best_values'] = json.load(f_dic)
        
        if print_terminal:
            print(f"model_{name} loaded")
    except:
        if name.split('-')[-1]=='cam':
            model['model'] = CAM.cam.CAM_model(original_model, D_out=2)
        else:
            model['model'] = CAM.cam.CAM(original_model, D_out=2)
       
        model['best_values'] = {'loss': math.inf, 'acc':0., 'epochs':0}

        if print_terminal:
            print(f"model_{name} not found") 
        
        
    model['model'] = model['model'].to(device)
    
    return model


############################################################################
# ENTRENAMIENTO
############################################################################
def train_model(model, name, dic_best_values, dataloaders, dataset_sizes,criterion, optimizer, scheduler, early_stopping=-1, num_epochs=2, device='cuda', writer = None):
    since = time.time()

    # Saving actual weights as best weights
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = dic_best_values['loss']
    best_acc = dic_best_values['acc']
    curr_epoch = dic_best_values['epochs']
    num_epochs_ran = 0
    epochs_without_improvements = 0
    finish_train = False
    
    for epoch in range(num_epochs):
        if finish_train:
            break
        
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            images_seen = 0

            # Iterate over data.
            for i, (inputs, _, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

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
                    
                images_seen += inputs.size(0)
                if (i %  int(dataset_sizes[phase]/(20*len(inputs))+0.5)==0 and i>0) or i==dataset_sizes[phase]-1:
                    print(f'[{name}]. Epoch [{epoch + 1}/{num_epochs}], Batch[{i}/{int(dataset_sizes[phase]/len(inputs))}] loss: {running_loss / (i * inputs.size(0)):.3f}')
                    
                    if not writer is None:
                        # ...log the running loss
                        writer.add_scalar(f'{phase} loss',
                                            running_loss / (i * inputs.size(0)),
                                            (curr_epoch+epoch) * dataset_sizes[phase] + images_seen)


                        
            
                        
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            num_epochs_ran = epoch 

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

                # early stopping
                epochs_without_improvements = 0

            else:
                # No hay mejora. 
                # early stopping
                epochs_without_improvements +=1
                if epochs_without_improvements >= early_stopping and early_stopping>0:
                    finish_train = True
                    
                    print(f'Early stopping reached. The last best model found was {early_stopping} epochs ago.')
                    
                    break

            print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:.4f} Best val loss: {:.4f}'.format(best_acc, best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, best_loss, best_acc, curr_epoch+num_epochs_ran+1


def train_cam_models(path_modelos, path_dataset, cam=True, cam_pro=True, epochs=1, learning_rate=0.001, early_stopping=-1, momentum=0.9, device=None):
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

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('DEVICE: ', device)
    
    
    models_base = { 'VGG': {'model':models.vgg16(pretrained=True),
                            'input_size': 224},
                    'RESNET18': {'model':models.resnet18(pretrained=True),
                               'input_size': 224},
                    'MOBILENET': {'model':models.mobilenet_v2(pretrained=True),
                                  'input_size': 224},
                    'EFFICIENTNET': {'model':torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 
                                                            'nvidia_efficientnet_b0', 
                                                            pretrained=True),
                                     'input_size': 224},
                    'RESNET101': {'model': torch.hub.load('pytorch/vision:v0.10.0',
                                                          'resnet101', 
                                                          pretrained=True),
                                  'input_size': 224}
                   }
    
    models_dic = {}
    
    
    for modelo_base in list(models_base.keys())[4:]:
        for name in technics.keys():
            if not technics[f'{name}']:
                continue
            
            
            writer = SummaryWriter(f'runs/train/{modelo_base}-{name}')

            # Read datasets
            dataloaders, dataset_sizes = load_data(path_dataset,
                                                   dsize = models_base[f'{modelo_base}']['input_size'])
            
            
            print('\n', name.upper())
            
            # Load model
            models_dic[f'{modelo_base}-{name}'] = load_model(path_modelos,
                                                            f'{modelo_base}-{name}', 
                                                            device, 
                                                            original_model=models_base[f'{modelo_base}']['model'])
            
            
            # Cogemos el optimizador y el criterio de aprendizaje
            optimizer = optim.SGD(models_dic[f'{modelo_base}-{name}']['model'].parameters(), lr=learning_rate, momentum=momentum)
            criterion = torch.nn.CrossEntropyLoss() 
        
            # Decay LR by a factor of 0.5 every 5 epochs
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        
            # Entrenamos
            model, best_val_loss, best_val_acc, epochs_ran = train_model(models_dic[f'{modelo_base}-{name}']['model'],
                                                                        f'{modelo_base}-{name}',
                                                                        models_dic[f'{modelo_base}-{name}']['best_values'],
                                                                        dataloaders, 
                                                                        dataset_sizes,
                                                                        criterion,
                                                                        optimizer,
                                                                        exp_lr_scheduler,
                                                                        early_stopping=early_stopping,
                                                                        num_epochs = epochs,
                                                                        writer = writer,
                                                                        device = device)
        
            # Guardamos el mejor modelo del entrenamiento
            torch.save(model, path_modelos+f"model_{modelo_base}-{name.lower()}.pth")
    
            try:
                acc = best_val_acc.item()
            except:
                acc = best_val_acc

            models_dic[f'{modelo_base}-{name}']['best_values'] = {'loss': best_val_loss, 
                                                                  'acc': acc,
                                                                  'epochs': epochs_ran}
            with open(path_modelos+f'dic_best_values_model_{modelo_base}-{name.lower()}.json','w') as f_dic:
                json.dump(models_dic[f'{modelo_base}-{name}']['best_values'], f_dic)
            
            print(f'model_{modelo_base}-{name.lower()} guardado')
        
        
############################################################################
# TEST
############################################################################
def test_model(model, dataloader_test, test_size, criterion, device='cuda', writer=None):
    since = time.time()

    model.eval()   # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0

    tpfptnfn = [[0,0],[0,0]]

    # Iterate over data.
    for i, (inputs, mask, labels) in enumerate(dataloader_test):
        inputs = inputs.to(device)
        labels = labels.to(device)

        
        if i % 500 == 499:
            print('[%d] loss: %.3f' %( i, running_loss / (i * inputs.size(0))))
            if not writer is None:
                # ...log the running loss
                writer.add_scalar('Test loss',
                                    running_loss / (i * inputs.size(0)),
                                    i * inputs.size(0))
               
        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # statistics
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data)
        
        for i in range(len(labels)):
           tpfptnfn[labels[i].item()][preds[i].item()]+=1

        

    # Positive means pathological
    TP = tpfptnfn[1][1] 
    FN = tpfptnfn[1][0] 
    FP = tpfptnfn[0][1] 
    TN = tpfptnfn[0][0] 
                        
    accuracy = (TP+TN)/(TP+FP+FN+TN)
    recall = (TP)/(TP+FN)
    precision = (TP)/(TP+FP)
    f1score = 2 * (precision*recall)/(precision+recall)

    print(f'TP: {TP},\t FN: {FN}')       
    print(f'FP: {FP},\t TN: {TN}\n')       
    print(f'Accuracy: {accuracy}')
    print(f'Recall: {recall}')
    print(f'Precision: {precision}')
    print(f'F1-Score: {f1score}')

    final_loss = running_loss / test_size
    final_acc = running_corrects.double() / test_size
    print('TEST\n','Loss: {:.4f} Acc: {:.4f}'.format(final_loss, final_acc))

    
    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return final_loss, final_acc


def test_cam_models(path_modelos, path_dataset, cam=True, cam_pro=True):
    # List of technics to train
    technics = {'cam': cam, 'cam_pro': cam_pro}
    
    modelos_base = ['VGG', 'RESNET18', 'MOBILENET', 'EFFICIENTNET', 'RESNET101']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('DEVICE: ', device)

    # Read datasets
    dataloaders, dataset_sizes = load_data(path_dataset)

    models_dic = {}
    
    for modelo_base in modelos_base:
        for name in technics.keys():
            if not technics[f'{name}']:
                continue
            
            # Load model
            models_dic[f'{modelo_base}-{name}'] = load_model(path_modelos, 
                                                             f'{modelo_base}-{name}', 
                                                             device)
    
            criterion = torch.nn.CrossEntropyLoss() 
    
            test_model(models_dic[f'{modelo_base}-{name}']['model'] , 
                       dataloaders['test'], 
                       dataset_sizes['test'],
                       criterion, 
                       device)
            
        
        
############################################################################
# UMBRAL PARA LA MÁSCARA
############################################################################
def prepare_mask(mask):
    try:
        mask.cpu()
    except:
        pass
    
    if len(mask.shape)!=2:
        if len(mask.shape)==4:
            mask = mask[0]
            
        if mask.shape[0]==3:
            mask = mask.permute(1,2,0)
        
        if not isinstance(mask,np.ndarray):
            mask = mask.numpy()
        
        
        if mask.shape[2]==3:
            mask = cv2.cvtColor(mask,cv2.COLOR_RGB2GRAY)
        
    # Reescalamos
    mask = cv2.resize(mask, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

    
    mask[mask>0.1]=1.
    mask[mask<0.1]=0.
    
    return mask

def iou(mask1, mask2):
    """
     INTERSECTION OVER UNION
    """
    mask1 = prepare_mask(mask1)
    mask2 = prepare_mask(mask2)
    
    iou = np.logical_and(mask1, mask2).sum() / np.logical_or(mask1, mask2).sum()
    
    return iou

    
def iom1(mask1, mask2):
    """
     INTERSECTION OVER MASK1
    """
    mask1 = prepare_mask(mask1)
    mask2 = prepare_mask(mask2)

    if mask1.sum()==0:
        arreglo_den = 1
    else:
        arreglo_den = 0
    
    iou = np.logical_and(mask1, mask2).sum() / (mask1.sum()+arreglo_den)
    
    return iou

    
    
    
def generate_mask_from_heatmap(heatmap, umbral, maximo=-1):
    try:
        # Pasamos a np array
        heatmap = heatmap.cpu().detach().numpy()
    except:
        pass

    # Normalizamos con el máximo representativo
    heatmap -= heatmap.min()
    relu = torch.nn.ReLU()

    heatmap_norm = heatmap #torch.log2(torch.ones_like(heatmap)+relu(heatmap))
    
    if maximo==-1:
        maximo = heatmap_norm.max()
    heatmap_norm /= maximo 
    
    
    
    # Reescalamos
    heatmap_aumentado = cv2.resize(heatmap, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    heatmap_norm_aumentado = cv2.resize(heatmap_norm, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    
    # Creamos la máscara
    mask = np.zeros_like(heatmap_norm_aumentado)
    mask[heatmap_norm_aumentado>=umbral] = 1.
    
    
    return heatmap_aumentado, mask
    
    
 
    

def curvas_umbral_mascara(model, technique, dataloader, alpha, maximo_porcentaje_mascara, device='cuda'):
    model.eval()   # Set model to evaluate mode

    # Cogemos los heatmaps para sacar un valor máximo de activación representativo
    maximos_heatmaps = list()
    heatmaps = list()
    masks_heatmaps =  list()
    labels_heatmaps =  list()

    num_imagenes_borrar = 0

    print("técnica: ",technique)
    for i, (inp, mask, label) in enumerate(dataloader):
        if label[0].item()!=1 or (mask.mean()/mask.max()>maximo_porcentaje_mascara):
            continue
    
        num_imagenes_borrar+=1
        
        inp = inp.to(device)
        label=label.to(device)

        heatmaps.append(model.saliency_map(inp,
                                            technique,
                                            n_noise=10,
                                            std=0.25, 
                                            device=device)[label[0].item()])
        """
        UMBRAL (GRADCAMPP Y SMOOTH):
            - 0.1 
                +(0.738 y 0.739)
                +(0.718 y 0.718)
                +(0.734 y 0.733)
                +(0.68 y 0.68)

            - 0.25
                +(0.738 y 0.743)
                +(0.718 y 0.709)
                +(0.734 y 0.717)
                +(0.68 y 0.687)
                +(0.721 y 0.712)

        
        """
        masks_heatmaps.append(mask)
        labels_heatmaps.append(label)

        maximos_heatmaps.append(heatmaps[-1].cpu().max())
            

    print(f"Num Borrar: ",num_imagenes_borrar)
    # Calculamos el valor máximo de activación representativo
    maximo_representativo = np.percentile(maximos_heatmaps, 5)
    
    iou_tabla = []
    io_originalmask_tabla = []
    io_modelmask_tabla = []
    
    
    
    nrows =1+int(math.sqrt(len(heatmaps)))
    ncols = int(math.sqrt(len(heatmaps)))
    """
    fig, axes = plt.subplots(nrows=nrows,
                             ncols=ncols,
                             figsize=(ncols*5,nrows*5))
    """
   
    i,j=0,0
    for heatmap, mask, label in zip(heatmaps, masks_heatmaps, labels_heatmaps):
        iou_fila = []
        io_originalmask_fila = []
        io_modelmask_fila = []
        
        for umbral in np.concatenate((np.linspace(0,0.225,int((0.5/alpha))),np.linspace(0.25,1,int((0.5/alpha)+1)))):
            heatmap_aumentado, mask_generated = generate_mask_from_heatmap(heatmap=heatmap,
                                                                           umbral=umbral)

            
            iou_fila.append(iou(mask, mask_generated))
            io_originalmask_fila.append(iom1(mask, mask_generated))
            io_modelmask_fila.append(iom1(mask_generated, mask))

        # Añadimos la fila del umbral a cada tabla
        iou_tabla.append(iou_fila)
        io_originalmask_tabla.append(io_originalmask_fila)
        io_modelmask_tabla.append(io_modelmask_fila)

    # Pasamos a np array y calculamos estadísticas
    iou_tabla = np.array(iou_tabla)
    io_originalmask_tabla = np.array(io_originalmask_tabla)
    io_modelmask_tabla = np.array(io_modelmask_tabla)

    # Media
    valores_iou_medios = iou_tabla.mean(axis=0)
    valores_io_originalmask_medios = io_originalmask_tabla.mean(axis=0)
    valores_io_modelmask_medios = io_modelmask_tabla.mean(axis=0)
    valores_medios = [valores_iou_medios, valores_io_originalmask_medios, valores_io_modelmask_medios]


    # Mejores umbrales
    mejor_umbral_iou = np.concatenate((np.linspace(0,0.225,int((0.5/alpha))),np.linspace(0.25,1,int((0.5/alpha)+1))))[np.argmax(valores_iou_medios)]
    mejor_umbral_io_originalmask = np.concatenate((np.linspace(0,0.225,int((0.5/alpha))),np.linspace(0.25,1,int((0.5/alpha)+1))))[np.argmax(valores_io_originalmask_medios)]
    mejor_umbral_io_modelmask = np.concatenate((np.linspace(0,0.225,int((0.5/alpha))),np.linspace(0.25,1,int((0.5/alpha)+1))))[np.argmax(valores_io_modelmask_medios)]
    mejores_umbrales = [mejor_umbral_iou, mejor_umbral_io_originalmask, mejor_umbral_io_modelmask]


    return valores_medios, mejores_umbrales, maximo_representativo
    
                
                
      
def run_umbrales_iou(path_modelos, path_dataset, validation, maximo_porcentaje_mascara=1., cam=True, cam_pro=True, device=None):
    # List of technics to train
    model_classes = {'cam': cam, 'cam_pro': cam_pro}
    modelos_base = ['VGG', 'RESNET18', 'MOBILENET', 'EFFICIENTNET', 'RESNET101']

    alpha = 0.05

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('DEVICE: ', device)

    # Read datasets
    dataloaders, _ = load_data(path_dataset)
    dataloader = dataloaders['val'] if validation else dataloaders['test']
    models_dic = {}
    
    for modelo_base in modelos_base:
        iou_tecnicas = list()
        areas_bajo_iou = list()
        
        
        fig, axes = plt.subplots(nrows=3,
                                 ncols=4,
                                 figsize=(20,15)) 
        
        plt.setp(axes, yticks=np.linspace(0,1,11))
        
        axe_idx=0
        for name in model_classes.keys():
            if not model_classes[f'{name}'] :
                continue
            
            # Load model
            models_dic[f'{modelo_base}-{name}'] = load_model(path_modelos, f'{modelo_base}-{name}', device)
            
            if 'cam' != name:
                technics = ['gradcam', 'gradcampp', 'smoothgradcampp']
            else:
                technics = ['cam']



            for t_idx, technique in enumerate(technics):
                valores_medios, mejores_umbrales, maximo_representativo = curvas_umbral_mascara(models_dic[f'{modelo_base}-{name}']['model'], 
                                                                                        technique, 
                                                                                        dataloader, 
                                                                                        alpha=alpha,
                                                                                        maximo_porcentaje_mascara=maximo_porcentaje_mascara,
                                                                                        device=device)
                iou_tecnicas.append(valores_medios[0])
                
                # Guardamos info 
                models_dic[f'{modelo_base}-{name}']['best_values'][f'{technique}-maximo_representativo'] = maximo_representativo
                models_dic[f'{modelo_base}-{name}']['best_values'][f'{technique}-umbral'] = mejores_umbrales[0]
                with open(path_modelos+f'dic_best_values_model_{modelo_base}-{name.lower()}.json','w') as f_dic:
                    json.dump(models_dic[f'{modelo_base}-{name}']['best_values'], f_dic)
                    
                colors=['green','blue','orange']
                for i in range(len(valores_medios)):
                    # Área bajo la curva
                    area = 0
                    dominio = np.concatenate((np.linspace(0,0.225,int((0.5/alpha))),np.linspace(0.25,1,int((0.5/alpha)+1))))
                    for j in range(len(valores_medios[i])-1):
                        minimo = min(valores_medios[i][j],valores_medios[i][j+1])
                        maximo = max(valores_medios[i][j],valores_medios[i][j+1])
                        base = dominio[j+1]-dominio[j]
                        area += minimo*base + (maximo-minimo)*(base*0.5)   # Es como sumar el cuadrado y luego el triángulo que queda arriba     

                    areas_bajo_iou.append(area)
                        
                    if i==0:
                        title = f"Mejor valor: {valores_medios[0].max():.4f}\n con el umbral: {mejores_umbrales[i]:.4f}\n"
                        title += f"Área bajo la curva: {area:.4f}"
                    elif i==1:
                        title = f"Recall del mejor umbral: {valores_medios[1][valores_medios[0].argmax()]:.4f}"
                    elif i==2:
                        title = f"Precision del mejor umbral: {valores_medios[2][valores_medios[0].argmax()]:.4f}"

                    print(title)
                    # Ploteamos
                    axes[i][axe_idx].plot(dominio, valores_medios[i], 'o', color=colors[i])
                    axes[i][axe_idx].fill_between(dominio, valores_medios[i], color=colors[i], alpha=.25)
                    axes[i][axe_idx].title.set_text(title)
                    axes[i][axe_idx].set_yticks(np.linspace(0,1,11))
                axe_idx+=1
                    
        plt.show()
    
    
        
        
        
def comprobacion_max_iou_posible(path_dataset, alpha=0.05):
    # Read datasets
    dataloaders, dataset_sizes = load_data(path_dataset)
    
    iou_tabla = []
    for i, (inp, mask_, label) in enumerate(dataloaders['test']):
        mask = mask_[0]
        mask_sm = cv2.resize(mask[0].numpy(), dsize=(7,7), interpolation=cv2.INTER_CUBIC)

        iou_fila = []

        if i%200==0:
            print("Step ",i)
        
        for umbral in np.linspace(0,1,int((1./alpha)+1)):
            _, mask_generated = generate_mask_from_heatmap(heatmap=mask_sm,
                                                           umbral=umbral)

            iou_fila.append(iou(mask, mask_generated))

        iou_tabla.append(iou_fila)

    iou_tabla = np.array(iou_tabla)
    iou_medios = iou_tabla.mean(axis=0)

    mejor_umbral = np.argmax(iou_medios)*alpha

    # Área bajo la curva
    area = 0
    for i in range(len(iou_medios)-1):
        minimo = min(iou_medios[i],iou_medios[i+1])
        maximo = max(iou_medios[i],iou_medios[i+1])
        area += minimo*alpha + (maximo-minimo)*(alpha*0.5)   # Es como sumar el cuadrado y luego el triángulo que queda arriba     



    title = f"Mejor valor: {iou_medios.max():.3f}\n con el umbral: {mejor_umbral:.4f}\n"
    title += f"Área bajo la curva: {area}"

    print(title)
    
    fig, ax = plt.subplots(nrows=1,
                           ncols=1,
                           figsize=(20,5)) 
    
    plt.setp(ax, yticks=np.linspace(0,1,11))
    # Ploteamos
    ax.plot(np.linspace(0,1,int((1./alpha)+1)), iou_medios, 'o')
    ax.fill_between(np.linspace(0,1,int((1./alpha)+1)), iou_medios, color='blue', alpha=.25)
    ax.title.set_text(title)

    plt.show()
