from torch import nn
from torchvision.models.resnet import BasicBlock
import cv2
import sys
import matplotlib.pyplot as plt
import numpy as np

try:
    from utils import iou, load_model, prepare_mask
except:
    sys.path.append("../")
    from utils import iou, load_model, prepare_mask
    

def plot_grid(x, mask, y, net, n_noise=10, std=0.1, cam=True, gradcam=True, gradcampp=True, smoothgradcampp=True, device='cuda', path_modelos="modelos/"):
    dic_models = {'cam':cam,
                  'cam_pro':gradcam or gradcampp or smoothgradcampp
                    }
    #############################
    # Prepare original data to plot
    # Unnormalize
    mean = [0.485, 0.456, 0.406]
    var = [0.229, 0.224, 0.225]
        
    x_plot = ((x[0].cpu().permute(1,2,0).numpy())*var)+mean
    x_plot = cv2.resize(x_plot, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)


    # Prepare mask to plot
    mask = prepare_mask(mask)
    
    
    #########################
    # Generate masks 
    columnas_grid = 1 + int(cam + gradcam + gradcampp + smoothgradcampp)
    filas_grid = 2 # [Imagen o Heatmap, Máscara]
    fig, axes = plt.subplots(nrows=filas_grid,
                             ncols=columnas_grid,
                             figsize=(25,10))
    
    axes[0][0].title.set_text("IMAGEN ORIGINAL")
    axes[0][0].imshow(x_plot)
    axes[0][0].set_xticks([])
    axes[0][0].set_yticks([])
    
    axes[1][0].imshow(cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB))
    axes[1][0].set_xticks([])
    axes[1][0].set_yticks([])    
    
    curr_axe_idx = 1
    for model_name in dic_models.keys():
        model_dic = load_model(path_modelos, f'{net}-{model_name}', device)
        model = model_dic['model']
        
        dic_heatmaps,dic_masks = model.generate_masks(x,    
                                                      dic_best_values=model_dic['best_values'],
                                                      n_noise=n_noise,
                                                      std=std,
                                                      gradcam=True,
                                                      gradcampp=True,
                                                      smoothgradcampp=True,
                                                      device='cuda')
    
        # utils
        soft = nn.Softmax(dim=1)
        
        # Taking the class predicted
        y_prob = soft(model(x))
        y_pred_mod_new = np.argmax(y_prob.cpu().detach().numpy(), axis=1)

        classes = ['BENIGN', 'PATHOLOGICAL']

        ###########################
        #   HACEMOS PLOT
        for technique in dic_heatmaps.keys():
            heatmap = dic_heatmaps[f'{technique}'][y_pred_mod_new[0]]
            
            mask_model = dic_masks[f'{technique}'][y_pred_mod_new[0]]
            
            # IoU
            iou_valor = iou(mask, mask_model)
               
            # Heatmap
            title = (f'[{net}-{technique}]'+"\n"+"{0}, {1:.1f}%"+"\n"+"(label: {2})"+f"\nIou: {iou_valor}").format(
                                                                                 classes[y_pred_mod_new[0]],
                                                                                 y_prob[0,y_pred_mod_new[0]] * 100.0,
                                                                                 classes[y[0]])
            axes[0][curr_axe_idx].set_title(title, color=("green" if y_pred_mod_new[0]==y[0] else "red"))
            
            axes[0][curr_axe_idx].imshow(heatmap, cmap=plt.get_cmap('turbo')) 
            axes[0][curr_axe_idx].imshow(x_plot, alpha=0.5)
    
            axes[0][curr_axe_idx].set_xticks([])
            axes[0][curr_axe_idx].set_yticks([])
            
            # Máscara
            title_mask = f"umbral: {model_dic['best_values'][f'{technique}-umbral']:.3f}"
            axes[1][curr_axe_idx].set_title(title_mask)

            axes[1][curr_axe_idx].imshow(cv2.cvtColor(mask_model,cv2.COLOR_GRAY2RGB))
            
            axes[1][curr_axe_idx].set_xticks([])
            axes[1][curr_axe_idx].set_yticks([])
            
            # Update current axe
            curr_axe_idx +=1
    plt.show()

    return


def get_output_last_linear(module):
    out_features_channels = -1  # output
    for m in module.children():
        # Test the type
        if isinstance(m, nn.Linear):
            # If it's Conv2d
            out_features_channels = m.out_features
        elif isinstance(m, nn.Sequential):
            # If it's sequential
            out_features_channels_new = get_output_last_conv(m)
            if out_features_channels_new != -1:
                out_features_channels = out_features_channels_new

    return out_features_channels



def get_output_last_conv(module):
    out_features_channels = -1  # output
    for m in module.children():
        # Test the type
        if isinstance(m, nn.Conv2d):
            # If it's Linear
            out_features_channels = m.out_channels
        elif isinstance(m, nn.Sequential) or isinstance(m, BasicBlock):
            # If it's sequential
            out_features_channels_new = get_output_last_conv(m)
            if out_features_channels_new != -1:
                out_features_channels = out_features_channels_new

    return out_features_channels


def remove_modules_type(module, types_module):
    """
    Parameters:
    -----------
    module: nn.Module
      Module where we want to remove {type_module} modules.

    types_module: list of class which inherits from nn.Module
      Types to remove.

    Description:
    ------------
    This function remove all {type_module} modules from
    a first module: {module}

    Return:
    -------
    List of nn.Module
    """
    presence_type = False
    mod_without_type = []
    anadir_capa = True

    # Prueba:
    in_features_channels = []  # input of the module we are removing
    for m in module.children():
        # Test the type
        for type_module in types_module:
            if isinstance(m, type_module) and not presence_type:
                try:
                    if isinstance(m, nn.AdaptiveAvgPool2d):
                        if m.output_size==1 or m.output_size==(1,1):
                            # Queremos quitar solo los global avg pooling
                            # para que saque un tensor de activación el extractor
                            # de características
                            anadir_capa = False
                        
                    # If it's Linear
                    if isinstance(m, nn.Linear):
                        presence_type = True
                        in_features_channels.append(m.in_features)
                        anadir_capa = False
                except:
                    try:
                        # If it's Conv2d
                        in_features_channels.append(m.in_channels)
                    except:
                        pass

                break
        if isinstance(m, nn.Sequential):
            # If it's sequential
            modules_removed, in_features_channels_new = remove_modules_type(m, types_module)
            in_features_channels = in_features_channels + in_features_channels_new

            if len(modules_removed)>0:
                mod_without_type.append(nn.Sequential(*modules_removed))
        elif anadir_capa:
            mod_without_type.append(m)
            

    return mod_without_type, in_features_channels
