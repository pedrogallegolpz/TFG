from torch import nn
from torchvision.models.resnet import BasicBlock
import cv2
import sys
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

try:
    from utils import iou, load_model, prepare_mask
except:
    sys.path.append("../")
    from utils import iou, load_model, prepare_mask
    

def plot_grid(x, mask, y, 
              n_noise=10, 
              std=0.1, 
              cam=True, gradcam=True, gradcampp=True, smoothgradcampp=True, 
              vgg=True, resnet=True, mobilenet=True, efficientnet=True,
              device='cuda', 
              path_modelos="modelos/"):
    
    dic_models = {'cam':cam,
                  'cam_pro':gradcam or gradcampp or smoothgradcampp,
                }

    dic_nets={'vgg': vgg,
              'resnet': resnet,
              'mobilenet': mobilenet,
              'efficientnet': efficientnet
              }

    #############################
    # Prepare original data to plot
    # Unnormalize
    mean = [0.485, 0.456, 0.406]
    var = [0.229, 0.224, 0.225]
        
    x_plot = ((x[0].cpu().permute(1,2,0).numpy())*var)+mean
    x_plot = cv2.resize(x_plot, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    x_plot -= x_plot.min()
    x_plot /= x_plot.max()

    classes = ['BENIGN', 'PATHOLOGICAL']

    # Prepare mask to plot
    mask = prepare_mask(mask)
    
    
    #########################
    # Matplotlib 
    font = {'family' : 'serif',
        'weight' : 'bold',
        'size'   : 18}

    plt.rc('font', **font)

    columnas_grid = 1 + int(cam + gradcam + gradcampp + smoothgradcampp)
    filas_grid = int(vgg + resnet + mobilenet + efficientnet)
            
    fig = plt.figure(figsize=(5*columnas_grid, 5*filas_grid))
    gs = GridSpec(filas_grid, columnas_grid, figure=fig)

    ax_im_original = fig.add_subplot(gs[1,0])
    ax_im_original.title.set_text(f"{classes[y[0]]}")
    ax_im_original.imshow(x_plot)
    ax_im_original.set_xticks([])
    ax_im_original.set_yticks([])

    ax_mask_original = fig.add_subplot(gs[2,0])
    
    ax_mask_original.imshow(cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB))
    ax_mask_original.set_xticks([])
    ax_mask_original.set_yticks([])  

    
    curr_row_idx = -1
    for net_name in dic_nets.keys():
        if not dic_nets[f'{net_name}']:
            continue
        
        curr_row_idx +=1
        curr_col_idx = 0
        for model_name in dic_models.keys():
            model_dic = load_model(path_modelos, f'{net_name.upper()}-{model_name}', device, print_terminal=False)
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

            ###########################
            #   HACEMOS PLOT
            for technique in dic_heatmaps.keys():
                curr_col_idx += 1

                heatmap = dic_heatmaps[f'{technique}'][y_pred_mod_new[0]]
                
                #mask_model = dic_masks[f'{technique}'][y_pred_mod_new[0]]
                
                # IoU
                #iou_valor = iou(mask, mask_model)
                
                ax = fig.add_subplot(gs[curr_row_idx,curr_col_idx])
                if curr_col_idx == 1:
                    ax.set_ylabel(f'{net_name.upper()}', fontdict=font)

                if curr_row_idx == 0:
                    title = f'{technique.upper().replace("GRAD","Grad-").replace("PP","++").replace("SMOOTH", "Smooth ")}'
                    ax.set_title(title, fontdict=font) 

                if y_pred_mod_new[0]==y[0]:
                    plt.setp(ax.spines.values(), color='green', linewidth=5.)
                else:
                    plt.setp(ax.spines.values(), color='red', linewidth=5.)                

                ax.imshow(heatmap, cmap=plt.get_cmap('turbo')) 
                ax.imshow(x_plot, alpha=0.5)
        
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)

                ax.text(0.05, 0.95, f'Prob: {y_prob[0][y_pred_mod_new[0]]:.3f}', transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', bbox=props)

                ax.set_xticks([])
                ax.set_yticks([])

                

                
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
