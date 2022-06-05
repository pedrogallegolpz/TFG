import abc  # For implementing abstract methods
import cv2
from matplotlib.colors import LinearSegmentedColormap
import torch
from torch import nn

import sys
try:
    from utils import generate_mask_from_heatmap
except:
    sys.path.append("../")
    from utils import generate_mask_from_heatmap
    
    
import matplotlib.pyplot as plt
    
# Define colormap
colors = [(1, 0, 0), (0, 0, 1),  (0, 1, 0)]  # R -> G -> B
n_bins = [3, 6, 10, 100]  # Discretizes the interpolation into bins
cmap_name = 'good_vs_evil'
cmap_good_vs_evil = LinearSegmentedColormap.from_list(cmap_name, colors)


class CAM_abstract:
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractproperty  # Abstract attribute
    def name(self):
        pass

    
    @abc.abstractmethod
    def get_activations(self, x):
        """
        Parameters:
        -----------
            -x: input image (tensor)
            
        Return:
        -------
            activations, the last values before passing
            through the first fully connected layer
        """
        return
    
    @abc.abstractmethod
    def get_weights(self,technic, activations=None, device='cuda'):
        """
        Parameters:
        -----------
            -technic: str
                for CAM is unnecessary. For CAM_pro indicates the technic
                to use for getting weights in order to plot a saliency map.
                the available values are: 
                        ['gradcam','gradcampp','smoothgradcampp']
                        
            -activations: tensor
                the last values before passing
                through the first fully connected layer
                
            -n_noise: int
                number of noised inputs
            
            -std: desviación para smooth
            
        Return:
        -------
            weights which are used to multiply the activations.
        """        
        return
    
    @abc.abstractmethod
    def get_subweights(self, technic, activations=None, grad=None):
        """
        Parameters:
        -----------
            -technic: str
                for CAM is unnecessary. For CAM_pro indicates the technic
                to use for getting weights in order to plot a saliency map.
                the available values are: 
                        ['gradcam','gradcampp','smoothgradcampp']
            
            -activations: tensor
                the last values before passing through
                the first fully connected layer

            -grad: tensor
                activations gradient for GradCAM++.
                None for CAM and GradCAM
                
            
        Return:
        -------
            weights which are used to multiply the gradients.
        """        
        return
    

    def saliency_map(self, x, technique, n_noise=1, std=0, device='cuda'):
        # Evaluate mode
        self.eval()

        if n_noise>1 and technique=='smoothgradcampp':
            # Calculate (if it's the case) the noisy inputs
            std_tensor = torch.ones_like(x) * std
            input_with_noise = torch.tensor([]).to(device)
            input_with_noise = torch.cat((input_with_noise,x))

            for n in range(n_noise):
                # Add noise to input
                x_noise = torch.normal(mean=x, std=std_tensor)
                input_with_noise = torch.cat((input_with_noise,x_noise))
                
            x = input_with_noise # rename in order to reuse the next 'get_weight' function for every case
            
        # We generate the activation map
        activations = self.get_activations(x)
        
        # Getting the parameters from the first layer of self.fc (the unique layer)
        parameters = self.get_weights(technique, activations, device=device)

        # Getting the tensor with three dimensions (if there are noise, we get the average of the noisy activations)
        activations = activations.mean(axis=0)
            
        # utils
        relu = nn.ReLU()
        
        heatmaps=torch.tensor([]).to(device)
        for class_i in range(self.n_classes):
            # Getting the heatmaps: w_1*Act_1 + w_2*Act_2 +...+ w_n*Act_n activations.shape
            activations_final = relu(((parameters[class_i]*activations.T).T).sum(axis=0)).to(device)
            
            heatmaps = torch.cat((heatmaps, activations_final[None,:,:]))

        return heatmaps.detach().clone()
    
    def generate_masks(self, x, dic_best_values, n_noise=10, std=0.1, gradcam=True, gradcampp=True, smoothgradcampp=True, device='cuda'):
        """
        Parameters:
        -----------
            - x: input 
            -n_noise: int
                number of noised inputs
            - std: float
                amount of standard desviation
        """
        dic_technics = {'cam':self.name=='CAM' and self.name=='CAM',
                        'gradcam':gradcam and self.name!='CAM',
                        'gradcampp':gradcampp and self.name!='CAM',
                        'smoothgradcampp':smoothgradcampp and self.name!='CAM'
                        }
        
        
        
        dic_heatmaps = {}
        dic_masks = {}
        for technique in dic_technics.keys():
            if dic_technics[f'{technique}']:
                # Getting the heatmaps
                heatmaps_pre = self.saliency_map(x, technique=technique, n_noise=n_noise, std=std, device=device)
                
                
                # Dividimos por el mayor absoluto para tener mapas 
                dic_heatmaps[f'{technique}'] = list()
                dic_masks[f'{technique}'] = list()
                for i in range(len(heatmaps_pre)):
                    aux_heatmap, aux_mask = generate_mask_from_heatmap(heatmaps_pre[i],
                                                                       umbral=0.15#dic_best_values[f'{technique}-umbral']
                                                                       )
                    dic_heatmaps[f'{technique}'].append(aux_heatmap) 
                    dic_masks[f'{technique}'].append(aux_mask) 
                    
        
        return dic_heatmaps, dic_masks
        
        
        
        
        
        
        
    