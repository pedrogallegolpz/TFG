import abc  # For implementing abstract methods
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch
from torch import nn

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
    def get_weights(self, activations=None, device='cuda'):
        """
        Parameters:
        -----------
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
    def get_subweights(self, activations=None, grad=None):
        """
        Parameters:
        -----------
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
    

    def saliency_map(self, x, n_noise=1, std=0, device='cuda'):
        # Evaluate mode
        self.eval()

        if n_noise>1:
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
        parameters, y = self.get_weights(activations, device=device)
        
        # Getting the tensor with three dimensions (if there are noise, we get the average of the noisy activations)
        activations = activations.mean(axis=0)
            
        # utils
        relu = nn.ReLU()
        
        heatmaps=torch.tensor([]).to(device)
        for class_i in range(self.n_classes):
            # Getting the heatmaps: w_1*Act_1 + w_2*Act_2 +...+ w_n*Act_n activations.shape
            activations_final = relu(((parameters[class_i]*activations.T).T).sum(axis=0))
   
            heatmaps = torch.cat((heatmaps, activations_final[None,:,:]))

        return heatmaps, y 
    
    def plot_saliency_map(self, x, y, class_plot=-1, n_noise=1, std=0, device='cuda'):
        """
        Parameters:
        -----------
            - x: input 

            - y: {x} actual class

            - class_plot:
                    {-2} to plot all classes
                    {-1} to plot the class predicted
                    {n}  to plot the n-class (n in {0,1,2,...})
                    
            -n_noise: int
                number of noised inputs
        """
        # utils
        soft = nn.Softmax(dim=1)

        mean = [0.485, 0.456, 0.406]
        var = [0.229, 0.224, 0.225]
        x_plot=((torch.reshape(x.cpu(), (3,224,224)).permute(1,2,0).numpy())*var)+mean
        
        # Getting the heatmaps
        heatmaps_pre, y_pred = self.saliency_map(x, n_noise=n_noise, std=std, device=device)
        y_prob = soft(y_pred)
    
        heatmaps_new = list()
        
        # Dividimos por el mayor absoluto para tener mapas relativos
        heatmaps_pre = heatmaps_pre / torch.max(torch.abs(heatmaps_pre))
        
        # Visualización
        for hm in heatmaps_pre: 
            heatmaps_new.append(hm.cpu().detach().numpy())

        res=[]
        res.append(cv2.resize(heatmaps_new[0], dsize=(224, 224), interpolation=cv2.INTER_CUBIC))
        res.append(cv2.resize(heatmaps_new[1], dsize=(224, 224), interpolation=cv2.INTER_CUBIC))

        
        # Taking the class predicted
        y_pred_mod_new = torch.argmax(y_prob, dim=1)
        dic_prob = {"sano": y_prob[0][0], "cancer": y_prob[0][1] }
        name_classes = list(dic_prob.keys())
        
        cam_pred_name = "sano" if y_pred_mod_new[0]==0 else "cancer"
        cam_act_name = "sano" if y==0 else "cancer"

        print('PROB {}:\n\t- SANO: {:.5f}\n\t- CANCER: {:.5f}'.format(self.name, dic_prob["sano"], dic_prob["cancer"]))
        print(f'(CLASS PREDICTED -- {cam_pred_name}) vs ({cam_act_name} -- ACTUAL CLASS)')
        
        
        ###########################
        #   HACEMOS PLOT
        plot_hm = np.zeros(self.n_classes)
        for i in range(self.n_classes):
            if class_plot==-2 or (class_plot==-1 and i==int(y_pred_mod_new)) or (class_plot==i):
                plot_hm[i]=1
        
                    
        n_cols=int(plot_hm.sum())+1
        fig, axes = plt.subplots(nrows=1,
                                 ncols=n_cols,
                                 gridspec_kw={'width_ratios':np.ones(n_cols)})
        
        # Plot de la imagen original
        im = axes[0].imshow(x_plot)
        
        axes[0].title.set_text("----------- IMAGEN ORIGINAL -----------")
        
        curr_axe_idx = 0
        for i in range(self.n_classes):
            if plot_hm[i]==1:
                try:
                    curr_axe = axes[curr_axe_idx+1]
                except:
                    curr_axe = axes
                
                im = curr_axe.imshow(res[i], cmap=plt.get_cmap('turbo'), aspect='auto')#cmap_good_vs_evil)
                curr_axe.imshow(x_plot, alpha=0.5)
                if i==0:
                    curr_axe.title.set_text("----------- HEATMAP T. SANO -----------")
                else:
                    curr_axe.title.set_text("----------- HEATMAP T. CANCERÍGENO -----------")

                # Update current axe
                curr_axe_idx +=1
        plt.show()
