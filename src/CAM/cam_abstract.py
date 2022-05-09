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
    

    def saliency_map(self, x, technic='gradcam', n_noise=1, std=0, device='cuda'):
        # Evaluate mode
        self.eval()

        if n_noise>1 and technic=='smoothgradcampp':
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
        parameters = self.get_weights(technic, activations, device=device)
        
        # Getting the tensor with three dimensions (if there are noise, we get the average of the noisy activations)
        activations = activations.mean(axis=0)
            
        # utils
        relu = nn.ReLU()
        
        heatmaps=torch.tensor([]).to(device)
        for class_i in range(self.n_classes):
            # Getting the heatmaps: w_1*Act_1 + w_2*Act_2 +...+ w_n*Act_n activations.shape
            activations_final = relu(((parameters[class_i]*activations.T).T).sum(axis=0))
   
            heatmaps = torch.cat((heatmaps, activations_final[None,:,:]))

        return heatmaps
    
    def plot_saliency_map(self, x, y, technic, mask=None, class_plot=-1, n_noise=10, std=0.3, device='cuda'):
        """
        Parameters:
        -----------
            - x: input 

            - technic: Class Activation Mapping technic to use
            
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
        
        if mask is not None:
            mask_plot = torch.reshape(mask.cpu(), (3,224,224)).permute(1,2,0).numpy()
        
        # Getting the heatmaps
        heatmaps_pre = self.saliency_map(x, technic=technic, n_noise=n_noise, std=std, device=device)
        y_prob = soft(self(x))
    
        heatmaps_new = list()
        
        # Dividimos por el mayor absoluto para tener mapas 
        for i in range(len(heatmaps_pre)):
            maximo = heatmaps_pre[i].max()
            heatmaps_pre[i] /= maximo

        # Visualización
        for hm in heatmaps_pre: 
            heatmaps_new.append(hm.cpu().detach().numpy())

        res=[]
        res.append(cv2.resize(heatmaps_new[0], dsize=(224, 224), interpolation=cv2.INTER_CUBIC))
        res.append(cv2.resize(heatmaps_new[1], dsize=(224, 224), interpolation=cv2.INTER_CUBIC))
        
        res_mask=[]
        for i in range(len(res)):
            aux = np.zeros_like(res[i])
            aux[res[i]>0.25] = 1.
            
            res_mask.append(cv2.cvtColor(aux,cv2.COLOR_GRAY2RGB))
        
        
        # Taking the class predicted
        y_pred_mod_new = torch.argmax(y_prob, dim=1)
        dic_prob = {"sano": y_prob[0][0], "cancer": y_prob[0][1] }
        name_classes = list(dic_prob.keys())
        
        cam_pred_name = name_classes[y_pred_mod_new[0]] 
        cam_act_name = name_classes[y] 

        if self.name=='CAM':
            technic=='CAM'
            
        print('PROB {}:\n\t- SANO: {:.5f}\n\t- CANCER: {:.5f}'.format(technic.upper(), dic_prob["sano"], dic_prob["cancer"]))
        print(f'(CLASS PREDICTED -- {cam_pred_name}) vs ({cam_act_name} -- ACTUAL CLASS)')
        
        
        ###########################
        #   HACEMOS PLOT
        plot_hm = np.zeros(self.n_classes)
        for i in range(self.n_classes):
            if class_plot==-2 or (class_plot==-1 and i==int(y_pred_mod_new)) or (class_plot==i):
                plot_hm[i]=1
        
                    
        n_cols=int(plot_hm.sum())+1
        n_rows=2
        
        fig, axes = plt.subplots(nrows=n_rows,
                                 ncols=n_cols,
                                 gridspec_kw={'width_ratios':np.ones(n_cols)})
        
        # Plot de la imagen original
        axes[0][0].title.set_text("---- IMAGEN ORIGINAL ----")
        axes[0][0].imshow(x_plot)
        
        if mask is not None:
            axes[1][0].title.set_text("---- MÁSCARA ORIGINAL ----")
            axes[1][0].imshow(mask_plot)
            
        curr_axe_idx = 0
        for i in range(self.n_classes):
            if plot_hm[i]==1:
                mask_2 = cv2.cvtColor(mask_plot,cv2.COLOR_RGB2GRAY)
                mask_2 -= mask_2.min()
                mask_2 /= mask_2.max()
                print("MAX", mask_2.max(), "MIN", mask_2.min(), "SHAPE", mask_2.shape)
                print("MEJOR VALOR: ", (res[i]*mask_2).mean())
                
                if i==0:
                    axes[1][curr_axe_idx+1].title.set_text("---- MÁSCARA T. SANO ----")
                else:
                    axes[1][curr_axe_idx+1].title.set_text("---- MÁSCARA T. CANCERÍGENO ----")
                axes[1][curr_axe_idx+1].imshow(res_mask[i])
                
                
                if i==0:
                    axes[0][curr_axe_idx+1].title.set_text("---- HEATMAP T. SANO ----")
                else:
                    axes[0][curr_axe_idx+1].title.set_text("---- HEATMAP T. CANCERÍGENO ----")
                axes[0][curr_axe_idx+1].imshow(res[i], cmap=plt.get_cmap('turbo'))#cmap_good_vs_evil)
                axes[0][curr_axe_idx+1].imshow(x_plot, alpha=0.5)
                
                
                
                

                # Update current axe
                curr_axe_idx +=1
        
            
        plt.subplots_adjust(left=0.1,
                            bottom=0.1, 
                            right=0.9, 
                            top=0.9, 
                            wspace=0.4, 
                            hspace=0.4)
        plt.show()
