import math
import torch
from torch import nn
try:
    from cam_abstract import CAM_abstract
except:
    from CAM.cam_abstract import CAM_abstract

try:
    from utils_cam import remove_modules_type
    from utils_cam import get_output_last_conv
except:
    from CAM.utils_cam import remove_modules_type
    from CAM.utils_cam import get_output_last_conv
    
class CAM_model(nn.Module, CAM_abstract):
    name = "CAM"

    def __init__(self, originalModel, D_out):
        super(CAM_model, self).__init__()
        
        # Quitamos la FC
        fc_removed, in_features = remove_modules_type(originalModel, [nn.Linear])
        channels = get_output_last_conv(originalModel)

        self.list_modules = nn.ModuleList(fc_removed)
        self.avgPool_CAM = nn.AdaptiveAvgPool2d(output_size=(1))

        self.fc = nn.Linear(in_features=channels, out_features=D_out, bias=False)
        self.n_classes = D_out
        


    def forward(self, x):
        # utils
        flat = nn.Flatten()

        # forward
        x_mod = self.get_activations(x)
        x_mod = self.avgPool_CAM(x_mod) 
        x_mod = flat(x_mod) # flatten
    
        x_mod = self.fc(x_mod)

        return x_mod
    
    def get_activations(self, x):
        x_mod = x
        for mod in self.list_modules:
            x_mod = mod(x_mod)
        
        return x_mod

    def get_weights(self, activations=None, device='cuda'):
        # With default parameters
        flat = nn.Flatten()

        # Calculate output
        x_mod = activations
        x_mod = self.avgPool_CAM(x_mod) 
        x_mod = flat(x_mod) # flatten
        y = self.fc(x_mod)

        return next(iter(self.fc.parameters())), y
        
    def get_subweights(self, activations=None, grad=None):
        return None
        
        
        
class GradCAM_model(nn.Module, CAM_abstract):
    name = "GradCAM"
    
    def __init__(self, originalModel, D_out):
        super(GradCAM_model, self).__init__()
        
        # Quitamos la FC
        fc_removed, in_features_list = remove_modules_type(originalModel, [nn.Linear])
        self.list_modules = nn.ModuleList(fc_removed)
               
        # Definimos las FC nuevas con salida D_out
        in_features = in_features_list[-1]
        
        exp = int(0.5+math.log(in_features, 2)/2)

        in_out = [in_features, 2**exp, 2**exp, D_out]
        
        layers = []
        for i in range(len(in_out)-1):
            in_f = in_out[i]
            out_f = in_out[i+1]
            layers.append(nn.Linear(in_features=in_f, out_features=out_f, bias=True))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=0.5, inplace=False))
          
        self.fc = nn.Sequential(*layers[:-2])
        
        self.n_classes = D_out
        
    def forward(self, x):
        # utils
        flat = nn.Flatten()

        # forward
        x_mod = self.get_activations(x)
        x_mod = flat(x_mod) # flatten
        x_mod = self.fc(x_mod)

        return x_mod
    
    def get_activations(self, x):
        x_mod = x
        for mod in self.list_modules:
            x_mod = mod(x_mod)
        
        return x_mod

    def get_weights(self, activations=None, device='cuda'):
        assert(activations!=None)
        # With default parameters
        flat = nn.Flatten()
        
        # Enables this Tensor to have their grad populated during backward().
        activations.retain_grad() 
        
        weights = torch.tensor([]).to(device)
        for class_i in range(self.n_classes):
            # Set gradients to zero
            activations.grad = torch.zeros_like(activations)
            
            # forward
            y_pred = self.fc(flat(activations))
            torch.autograd.backward(y_pred[0][class_i], retain_graph=True)

            # Getting the parameters as the mean of the gradients
            weights = torch.cat((weights, torch.mean(activations.grad[0],(1,2))[None,:]))
            
        # Calculate output
        x_mod = activations
        x_mod = flat(x_mod) # flatten
        y = self.fc(x_mod)
            
        return weights, y

    def get_subweights(self, activations=None, grad=None):
        return None
   
        
        
        
class GradCAMpp_model(nn.Module, CAM_abstract):
    name = "GradCAM++"
    
    def __init__(self, originalModel, D_out):
        super(GradCAMpp_model, self).__init__()
        
        # Drop FC
        fc_removed, in_features_list = remove_modules_type(originalModel, [nn.Linear])
        self.list_modules = nn.ModuleList(fc_removed)
               
        # Define new FC with out_features=D_out
        in_features = in_features_list[-1]
        
        exp = int(0.5+math.log(in_features, 2)/2)

        in_out = [in_features, 2**exp, 2**exp, D_out]
        
        layers = []
        for i in range(len(in_out)-1):
            in_f = in_out[i]
            out_f = in_out[i+1]
            layers.append(nn.Linear(in_features=in_f, out_features=out_f, bias=True))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=0.5, inplace=False))
          
        self.fc = nn.Sequential(*layers[:-2])
                
        self.n_classes = D_out
        
    def forward(self, x):
        # utils
        flat = nn.Flatten()

        # forward
        x_mod = self.get_activations(x)
        x_mod = flat(x_mod) # flatten
        x_mod = self.fc(x_mod)
        
        return x_mod

    
    def get_activations(self, x):
        x_mod = x
        for mod in self.list_modules:
            x_mod = mod(x_mod)
        
        return x_mod

    def get_weights(self, activations=None, device='cuda'):
        assert(activations!=None)
        # With default parameters
        flat = nn.Flatten()

        
        # utils
        relu = nn.ReLU()
        
        # Enables this Tensor to have their grad populated during backward().
        activations.retain_grad() 
        
        weights = torch.tensor([]).to(device)
        for class_i in range(self.n_classes):
            # Set gradients to zero
            activations.grad = torch.zeros_like(activations)
            
            # forward
            s_c = self.fc(flat(activations))
            torch.autograd.backward(s_c[0][class_i], retain_graph=True)
            
            # get the dy/dA=exp(s)*ds/dA
            dydA = torch.exp(s_c[0][class_i])*activations.grad[0]

            # Getting the parameters as a weighted sum of the gradients
            subweights = self.get_subweights(activations[0], activations.grad[0])
            
            relu_dydA = relu(dydA)
            """
            # Normalizamos los alpha            
            subweights_thresholding = torch.where(relu_dydA>0, subweights, relu_dydA) # donde relu_dydA es cero, esto será cero
            
            subweights_normalization_constant = subweights_thresholding.sum(axis=(1,2))
            subweights_normalization_constant_processed = torch.where(subweights_normalization_constant != 0.0, 
                                                                      subweights_normalization_constant, 
                                                                      torch.ones_like(subweights_normalization_constant))
            
            subweights /= subweights_normalization_constant_processed[:,None,None].expand(subweights.shape)
            """
            new_weight = (subweights*relu_dydA).sum(axis=(1,2))
            
            weights = torch.cat((weights,new_weight[None,:]))
           
        
        # Calculate output
        x_mod = activations
        x_mod = flat(x_mod) # flatten
        y = self.fc(x_mod)
            
        return weights, y

    
    def get_subweights(self, activations=None, gradients=None):
        assert(activations!=None and activations!=None)
        # Numerator
        numerator = gradients.pow(2)

        print(activations.shape)
        print("activations.mean(axis=[1,2])",activations.sum(axis=[1,2])[:,None,None].expand(-1,7,7).shape)
        print(gradients.pow(3).shape)
        # Denominator
        ag = activations.sum(axis=[1,2])[:,None,None].expand(-1,7,7) * gradients.pow(3)

        print("ag.view..",ag.shape)
        print("ag.view..",ag.view(gradients.shape[0], -1).shape)
        print("ag.view..",ag.view(gradients.shape[0], -1).sum(-1, keepdim=True).shape)
        print("ag.view..",ag.view(gradients.shape[0], -1).sum(-1, keepdim=True).view(gradients.shape[0], 1, 1).shape)
        
        denominator = 2 * gradients.pow(2) 
        denominator += ag #ag.view(gradients.shape[0], -1).sum(-1, keepdim=True).view(gradients.shape[0], 1, 1)
        denominator = torch.where(denominator != 0.0, denominator, torch.ones_like(denominator))
        
        # Alpha
        alpha = numerator / denominator        
        
        alpha = torch.ones_like(gradients)/(2.*torch.ones_like(gradients)+activations.sum(axis=[1,2])[:,None,None].expand(-1,7,7) * gradients)
        
        return alpha
    
    
    
    
       
class SmoothGradCAMpp_model(nn.Module, CAM_abstract):
    name = "SmoothGradCAM++"
    
    def __init__(self, originalModel, D_out):
        super(SmoothGradCAMpp_model, self).__init__()
        
        # Drop FC
        fc_removed, in_features_list = remove_modules_type(originalModel, [nn.Linear])
        self.list_modules = nn.ModuleList(fc_removed)
               
        # Define new FC with out_features=D_out
        in_features = in_features_list[-1]
        
        exp = int(0.5+math.log(in_features, 2)/2)

        in_out = [in_features, 2**exp, 2**exp, D_out]
        
        layers = []
        for i in range(len(in_out)-1):
            in_f = in_out[i]
            out_f = in_out[i+1]
            layers.append(nn.Linear(in_features=in_f, out_features=out_f, bias=True))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=0.5, inplace=False))
          
        self.fc = nn.Sequential(*layers[:-2])
                
        self.n_classes = D_out

        
    def forward(self, x):
        # utils
        flat = nn.Flatten()

        # forward
        x_mod = self.get_activations(x)
        x_mod = flat(x_mod) # flatten
        x_mod = self.fc(x_mod)
        
        return x_mod

    
    def get_activations(self, x):
        x_mod = x
        for mod in self.list_modules:
            x_mod = mod(x_mod)
        
        return x_mod

    def get_weights(self, activations=None, device='cuda'):
        assert(activations!=None)        
        # With default parameters
        flat = nn.Flatten()

        # utils
        relu = nn.ReLU()
        
        s_c_with_noise = torch.tensor([]).to(device)
        gradients_with_noise = [torch.tensor([]).to(device),torch.tensor([]).to(device)]
        
        # Calculating gradients with noise
        for n in range(activations.shape[0]):
            # Enables this Tensor to have their grad populated during backward().
            act_noise = activations[n][None,:,:,:]
            act_noise.retain_grad() 

            weights = torch.tensor([]).to(device)
            s_c_with_noise_classes = torch.tensor([]).to(device)
            for class_i in range(self.n_classes):
                # Set gradients to zero
                act_noise.grad = torch.zeros_like(act_noise)
               
                # forward
                s_c = self.fc(flat(act_noise))
                
                # Append into tensor s_c
                s_c_expanded = torch.ones_like(act_noise)*s_c[0][class_i]
                s_c_with_noise_classes = torch.cat((s_c_with_noise_classes, s_c_expanded))

                torch.autograd.backward(s_c[0][class_i], retain_graph=True)
                
                gradients_with_noise[class_i] = torch.cat((gradients_with_noise[class_i], act_noise.grad)) #, act_noise.grad[0][None,:]))
            
            # Añadimos la salida s_c
            s_c_with_noise = torch.cat((s_c_with_noise, s_c_with_noise_classes[None,:]))

        
        # Calculate the mean of the activations (axis 0)
        mean_noisy_s_c = s_c_with_noise.mean(axis=0)
        
        # Calculatin weights 
        weights = torch.tensor([]).to(device)
        for class_i in range(self.n_classes):
            #mean_noisy_grads = gradients_with_noise[class_i].mean(axis=0)

            # get the dy/dA=exp(s)*ds/dA
            exp_s = torch.exp(mean_noisy_s_c[class_i])
            dsdA = gradients_with_noise[class_i].mean(axis=0)
            dydA = exp_s*dsdA

            # Getting the parameters as a weighted sum of the noisy gradients
            #subweights = self.get_subweights(activations[0], gradients_with_noise[class_i]) #
            subweights = self.get_subweights( activations.mean(axis=0), gradients_with_noise[class_i]) 

            new_weight = (subweights*relu(dydA)).sum(axis=(1,2))

            weights = torch.cat((weights,new_weight[None,:]))

        # Calculate output
        x_mod = activations
        x_mod = flat(x_mod) # flatten
        y = self.fc(x_mod)
        
        return weights, y.mean(axis=0)[None,:]

    
    def get_subweights(self, activations=None, gradients=None):
        assert(activations!=None and activations!=None)
        
        # Numerator 
        numerator = gradients.pow(2).mean(axis=0)

        # Denominator
        ag = activations.sum(axis=[1,2])[:,None,None].expand(-1,7,7) * gradients.pow(3).mean(axis=0)

        denominator = 2 * gradients.pow(2).mean(axis=0) 
        denominator += ag #ag.view(gradients.shape[1], -1).sum(-1, keepdim=True).view(gradients.shape[1], 1, 1)
        denominator = torch.where(denominator != 0.0, denominator, torch.ones_like(denominator))
        
        # Alpha
        alpha = numerator / denominator
                
        return alpha