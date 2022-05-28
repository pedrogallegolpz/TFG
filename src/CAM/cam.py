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
        
        try:
            if list(originalModel.children())[-1]==originalModel.classifier:
                self.features = nn.ModuleList(list(originalModel.children())[:-1])
            else:
                # Drop FC
                fc_removed, in_features_list = remove_modules_type(originalModel, [nn.Linear, nn.Flatten, nn.AdaptiveAvgPool2d])
                
                self.features = nn.ModuleList(fc_removed)
        except:
            # Si no existe el atributo classifier
            # Drop FC
            fc_removed, in_features_list = remove_modules_type(originalModel, [nn.Linear, nn.Flatten, nn.AdaptiveAvgPool2d])
            
            self.features = nn.ModuleList(fc_removed)

        self.avgPool_CAM = nn.AdaptiveAvgPool2d(output_size=(1))
        
        channels = get_output_last_conv(originalModel)

        self.classifier = nn.Linear(in_features=channels, out_features=D_out, bias=False)
        self.n_classes = D_out
        


    def forward(self, x):
        # utils
        flat = nn.Flatten()

        # forward
        x_mod = self.get_activations(x)
        x_mod = self.avgPool_CAM(x_mod) 
        
        x_mod = flat(x_mod) # flatten
    
        x_mod = self.classifier(x_mod)

        return x_mod
    
    def get_activations(self, x):
        x_mod = x
        for mod in self.features:
            x_mod = mod(x_mod)
        
        return x_mod

    def get_weights(self, technic=None, activations=None, device='cuda'):
        return next(iter(self.classifier.parameters()))
        
    def get_subweights(self, technic=None, activations=None, grad=None):
        return None
       
class CAM(nn.Module, CAM_abstract):   
    name = "CAM_PRO"

    def __init__(self, originalModel, D_out):
        super(CAM, self).__init__()
        
        # Drop FC
        fc_removed, in_features_list = remove_modules_type(originalModel, [nn.Linear, nn.Flatten, nn.AdaptiveAvgPool2d])
        
        try:
            if list(originalModel.children())[-1]==originalModel.classifier:
                self.features = nn.ModuleList(list(originalModel.children())[:-1])
            else:
                self.features = nn.ModuleList(fc_removed)
        except:
            # Si no existe el atributo classifier
            self.features = nn.ModuleList(fc_removed)
           
            
        # Define new FC with out_features=D_out
        self.in_features_to_fc = in_features_list[-1]
        
        exp = int(0.5+math.log(self.in_features_to_fc, 2)/2)
        in_out = [self.in_features_to_fc, 2**exp, 2**exp, D_out]
        
        layers = []
        for i in range(len(in_out)-1):
            in_f = in_out[i]
            out_f = in_out[i+1]
            layers.append(nn.Linear(in_features=in_f, out_features=out_f, bias=True))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=0.5, inplace=False))
          
        
        self.classifier = nn.Sequential(*layers[:-2])
                
        self.n_classes = D_out

        
    def forward(self, x):
        # forward
        x_mod = self.get_activations(x)
        x_mod = self.classify(x_mod) 
        
        return x_mod

    
    def get_activations(self, x):
        x_mod = x
        for mod in self.features:
            x_mod = mod(x_mod)
        
        return x_mod
    
    def classify(self,x):
        # utils
        flat = nn.Flatten()
        
        x_mod = x
        in_features = x_mod.shape[1]*x_mod.shape[2]*x_mod.shape[3]
        if in_features != self.in_features_to_fc:
            # En el caso de mobilenet, por ejemplo. Hacen un average pooling, y no lo meten como capa
            x_mod = nn.functional.adaptive_avg_pool2d(x_mod, (1, 1))
        
        x_mod = flat(x_mod) # flatten
        
        x_mod = self.classifier(x_mod)
        
        return x_mod

    def get_weights(self, technique, activations=None, device='cuda'):
        assert(activations!=None)     
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
                s_c = self.classify(act_noise)
                
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
            relu_dydA = relu(dydA)
            # Getting the parameters as a weighted sum of the noisy gradients
            #subweights = self.get_subweights(activations[0], gradients_with_noise[class_i]) #
            subweights = self.get_subweights(technique = technique, 
                                             activations = activations.mean(axis=0), 
                                             gradients = gradients_with_noise[class_i]) 


            if technique!='gradcam':
                # normalizamos
                
                subweights_thresholding = torch.where(relu_dydA>0, subweights, relu_dydA) # donde relu_dydA es cero, esto será cero
                
                subweights_normalization_constant = subweights_thresholding.sum(axis=(1,2))
                subweights_normalization_constant_processed = torch.where(subweights_normalization_constant != 0.0, 
                                                                          subweights_normalization_constant, 
                                                                          torch.ones_like(subweights_normalization_constant))
                subweights /= subweights_normalization_constant_processed[:,None,None].expand(subweights.shape)

            new_weight = (subweights*relu_dydA).sum(axis=(1,2))
            
            torch.cuda.empty_cache()
            
            weights = torch.cat((weights,new_weight[None,:]))

        return weights

    
    def get_subweights(self, technique, activations=None, gradients=None):
        assert(activations!=None and activations!=None)
        
        if technique!='gradcam':
            # Numerator 
            numerator = gradients.pow(2).mean(axis=0)
    
            # Denominator
            ag = activations.sum(axis=[1,2])[:,None,None].expand(-1, gradients.shape[-2], gradients.shape[-1]) * gradients.pow(3).mean(axis=0)
    
            denominator = 2 * gradients.pow(2).mean(axis=0) 
            denominator += ag #ag.view(gradients.shape[1], -1).sum(-1, keepdim=True).view(gradients.shape[1], 1, 1)
            denominator = torch.where(denominator != 0.0, denominator, torch.ones_like(denominator))
            
            # Alpha
            alpha = numerator / denominator
            
            
        else:
            alpha = (1./torch.ones_like(activations).sum()) * torch.ones_like(activations)
                
        return alpha