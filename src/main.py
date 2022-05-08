import torch
from utils import load_model, load_data, train_cam_models, test_cam_models

# Reset CUDA cache
torch.cuda.empty_cache()

path_guardado_modelos = 'modelos/'

# List of technics to train
technics = ['cam','gradcam', 'gradcampp', 'smoothgradcampp']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('DEVICE: ', device)

# Read datasets
dataloaders, dataset_sizes = load_data(r'..\..\SICAPv1\299_patch_impar')
print('\n\n\n\n')

"""
# Entrenamos
train_cam_models(path_guardado_modelos, epochs=10)
print('\n\n\n\n')

# Testeamos 
test_cam_models(path_guardado_modelos)
print('\n\n\n\n')
"""

models_dic = {}
for name in technics:
    # Load model
    models_dic[f'{name}'] = load_model(path_guardado_modelos, name, device)


for i in range(10):
    # Cogemos la activación de las capas
    x, act_classes=next(iter(dataloaders['test']))
    x=x.to(device)
    for name in technics:
        act_classes=act_classes.to(device)

        models_dic[f'{name}']['model'].plot_saliency_map(x, act_classes[0], -1)

        print("#"*70)
        print("#"*70)
        print("#"*70)
