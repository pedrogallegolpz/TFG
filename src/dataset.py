from torchvision import datasets

class ImageFolder_and_MaskFolder(datasets.ImageFolder):  
    def __init__(self, folder, transform):
        """
        Parameters
        ----------
        group1 : list of two str [clase, tipo_mascara]: por ejemplo: ['benign','noannotationMask']
        group2 : list of two str [clase, tipo_mascara] 
        """
        super().__init__(folder, transform)
        self.dataset_mask = datasets.ImageFolder(folder+'_mask', transform)
    
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        mask, _ = self.dataset_mask[index]

        return img, mask, label      
    