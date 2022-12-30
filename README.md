# Diagnóstico de imágenes histológicas de próstata utilizando métodos de explicabilidad en Deep Learning
#### Autor: Pedro Gallego López

Keywords: CAM, Grad-CAM, Grad-CAM++, Smooth Grad-CAM++, image-diagnosis, explainability, deep learning, localization.

This thesis analyses the unsupervised localisation capacity of a series of classification models based on convolutional neural networks for cancer detection. For this purpose, a database of histological images of prostate cancer is used, in which several neural models are trained by solving a classification problem. After this training, different explainability techniques based on the generation of activation maps are applied to identify which regions of the image have been useful for determining the presence of cancer. The study tries to discern whether the neural models that achieve a higher accuracy rate in determining whether the tissue is affected by cancer discriminate better the affected region when using activation map generation techniques. In addition, by fixing a classification model, it will be tested whether the activation map generation techniques that are considered more powerful in the literature are more accurate in selecting the region affected by cancer.


![Alt text](https://github.com/pedrogallegolpz/TFG/blob/main/ejemplo_explicado.png "Example where all technics are executed in every model")

### Repository structure
- `src`: This folder contains the code necessary to execute all the functionalities in terms of models and explainability techniques that have been applied in the work.
  -  `CAM`:  hosts the python files concerning the explanability treatise for the models.
      - `cam.py`: definition of two classes. `CAM_model`,  is a type of model used for the CAM technique (the most basic) that allows only one fully connected layer. `CAM`, is a class that incorporates a fully connected three-layer classifier into the base network and generates a model on which the techniques of _Grad-CAM_, _Grad-CAM++_ y _Smooth Grad-CAM++_.
      -  `cam_abstract.py`: is the parent class of the classes that appear in `cam.py`. It defines the methods that the child classes have to define and some methods that are common. 
      -  `utils_cam.py`: functions which call to or are called by `CAM_abstract`: visualization functions, layer modifiers...
  -  `dataset.py`: implements the class which is going to be used to work with the data.
  -  `utils.py`: functionality on CAM models: training, testing, threshold search...
-  `image_diagnosis_CAM.ipynb`: example notebook for use and display of the work. You can open it with Google Colab and try things out by following the instructions. It incorporates functionalities such as information on thresholding, dataset distribution, execution of examples...
-  `prepare_data.ipynb`: notebook used to pre-process the dataset to make it fit our needs.
