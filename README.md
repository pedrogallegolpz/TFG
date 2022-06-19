# Diagnóstico de imágenes histológicas de próstata utilizando métodos de explicabilidad en Deep Learning
#### Autor: Pedro Gallego López

Keywords: CAM, Grad-CAM, Grad-CAM++, Smooth Grad-CAM++, image-diagnosis, explainability, deep learning, localization.

En este trabajo de fin de grado se analiza la capacidad de localización no supervisada de una serie de modelos de clasificación basados en redes neuronales convolucionales para la detección de cánceres. Para ello, se utiliza una base de datos de imágenes histológicas de cancer de próstata, en la que se entrenan diversos modelos neuronales resolviendo un problema de clasificación. Tras este entrenamiento, se aplican distintas técnicas de explicabilidad basadas en la generación de mapas de activación para identificar qué regiones de la imagen han sido útiles para determinar la presencia de cáncer. El estudio trata de discernir si los modelos neuronales que consiguen una mayor tasa de acierto a la hora de determinar si el tejido está afectado por cancer discriminan mejor la región afectada al utilizar las técnicas de generación de mapas de activación. Además, fijado un modelo de clasificación, se comprobará si las técnicas de generación de mapas de activación que se consideran más potentes en la literatura especializada son más precisos a la hora de seleccionar la región afectada por el cáncer

### Estructura del repositorio
- `src`: en esta carpeta se encuentra el código necesario para ejecutar todas las funcionalidades en cuanto a los modelos y las técnicas de explicabilidad que se han aplicado en el trabajo.
  -  `CAM`:  aloja los archivos de python referentes al tratado de la explicabilidad para los modelos.
    - `cam.py`: contiene dos clases: `CAM_model`, es un tipo de modelo utilizado para la técnica de CAM (la más básica) que solo permite una capa totalmente conectada. `CAM`, es una clase que incorpora a la red base un clasificador de tres capas totalmente conectadas y genera un modelo sobre el que se pueden aplicar las técnicas de _Grad-CAM_, _Grad-CAM++_ y _Smooth Grad-CAM++_.
    -  `cam_abstract.py`: es la clase padre de las clases que aparecen en `cam.py`. Aquí se definen los métodos que tienen que definir las clases hijas y algunos métodos que son comunes.
    -  `utils_cam.py`: contiene funciones que son llamadas o llaman a la clase padre `CAM_abstract`: funciones de visualización, de borrado de capas de red...
  -  `dataset.py`: implementa la clase que se va a usar para trabajar con el conjunto de datos.
  -  `utils.py`: funcionalidad sobre los modelos de CAM: entrenamiento, test, búsqueda de umbrales...
-  `image_diagnosis_CAM.ipynb`: cuaderno de ejemplo de uso y de exposición del trabajo. Se puede abrir con Google Colab y probar las cosas siguiendo las instrucciones. Incorpora funcionalidades como información sobre la umbralización, distribución del dataset, ejecución de ejemplos...
-  `prepare_data.ipynb`: cuaderno utilizado para preprocesar el conjunto de los datos para dejarlo ajustado a nuestras necesidades.
