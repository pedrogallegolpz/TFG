---
title: "Documentación inicial"
subtitle: "Trabajo de fin de grado"
author: ["Pedro Gallego López", "Francisco Luque Sánchez"]
date: "04/03/2020"
toc-own-page: true
titlepage: true
titlepage-background: /home/fluque/.pandoc/backgrounds/ugr.pdf
tables: true
colorlinks: true
urlcolor: 'blue'
script-font-size: \scriptsize
nncode-block-font-size: \scriptsize
---

# Documentación de partida

## Artículos sobre explicabilidad en CNNs

- Survey sobre XAI: <https://arxiv.org/abs/1910.10045> (<https://www.sciencedirect.com/science/article/pii/S1566253519308103>)
- Versiones de Grad-CAM
  - Class Activation Mappings (punto de partida) - [http://cnnlocalization.csail.mit.edu/](http://cnnlocalization.csail.mit.edu/)
  - Grad-CAM (Class Activation Mappings incluyendo el gradiente) - [http://gradcam.cloudcv.org/](http://gradcam.cloudcv.org/)
  - Grad-CAM++ (Mejoras a GradCAM) - [https://arxiv.org/abs/1710.11063](https://arxiv.org/abs/1710.11063)
  - Smooth Grad-CAM++ (Suavizado de GradCAM) - <https://arxiv.org/abs/1908.01224> (incluye esta técnica: <https://arxiv.org/abs/1706.03825>)

- Repositorio de código con implementaciones de los cuatro métodos: [https://github.com/yiskw713/SmoothGradCAMplusplus](https://github.com/yiskw713/SmoothGradCAMplusplus)
- [https://github.com/jacobgil/pytorch-grad-cam/blob/master/pytorch_grad_cam/base_cam.py](https://github.com/jacobgil/pytorch-grad-cam/blob/master/pytorch_grad_cam/base_cam.py)

## Bases de datos
- Segmentación de lesiones:
  - ISIC 2018 (segmentación de lesiones cutáneas): <https://challenge.isic-archive.com/data/>
  - Breast cancer dataset (segmentación de cánceres de mama)
	- <https://www.tamps.cinvestav.mx/~wgomez/downloads.html>
	- <https://scholar.cu.edu.eg/?q=afahmy/pages/dataset>
- Segmentación de cánceres: <https://camelyon17.grand-challenge.org/>
