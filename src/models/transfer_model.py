# Transfer Learning Model for Pneumonia Detection in Chest X-Rays

import torch.nn as nn
from torchvision import models

def get_transfer_model(num_classes=2):
    """
    Se cargó ResNet18 y se adaptó para imágenes en escala de grises.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Se congelaron los pesos del backbone
    for param in model.parameters():
        param.requires_grad = False
        
    # Se modificó la entrada para 1 canal (Grayscale)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # Se reemplazó la capa totalmente conectada
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )
    return model
