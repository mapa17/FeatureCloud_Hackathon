import numpy
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from typing import List

class ChestMNIST(Dataset):
    def __init__(self, xPath : str, yPath : str):
        self.img = np.load(xPath)
        self.img_labels = np.load(yPath)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img = Image.fromarray(self.img[idx])
        img = self.transform(img)
        label = self.img_labels[idx]
        return img, label



class Block(nn.Module):
    
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x

    
class ResNet18(nn.Module):
    
    def __init__(self, image_channels, num_classes):
        
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        #resnet layers
        self.layer1 = self.__make_layer(64, 64, stride=1)
        self.layer2 = self.__make_layer(64, 128, stride=2)
        self.layer3 = self.__make_layer(128, 256, stride=2)
        self.layer4 = self.__make_layer(256, 512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
    def __make_layer(self, in_channels, out_channels, stride):
        
        identity_downsample = None
        if stride != 1:
            identity_downsample = self.identity_downsample(in_channels, out_channels)
            
        return nn.Sequential(
            Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride), 
            Block(out_channels, out_channels)
        )
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x 
    
    def identity_downsample(self, in_channels, out_channels):
        
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(out_channels)
        )

class ModelTraining():
    def __init__(self,
            lr : float = 0.001, momentum : float =0.9,
            x_train_path : str ='/mnt/input/x_train.npy',
            y_train_path : str ='/mnt/input/y_train.npy',
            x_val_path : str ='/mnt/input/x_val.npy',
            y_val_path : str ='/mnt/input/y_train.npy'
        ):

        self.training_data = DataLoader(ChestMNIST(x_train_path, y_train_path), batch_size=8, shuffle=True)
        self.validation_data = DataLoader(ChestMNIST(x_val_path, y_val_path), batch_size=8, shuffle=True)

        self.model = ResNet18(image_channels=1, num_classes=14)
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum) 
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.__parameter_keys = self.model.state_dict().keys()

    def train_single_epoch(self):
        for inputs, targets in self.training_data:
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            targets = targets.to(torch.float32)
            loss = self.loss_fn(outputs, targets)
            
            loss.backward()
            self.optimizer.step()

    def get_weights(self) -> List[numpy.ndarray]:
        return [t.detach().numpy() for t in self.model.state_dict().values()]
    
    def set_weights(self, weights : List[numpy.ndarray]):
        new_parameters = {k: torch.from_numpy(v)  if v.size > 1 else torch.tensor(v) for k, v in zip(self.__parameter_keys, weights) }
        self.model.load_state_dict(new_parameters)

