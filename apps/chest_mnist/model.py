import numpy
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from typing import Any, List
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from typing import Callable
import time
from torch.utils.data.dataloader import default_collate
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
            lr : float = 0.001, momentum : float = 0.9,
            device_str : str = 'cpu',
            x_train_path : str ='/mnt/input/x_train.npy',
            y_train_path : str ='/mnt/input/y_train.npy',
            x_test_path : str ='/mnt/input/x_val.npy',
            y_test_path : str ='/mnt/input/y_val.npy'
        ):

        self.device = torch.device(device_str)

        if x_train_path is None:
            self.training_data = None
        else:
            self.training_data = DataLoader(ChestMNIST(x_train_path, y_train_path), batch_size=256, shuffle=True)
        
        if x_test_path is None:
            self.testing_data = None
        else:
            self.testing_data = DataLoader(ChestMNIST(x_test_path, y_test_path), batch_size=256, shuffle=True)


        self.model = ResNet18(image_channels=1, num_classes=14).to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum) 
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.__parameter_keys = self.model.state_dict().keys()

    #def train_single_epoch(self, log : Callable[str, None]):
    def train_single_epoch(self, log : Callable = None):

        losses = []

        for batch_idx, (inputs, targets) in enumerate(self.training_data):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            batch_start = time.time()
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            targets = targets.to(torch.float32)
            loss = self.loss_fn(outputs, targets)
            
            loss.backward()
            self.optimizer.step()
            batch_end = time.time()
            
            current_loss = loss.cpu().detach().numpy()
            losses.append(current_loss)

            if log:
                log(f'Training batch {batch_idx}/{len(self.training_data)} for {batch_end - batch_start:2.2f} sec , loss: {current_loss}...')

        avg_loss = np.mean(losses)
        return avg_loss


    def get_test_score(self, log, test_data_loader : Any = None) -> Any:
        self.model.eval()
        y_true = torch.zeros(1, 14, dtype=torch.float32).to(self.device)
        y_score = torch.zeros(1, 14).to(self.device)

        if test_data_loader is None:
            test_data_loader = self.testing_data

        with torch.no_grad():
            for inputs, targets in test_data_loader:
                targets = targets.to(torch.float32).to(self.device)
                inputs = inputs.to(self.device)

                logits = self.model(inputs)
                scores = torch.sigmoid(logits)
                
                # use softmax instead of standard normalization
                #outputs = outputs.softmax(dim=-1)
                #predictions = (scores > 0.5).float()
                predictions = scores
                log(f'Scores output: {scores}, Predictions: {predictions}')

                # Concatenate this batch to the complete results
                y_true = torch.cat((y_true, targets), 0)
                y_score = torch.cat((y_score, predictions), 0)

        y_true = y_true.cpu().detach().numpy()
        y_score = y_score.cpu().detach().numpy()
        return y_true, y_score
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_score)
        return precision, recall, f1, y_score, y_true
        auc = 0
        """
        for i in range(y_score.shape[1]):
            try:
                label_auc = roc_auc_score(y_true[:, i], y_score[:, i])
                precision, recall, f1, support = precision_recall_fscore_support(y_true[:, i])
                auc += label_auc
            except ValueError:
                # If all labels are of the same value, an ValueError is thrown
                # Equal to: auc += 0.0
                pass

        return auc / y_score.shape[1]
        """


    def get_weights(self) -> List[numpy.ndarray]:
        return [t.detach().numpy() for t in self.model.state_dict().values()]
    
    def set_weights(self, weights : List[numpy.ndarray]):
        #new_parameters = {k: torch.from_numpy(v)  if v.size > 1 else torch.tensor(v) for k, v in zip(self.__parameter_keys, weights) }
        new_parameters = {k: torch.tensor(v) for k, v in zip(self.__parameter_keys, weights)}
        self.model.load_state_dict(new_parameters)
    
    def save_model(self, path : str):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path : str):
        self.model.load_state_dict(torch.load(path))

