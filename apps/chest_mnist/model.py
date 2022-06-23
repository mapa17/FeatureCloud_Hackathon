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
        #self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.outputs = nn.ModuleList([nn.Sequential(nn.Dropout(p=0.2),nn.Linear(in_features=512, out_features=1)) for _ in range(num_classes)])
        
        #resnet layers
        self.layer1 = self.__make_layer(64, 64, stride=1)
        self.layer2 = self.__make_layer(64, 128, stride=2)
        self.layer3 = self.__make_layer(128, 256, stride=2)
        self.layer4 = self.__make_layer(256, 512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = nn.Linear(512, num_classes)

        
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
        #x = self.fc(x)

        results = torch.cat([output(x) for output in self.outputs], axis=1)
        return results
    
    def identity_downsample(self, in_channels, out_channels):
        
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(out_channels)
        )

class ModelTraining():
    def __init__(self,
            lr : float = 0.001, momentum : float = 0.9, batch_size=128,
            device_str : str = 'cpu',
            x_train_path : str ='/mnt/input/x_train.npy',
            y_train_path : str ='/mnt/input/y_train.npy',
            x_test_path : str ='/mnt/input/x_test.npy',
            y_test_path : str ='/mnt/input/y_test.npy',
            x_val_path : str ='/mnt/input/x_val.npy',
            y_val_path : str ='/mnt/input/y_val.npy'
        ):

        self.device = torch.device(device_str)

        if x_train_path is None:
            self.training_data = None
            class_weights = torch.Tensor([1.0])
        else:
            self.training_data = DataLoader(ChestMNIST(x_train_path, y_train_path), batch_size=batch_size, shuffle=True)
            T=torch.cat([x[1] for x in self.training_data])
            class_weights = T.shape[0] / T.sum(axis=0)
            class_weights = 4*np.sqrt(class_weights)
    
        
        if x_test_path is None:
            self.testing_data = None
        else:
            self.testing_data = DataLoader(ChestMNIST(x_test_path, y_test_path), batch_size=batch_size, shuffle=False)
    
        if x_val_path is None:
            self.val_data = None
        else:
            self.val_data = DataLoader(ChestMNIST(x_val_path, y_val_path), batch_size=batch_size, shuffle=False)

        self.model = ResNet18(image_channels=1, num_classes=14).to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum) 
        #self.optimizer = optim.Adam(self.model.parameters()) 
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(class_weights).to(self.device))
        self.__parameter_keys = self.model.state_dict().keys()

    #def train_single_epoch(self, log : Callable[str, None]):
    def train_single_epoch(self, log : Callable = None):

        avg_training_loss = []

        if log:
            #log(f'Training batch {batch_idx}/{len(self.training_data)} for {batch_end - batch_start:2.2f} sec , loss: {current_loss}...')
            log(f'Training ...')

        training_start = time.time()
        for batch_idx, (inputs, targets) in enumerate(self.training_data):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            #batch_start = time.time()

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            targets = targets.to(torch.float32)
            training_loss = self.loss_fn(outputs, targets)
            
            training_loss.backward()
            self.optimizer.step()
            #batch_end = time.time()
            
            current_loss = training_loss.cpu().detach().numpy()
            avg_training_loss.append(current_loss)

            if log:
                #log(f'Training batch {batch_idx}/{len(self.training_data)} for {batch_end - batch_start:2.2f} sec , loss: {current_loss}...')
                pass
        

        log(f'Validation ...')
        avg_val_loss = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.val_data):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                #batch_start = time.time()

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                
                targets = targets.to(torch.float32)
                val_loss = self.loss_fn(outputs, targets)
                
                #batch_end = time.time()
                current_loss = val_loss.cpu().detach().numpy()
                avg_val_loss.append(current_loss)

                if log:
                    #log(f'Validation batch {batch_idx}/{len(self.val_data)} for {batch_end - batch_start:2.2f} sec , loss: {current_loss}...')
                    #log(f'.', sep='')
                    pass

        training_end = time.time()
        avg_training_loss = np.mean(avg_training_loss)
        avg_val_loss = np.mean(avg_val_loss)

        if log:
            log(f'Training Time: {training_end-training_start:2.2f} sec, Avg. Training Loss: {avg_training_loss}, Avg. Val Loss: {avg_val_loss}')
        return avg_training_loss, avg_val_loss


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
                predictions = (scores > 0.5).float()

                # Concatenate this batch to the complete results
                y_true = torch.cat((y_true, targets), 0)
                y_score = torch.cat((y_score, predictions), 0)

        y_true = y_true.cpu().detach().numpy()
        y_score = y_score.cpu().detach().numpy()
        # Remove the first artificial entry
        y_true = y_true[1:]
        y_score = y_score[1:]
        
        # Cannot handle mixture
        #precision, recall, f1, support = precision_recall_fscore_support(y_true, y_score)
        #return precision, recall, f1, y_score, y_true
        auc = 0
        precisions = []
        recalls = []
        f1s = []
        for i in range(y_score.shape[1]):
            try:
                label_auc = roc_auc_score(y_true[:, i], y_score[:, i])
                precision, recall, f1, support = precision_recall_fscore_support(y_true[:, i], y_score[:, i])
                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)
                auc += label_auc
            except ValueError as e:
                if log:
                    log(f'Calculating precision recall for feature {i} failed! {e}')
                pass
        return precisions, recalls, f1s, y_score, y_true


    def get_weights(self) -> List[numpy.ndarray]:
        return [t.cpu().detach().numpy() for t in self.model.state_dict().values()]
    
    def set_weights(self, weights : List[numpy.ndarray]):
        #new_parameters = {k: torch.from_numpy(v)  if v.size > 1 else torch.tensor(v) for k, v in zip(self.__parameter_keys, weights) }
        new_parameters = {k: torch.tensor(v) for k, v in zip(self.__parameter_keys, weights)}
        self.model.load_state_dict(new_parameters)
    
    def save_model(self, path : str):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path : str):
        self.model.load_state_dict(torch.load(path))

