from cProfile import label
from requests import get
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

## for printing image
import matplotlib.pyplot as plt
import numpy as np
from util import get_accuracy

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Define model class
class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.d1 = nn.Linear(in_features=28*28,out_features=128)
        self.dropout = nn.Dropout(0.2)
        self.d2 = nn.Linear(in_features=128,out_features=10)
    def forward(self,x):
        self.fatten = x.flatten(start_dim=1)
        self.d1_out = self.d1(self.fatten)
        self.relu = F.relu(self.d1_out)
        self.dropout_out = self.dropout(self.relu)
        self.logits = self.d2(self.dropout_out)
        self.out = F.softmax(self.logits,dim=1)
        
        return self.out
        
# load the dataset

## parameter denoting the batch size
BATCH_SIZE = 32

## transformations
transform = transforms.Compose(
    [transforms.ToTensor()])

## download and load training dataset
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=0)



# call model class
model = Model().to(device)

param_dict = dict(
# define model parameters
# lr = 0.01,
loss = nn.CrossEntropyLoss(),
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
)



def train(model,trainloader,param_dict,save=False):
# train the model
    for epoch in range(5):
        total_train_loss =0
        train_acc = 0
        model = model.train()
        for i,(image,label) in enumerate(trainloader):
            image = image.to(device)
            label = label.to(device)
            output = model(image)
            loss = param_dict['loss'](output,label)
            param_dict['optimizer'].zero_grad()
            loss.backward()
            
            param_dict['optimizer'].step()
            
            total_train_loss += loss.detach().item()
            train_acc += get_accuracy(output,label,BATCH_SIZE)
        model.eval()
        print(f"Epoch:{epoch}, Loss:{round(total_train_loss,3)/i}, Accuracy:{round(train_acc/i,3)}")
        
    if save:
        torch.save(model.state_dict(),'./Model_saved/model.pth')