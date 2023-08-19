import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Build NN
class MLP(nn.Module):
  def __init__(self, H, W):
    super().__init__()
    
    self.layer = nn.Sequential(
        nn.Linear(H * W, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 19),
        nn.ReLU()
        )
  
  def forward(self, x):
    x = x.view(x.size(0), -1)
    Net_Out = self.layer(x)
    return Net_Out
 
  # Build Train Function
def train(dataloader, tasknn, scorenn, optimizer):
    pbar = tqdm(dataloader, desc=f'Training')
    for batch, (X, y) in enumerate(pbar):
        # X, y = X.to(device), y.to(device)
        # Feedforward
        pred, loss = scorenn.forward(X, y)
        pred, loss = tasknn.forward(X, y, loss)
 
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
# Build Test Function
def test(dataloader, tasknn, scorenn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    tasknn.eval()
    scorenn.eval()
    loss, correct = 0,0
    with torch.no_grad():
        for X, y in dataloader:
            # X, y = X.to(device), y.to(device)
            pred, loss = scorenn.forward(X, y)
            pred, loss = tasknn.forward(X, y, loss)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    loss /= num_batches
    correct /= size
    print(f'Test Accuracy: {(100*correct):>0.1f}%     Loss: {loss:>8f} \n')
 
    return 100 * correct, loss
 
