# Session-6---Assignment-QnA



import torch
import torch.nn as nn

class PointwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False):
        super(PointwiseConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        
    def forward(self, x):
        return self.conv(x)

import torch
import torch.nn as nn

class PointwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=0, bias=False):
        super(PointwiseConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        
    def forward(self, x):
        return self.conv(x)

# Move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PointwiseConv(in_channels=3, out_channels=40).to(device)

import torch
import torch.nn as nn

class PointwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, dilation=2, bias=False):
        super(PointwiseConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=bias)
        
    def forward(self, x):
        return self.conv(x)

# Move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PointwiseConv(in_channels=3, out_channels=40).to(device)

import torch
import torch.nn as nn

class PointwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, dilation=2, bias=False):
        super(PointwiseConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.dilated = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=bias)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.dilated(x)
        return x

# Move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PointwiseConv(in_channels=3, out_channels=40).to(device)

import numpy as np
import cv2
import albumentations as A

# Define the augmentation pipeline
def get_augmentations(mean):
    return A.Compose([
        A.HorizontalFlip(),
        A.ShiftScaleRotate(),
        A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16, min_width=16, fill_value=mean, mask_fill_value=None)
    ])

# Load an example image
img = cv2.imread('example.jpg')

# Make sure the image was successfully loaded
if img is None:
    raise ValueError('Failed to load the image')

# Compute the mean of the image
mean = np.mean(img)

# Apply the augmentation pipeline to the image
augmented_img = get_augmentations(mean=mean)(image=img)['image']

pip install albumentations

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Define the device to be used (if GPU available, use GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the Convolutional Neural Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Convolutional layer (C1)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # Convolutional layer (C2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # Convolutional layer (C3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        # Fully connected layer (C4)
        self.fc = nn.Linear(128 * 8 * 8, 40)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = self.fc(x)
        return x

# Initialize the CNN
net = Net().to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Load the CIFAR-40 dataset
transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.RandomAffine(degrees=0, translate=(0.1,0.1), scale=(0.9, 1.1), shear=None, resample=False, fillcolor=0),
     transforms.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16, min_width=16, fill_value=128, mask_fill_value=None),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
torchvision.datasets.CIFAR40(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR40(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

for epoch in range(100):
  running_loss = 0.0
for i, data in enumerate(trainloader, 0):
  inputs, labels = data
  inputs, labels = inputs.to(device), labels.to(device)


optimizer.zero_grad()
outputs = net(inputs)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
running_loss += loss.item()
print('Epoch %d loss: %.3f' % (epoch + 1, running_loss / (i + 1)))
print('Finished Training')
correct = 0
total = 0
with torch.no_grad():
for data in testloader:
images, labels = data
images, labels = images.to(device), labels.to(device)
outputs = net(images)
_, predicted = torch.max(outputs.data, 1)
total += labels.size(0)
correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
