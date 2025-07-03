"""This script is used to test the accuracy of the ResNet50 model on the MNIST test dataset."""

import torch
from nets.resnet50 import ResNet,Bottleneck
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision
import cv2
import time
from tqdm import tqdm

# Load model
# Path to the pretrained model
PATH = './logs/resnet50-mnist.pth'
# Ask user for batch size
# Batch_Size = int(input('The number of handwritten font images predicted each times：'))
Batch_Size = 1

# model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=10)
model = ResNet(Bottleneck, [3, 4, 6, 3])

# 모델에 저장된 파라미터(weight, bias)를 로드
# model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
# state_dict = torch.load()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

model = model.cpu()
model.eval()

#Load test dataset
test_dataset = datasets.MNIST(root='data/', train=False,
                                    transform=transforms.ToTensor(), download=True)
test_loader  = DataLoader(dataset=test_dataset, batch_size=Batch_Size, shuffle=False)

train_dataset = datasets.MNIST(root='data/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=Batch_Size, shuffle=True)

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images = images.to('cpu')
        labels = labels.to('cpu')

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

torch.save(model.state_dict(), 'resnet-model.pth')

# # Accuracy evaluation
# correct = 0
# total = 0

# n_dat = 1
# with torch.no_grad():
#     pbar = tqdm(test_loader, total=n_dat, desc="Testing")
#     idx = 0
#     for images, labels in pbar:
#         if (idx == n_dat):
#             break
#         images = images.cpu()
#         labels = labels.cpu()
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         idx += 1
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#         accuracy = 100 * correct / total
#         pbar.set_postfix({'Accuracy (%)': f"{accuracy:.2f}"})

# accuracy = 100 * correct / total
# print(f"✅ Accuracy on the MNIST test set: {accuracy:.2f}%")