"""This script is used to test the accuracy of the ResNet50 model on the MNIST test dataset."""

import torch
from nets.resnet50_1_imagenet import ResNet1_imagenet, Bottleneck1_imagenet
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision
import cv2
import time
from tqdm import tqdm

BATCH_SIZE = 64
CUSTOM_CONV_LAYER_INDEX = 4
NUM_WORKERS = 6
PATH = './weights/tinet_model-1_cusin-4_epoch-73.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet1_imagenet(Bottleneck1_imagenet, [3, 4, 6, 3],custom_conv_layer_index=CUSTOM_CONV_LAYER_INDEX, num_classes=200)
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu'), weights_only=True))
model = model.to(device)
model.eval()

transform = transforms.ToTensor()
train_dataset = datasets.ImageFolder("../tiny-imagenet-200/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

test_dataset = datasets.ImageFolder("../tiny-imagenet-200/test", transform=transform)
test_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

# 테스트 정확도 측정
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing", mininterval=1000000):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        if (idx == 100 or idx == 1000):
            pbar.set_postfix({'Accuracy (%)': f"{accuracy:.2f}"})
            pbar.refresh()

accuracy = 100 * correct / total
print(f"✅ Test Accuracy: {accuracy:.2f}%")
