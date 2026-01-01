import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from nets.resnet50_1_imagenet import ResNet1_imagenet, Bottleneck1_imagenet
from nets.resnet50_2_tinet import ResNet, Bottleneck
import argparse

parser = argparse.ArgumentParser(description='ResNet Model2 Training')
parser.add_argument('--cusin', type=int, default=1, help='custom convolution layer index')
parser.add_argument('--model', type=int, default=1, help='model number')
args = parser.parse_args()

# Hyperparameters
BATCH_SIZE = 64
NUM_WORKERS = 4
CUSTOM_CONV_LAYER_INDEX = args.cusin
WEIGHT_PATHS = [
    [
        "../weights/cifar10/cifar10_model-1_cusin-1_epoch",
        "../weights/cifar10/cifar10_model-1_cusin-2_epoch",
        "../weights/cifar10/cifar10_model-1_cusin-3_epoch",
        "../weights/cifar10/cifar10_model-1_cusin-4_epoch-80.pth"
    ],
    [
        "../weights/cifar10/cifar10_model-2_cusin-1_epoch-80.pth",
        "../weights/cifar10/cifar10_model-2_cusin-2_epoch-80.pth",
        "../weights/cifar10/cifar10_model-2_cusin-3_epoch-80.pth",
        "../weights/cifar10/cifar10_model-2_cusin-4_epoch-80.pth"
    ]
]
WEIGHT_PATH = WEIGHT_PATHS[args.model - 1][CUSTOM_CONV_LAYER_INDEX - 1]

# CUDA Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load Model
if args.model == 1: 
    model = ResNet1_imagenet(Bottleneck1_imagenet, [3, 4, 6, 3], num_classes=10, custom_conv_layer_index=CUSTOM_CONV_LAYER_INDEX).to(device)
else:
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=10, custom_conv_layer_index=CUSTOM_CONV_LAYER_INDEX).to(device)

# Load Checkpoint
if os.path.isfile(WEIGHT_PATH):
    print(f"Loading weights from '{WEIGHT_PATH}'")
    model.load_state_dict(torch.load(WEIGHT_PATH))
else:
    print(f"No weight file found at '{WEIGHT_PATH}'")
    exit()

# Test Dataset (using test split for final evaluation)
transform = transforms.ToTensor()
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# Evaluation
model.eval()
correct = 0
total = 0

print("Starting Evaluation on Test Set...")
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"✅ Final Test Accuracy (Model 1): {accuracy:.2f}%")
