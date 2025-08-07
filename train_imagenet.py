import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from nets.resnet50 import ResNet, Bottleneck
from nets.resnet50_2 import ResNet2, Bottleneck2
from nets.resnet50_2_imagenet import ResNet2_imagenet, Bottleneck2_imagenet
from nets.early_stopping import EarlyStopping

# CUDA 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 하이퍼파라미터
BATCH_SIZE = 512
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
MODEL_SAVE_PATH = "./resnet50-mnist.pth"
NUM_WORKERS = 0
CUSTOM_CONV_LAYER_INDEX = 1
MODELS = {
    "mnist-1": ResNet(Bottleneck, [3, 4, 6, 3], num_classes=10, custom_conv_layer_index=CUSTOM_CONV_LAYER_INDEX).to(device),
    "mnist-2": ResNet2(Bottleneck2, [3, 4, 6, 3], num_classes=10, custom_conv_layer_index=CUSTOM_CONV_LAYER_INDEX).to(device),
    "imagenet-2": ResNet2_imagenet(Bottleneck2_imagenet, [3, 4, 6, 3], num_classes=1000, custom_conv_layer_index=CUSTOM_CONV_LAYER_INDEX).to(device)
}
DATAS = {
    "mnist-1": {
        "train": datasets.MNIST(root='data/', train=True, transform=transforms.ToTensor(), download=True),
        "test": datasets.MNIST(root='data/', train=False, transform=transforms.ToTensor(), download=True)
    },
    "mnist-2": {
        "train": datasets.MNIST(root='data/', train=True, transform=transforms.ToTensor(), download=True),
        "test": datasets.MNIST(root='data/', train=False, transform=transforms.ToTensor(), download=True)
    },
    "imagenet-2": {
        "train": datasets.ImageFolder('~/imagenet/')
    }
}
MODEL = "imagenet-2"

# 모델 초기화
model = MODELS[MODEL]

# 손실함수 및 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 학습 데이터셋
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='data/', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

# 테스트 데이터셋
test_dataset = datasets.MNIST(root='data/', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)