import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from nets.resnet50_2_tinet import ResNet, Bottleneck
from nets.early_stopping import EarlyStopping
import argparse

parser = argparse.ArgumentParser(description='ResNet Model2 Training')
parser.add_argument('--cusin', type=int, default=1, help='custom convolution layer index')
args = parser.parse_args()

# 하이퍼파라미터
BATCH_SIZE = 128
NUM_EPOCHS = 120
LEARNING_RATE = 0.1
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
NUM_WORKERS = 6
CUSTOM_CONV_LAYER_INDEX = args.cusin

# CUDA 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 모델 초기화
model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=200, custom_conv_layer_index=CUSTOM_CONV_LAYER_INDEX).to(device)

# 학습 데이터셋
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),       # 회전 추가로 다양성 확보
    transforms.RandomCrop(64, padding=8), # 패딩을 늘려 더 과감한 증강
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # 색상 변형
    transforms.ToTensor(),
    transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
])

train_dataset = datasets.ImageFolder("../tiny-imagenet-200/train", transform=transform_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

val_dataset = datasets.ImageFolder("../tiny-imagenet-200/val", transform=transform_test)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, 
                      weight_decay=WEIGHT_DECAY, nesterov=True)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

# 조기 종료 조건 초기화
early_stopping = EarlyStopping(patience=7, delta=0.001)

# 학습 루프
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{NUM_EPOCHS}]")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss = loss.item()
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    
    avg_val_loss = val_loss / len(val_loader)
    val_acc = 100 * correct / total
    print(f"✅ Epoch {epoch+1}: Val Loss = {avg_val_loss:.4f} | Val Acc = {val_acc:.2f}%")

    scheduler.step()

    # 조기 종료 체크 (여기선 train_loss 기반이지만 val_loss가 있으면 교체 가능)
    early_stopping(avg_val_loss)
    if early_stopping.early_stop:
        print(f"⛔ Early stopping at epoch {epoch+1}")
        torch.save(model.state_dict(), f'tinet_model-2_cusin-{CUSTOM_CONV_LAYER_INDEX}_epoch-{epoch+1}.pth')
        break
    elif (epoch + 1) == NUM_EPOCHS:
        torch.save(model.state_dict(), f'tinet_model-2_cusin-{CUSTOM_CONV_LAYER_INDEX}_epoch-{epoch+1}.pth')

# 테스트 정확도 측정
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="Testing"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"✅ Test Accuracy: {accuracy:.2f}%")